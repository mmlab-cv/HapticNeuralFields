import os
import time
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
from PIL import Image
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF


class HapticDataLoader(Dataset):
    """
    Like haptic_dataloader_aligned.py but adds a practical mitigation for M << N:

    - We still compute signal windows (N) and pair them to images (M) deterministically.
    - Then, PER SPLIT, we cap how many windows each image contributes:
        max_windows_per_image_train / val / test
      This reduces the "same image seen with hundreds of different targets" problem and
      makes image-conditioning easier to learn (especially when using dvF-dropout).

    Recommended settings when M << N:
      - increase window_stride (e.g., 200-500) OR
      - set max_windows_per_image_train to 3..10
    """

    def __init__(
        self,
        image_path: str = "Digit",
        signals_path: str = "1-RecordedData_Friction",
        chunk_size: int = 1000,
        window_stride: int = 100,
        use_n_materials: int = 0,
        dft_bins_to_keep: int = 100,
        normalize_signals: bool = True,
        split_ratios: dict | None = None,  # e.g. {"train":80,"val":10,"test":10}
        seed: int = 1234,
        train_transform=None,
        eval_transform=None,
        train_image_augmentations=None,
        recon_split: bool = True,
        # NEW: cap windows per image (set None to disable)
        max_windows_per_image_train: int | None = 8,
        max_windows_per_image_val: int | None = None,
        max_windows_per_image_test: int | None = None,
    ):
        super().__init__()
        t0 = time.time()

        self.image_path = image_path
        self.signals_path = signals_path
        self.chunk_size = int(chunk_size)
        self.window_stride = int(window_stride)
        self.use_n_materials = int(use_n_materials)
        self.dft_bins_to_keep = int(dft_bins_to_keep)
        self.normalize_signals = bool(normalize_signals)
        self.recon_split = bool(recon_split)

        self.max_windows_per_image = {
            "train": max_windows_per_image_train,
            "val": max_windows_per_image_val,
            "test": max_windows_per_image_test,
        }

        self.rng = random.Random(seed)

        # --- transforms ---
        self.train_transform = train_transform or transforms.Compose([
            transforms.RandomCrop(480),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.eval_transform = eval_transform or transforms.Compose([
            transforms.CenterCrop(480),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.train_image_augmentations = train_image_augmentations or [
            (0, False),
        ]

        if split_ratios is None:
            split_ratios = {"train": 80, "val": 10, "test": 10}
        self.split_ratios = split_ratios
        assert sum(self.split_ratios.values()) == 100, "split_ratios must sum to 100"

        self.samples = []
        self.split_to_indices = defaultdict(list)

        self.material_to_accel_orig = {}

        # Global min/max for normalization
        self._min_force = float("inf")
        self._max_force = float("-inf")
        self._min_speed = float("inf")
        self._max_speed = float("-inf")
        self._min_accel_time = float("inf")
        self._max_accel_time = float("-inf")
        self._min_dft = float("inf")
        self._max_dft = float("-inf")

        num_materials = 0
        for material in sorted(os.listdir(self.image_path)):
            mat_img_dir = os.path.join(self.image_path, material)
            mat_sig_dir = os.path.join(self.signals_path, material)
            if not os.path.isdir(mat_sig_dir) or not os.path.isdir(mat_img_dir):
                continue

            num_materials += 1
            if self.use_n_materials and num_materials > self.use_n_materials:
                break

            print(f"Processing material {material} ({num_materials})...")

            accel_path, force_path, position_path = self._find_signal_files(mat_sig_dir)
            if accel_path is None or force_path is None or position_path is None:
                print(f"  Skipping {material}: missing Accel/Force/Position.")
                continue

            img_paths = self._list_images(mat_img_dir)
            if len(img_paths) == 0:
                print(f"  Skipping {material}: no images found.")
                continue

            accel_dft_chunks, accel_time_chunks, force_chunks, speed_chunks, direction_chunks, accel_orig = \
                self._load_and_chunk_signals(accel_path, force_path, position_path)

            N = accel_dft_chunks.shape[0]
            self.material_to_accel_orig[material] = accel_orig

            self._update_global_ranges(accel_dft_chunks, accel_time_chunks, force_chunks, speed_chunks)

            # Pair each window j -> image idx
            paired_img = self._pair_images_to_chunks(img_paths, N)  # list[str], len N

            # Compute signal split indices
            sig_idxs = self._blocked_split_indices(N)
            if self.recon_split:
                sig_idxs["recon"] = list(range(N))

            # NEW: per split, cap windows per image
            capped_sig_idxs = {}
            for split_name in ["train", "val", "test"]:
                cap = self.max_windows_per_image[split_name]
                capped_sig_idxs[split_name] = self._cap_windows_per_image(
                    sig_idxs[split_name], paired_img, cap, seed=self.rng.randint(0, 10**9)
                )
            if self.recon_split:
                capped_sig_idxs["recon"] = sig_idxs["recon"]  # keep full

            # Create samples
            for split_name in ["train", "val", "test"] + (["recon"] if self.recon_split else []):
                aug_list = self.train_image_augmentations if split_name == "train" else [(0, False)]
                for j in capped_sig_idxs[split_name]:
                    img_path = paired_img[j]
                    for aug_idx in range(len(aug_list)):
                        self._append_sample(
                            split_name=split_name,
                            material=material,
                            j=j,
                            accel_dft_chunks=accel_dft_chunks,
                            accel_time_chunks=accel_time_chunks,
                            force_chunks=force_chunks,
                            speed_chunks=speed_chunks,
                            direction_chunks=direction_chunks,
                            img_path=img_path,
                            aug_idx=aug_idx,
                        )

        self.norm_info = {
            "min_force": self._min_force, "max_force": self._max_force,
            "min_speed": self._min_speed, "max_speed": self._max_speed,
            "min_accel": self._min_accel_time, "max_accel": self._max_accel_time,
            "min_dft": self._min_dft, "max_dft": self._max_dft,
        }

        if self.normalize_signals:
            self._normalize_samples_inplace()

        print(f"Built dataset with {len(self.samples)} samples in {time.time() - t0:.2f}s")
        for s in ["train", "val", "test"] + (["recon"] if self.recon_split else []):
            print(f"  {s}: {len(self.split_to_indices[s])} samples")

    # -------------------------
    # public
    # -------------------------
    def get_subset(self, split: str) -> Subset:
        if split not in self.split_to_indices:
            raise ValueError(f"Unknown split '{split}'. Available: {list(self.split_to_indices.keys())}")
        return Subset(self, self.split_to_indices[split])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        accel_chunk = s["accel_dft"]
        accel_chunk_time = s["accel_time"]
        force_chunk = s["force"]
        speed_chunk = s["speed"]
        direction_chunk = s["direction"]
        material = s["material"]
        split = s["split"]

        accel_orig = self.material_to_accel_orig[material]

        image = Image.open(s["img_path"]).convert("RGB")
        rotation_angle, do_flip = (self.train_image_augmentations if split == "train" else [(0, False)])[s["aug_idx"]]
        if do_flip:
            image = TF.hflip(image)
        if rotation_angle:
            image = TF.rotate(image, rotation_angle)

        image = self.train_transform(image) if split == "train" else self.eval_transform(image)

        return (
            accel_chunk,
            accel_chunk_time,
            force_chunk,
            speed_chunk,
            direction_chunk,
            image,
            material,
            accel_orig,
            self.norm_info,
        )

    # -------------------------
    # internals
    # -------------------------
    def _cap_windows_per_image(self, window_indices: list[int], paired_img: list[str], cap: int | None, seed: int) -> list[int]:
        """Given a list of window indices for a split, keep at most 'cap' windows per image path."""
        if cap is None:
            return window_indices

        img_to_js = defaultdict(list)
        for j in window_indices:
            img_to_js[paired_img[j]].append(j)

        rng = random.Random(seed)
        kept = []
        for img_path, js in img_to_js.items():
            if len(js) <= cap:
                kept.extend(js)
            else:
                # keep a deterministic, uniformly spread subset
                js_sorted = sorted(js)
                # pick cap indices spread across time
                if cap == 1:
                    kept.append(js_sorted[len(js_sorted)//2])
                else:
                    # evenly spaced picks
                    picks = [js_sorted[int(round(k*(len(js_sorted)-1)/(cap-1)))] for k in range(cap)]
                    kept.extend(picks)

        kept = sorted(kept)
        return kept

    def _find_signal_files(self, mat_sig_dir: str):
        accel_path = force_path = position_path = None
        for item in os.listdir(mat_sig_dir):
            head = item.split("_")[0]
            p = os.path.join(mat_sig_dir, item)
            if head == "Accel":
                accel_path = p
            elif head == "Force":
                force_path = p
            elif head == "Position":
                position_path = p
        return accel_path, force_path, position_path

    def _list_images(self, mat_img_dir: str):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
        files = [f for f in os.listdir(mat_img_dir) if f.lower().endswith(exts)]
        files.sort()
        return [os.path.join(mat_img_dir, f) for f in files]

    def _load_and_chunk_signals(self, accel_path: str, force_path: str, position_path: str):
        # ACCEL
        aroot = ET.parse(accel_path).getroot()
        accel_np = np.array([float(v.text) for v in aroot.findall("Accel/value")], dtype=np.float32)
        accel_np = self.lowpass_filter(accel_np, cutoff=1000.0, fs=10000.0, order=5)

        accel_orig = torch.from_numpy(np.ascontiguousarray(accel_np[500:-500])).float()
        accel = torch.from_numpy(np.ascontiguousarray(accel_np)).float()

        spec = torchaudio.transforms.Spectrogram(
            window_fn=torch.hann_window,
            n_fft=1000,
            hop_length=self.window_stride,
            win_length=self.chunk_size,
            power=1,
            onesided=True,
            center=False,
        )
        stft = spec(accel)
        accel_dft_chunks = stft.transpose(0, 1)[:, : self.dft_bins_to_keep].contiguous()

        accel_time_chunks = torch.stack(
            [accel[i:i + self.chunk_size] for i in range(0, accel.numel() - self.chunk_size + 1, self.window_stride)],
            dim=0,
        ).contiguous()

        # FORCE
        froot = ET.parse(force_path).getroot()
        force_np = np.array([float(v.text) for v in froot.findall("ForceNormal/value")], dtype=np.float32)
        force_np = self.lowpass_filter(force_np, cutoff=20.0, fs=10000.0, order=5)
        force = torch.from_numpy(np.ascontiguousarray(force_np)).float()
        force_chunks = torch.stack(
            [force[i:i + self.chunk_size] for i in range(0, force.numel() - self.chunk_size + 1, self.window_stride)],
            dim=0,
        ).contiguous()

        # SPEED
        proot = ET.parse(position_path).getroot()
        speed_np = np.array([float(v.text) for v in proot.findall("Speed/value")], dtype=np.float32)
        speed_np = self.lowpass_filter(speed_np, cutoff=20.0, fs=10000.0, order=5)
        speed = torch.from_numpy(np.ascontiguousarray(speed_np)).float()
        speed_chunks = torch.stack(
            [speed[i:i + self.chunk_size] for i in range(0, speed.numel() - self.chunk_size + 1, self.window_stride)],
            dim=0,
        ).contiguous()

        # DIRECTION
        pos_x_np = np.array([float(v.text) for v in proot.findall("Position_x/value")], dtype=np.float32)
        pos_y_np = np.array([float(v.text) for v in proot.findall("Position_y/value")], dtype=np.float32)
        pos_x_np = self.lowpass_filter(pos_x_np, cutoff=20.0, fs=10000.0, order=5)
        pos_y_np = self.lowpass_filter(pos_y_np, cutoff=20.0, fs=10000.0, order=5)

        dx = np.diff(pos_x_np, prepend=pos_x_np[0]).astype(np.float32)
        dy = np.diff(pos_y_np, prepend=pos_y_np[0]).astype(np.float32)
        ang = np.arctan2(dy, dx).astype(np.float32)
        dir_global = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)

        dir_global_t = torch.from_numpy(np.ascontiguousarray(dir_global)).float()
        direction_chunks = torch.stack(
            [dir_global_t[i:i + self.chunk_size] for i in range(0, dir_global_t.shape[0] - self.chunk_size + 1, self.window_stride)],
            dim=0,
        ).contiguous()

        N = accel_dft_chunks.shape[0]
        assert N == accel_time_chunks.shape[0] == force_chunks.shape[0] == speed_chunks.shape[0] == direction_chunks.shape[0], \
            "Signal chunk count mismatch"

        return accel_dft_chunks, accel_time_chunks, force_chunks, speed_chunks, direction_chunks, accel_orig

    def _pair_images_to_chunks(self, img_paths: list[str], N: int) -> list[str]:
        """Map each chunk j to an image path. This is still many-to-one if M<<N; use caps to mitigate."""
        M = len(img_paths)
        paired = []
        for j in range(N):
            img_idx = int(j * M / N)
            if img_idx >= M:
                img_idx = M - 1
            paired.append(img_paths[img_idx])
        return paired

    def _blocked_split_indices(self, N: int) -> dict:
        ratios = self.split_ratios
        counts = np.array([ratios["train"], ratios["val"], ratios["test"]], dtype=int)
        g = np.gcd.reduce(counts) if np.any(counts > 0) else 1
        tr, va, te = (counts // max(g, 1)).tolist()
        block = tr + va + te

        sig_idxs = {"train": [], "val": [], "test": []}
        for i in range(N):
            r = i % block
            if r < tr:
                sig_idxs["train"].append(i)
            elif r < tr + va:
                sig_idxs["val"].append(i)
            else:
                sig_idxs["test"].append(i)
        return sig_idxs

    def _append_sample(
        self,
        split_name: str,
        material: str,
        j: int,
        accel_dft_chunks: torch.Tensor,
        accel_time_chunks: torch.Tensor,
        force_chunks: torch.Tensor,
        speed_chunks: torch.Tensor,
        direction_chunks: torch.Tensor,
        img_path: str,
        aug_idx: int,
    ):
        self.samples.append({
            "split": split_name,
            "material": material,
            "j": int(j),
            "accel_dft": accel_dft_chunks[j],
            "accel_time": accel_time_chunks[j],
            "force": force_chunks[j],
            "speed": speed_chunks[j],
            "direction": direction_chunks[j],
            "img_path": img_path,
            "aug_idx": int(aug_idx),
        })
        self.split_to_indices[split_name].append(len(self.samples) - 1)

    def _update_global_ranges(self, accel_dft_chunks, accel_time_chunks, force_chunks, speed_chunks):
        self._min_dft = min(self._min_dft, float(accel_dft_chunks.min().item()))
        self._max_dft = max(self._max_dft, float(accel_dft_chunks.max().item()))
        self._min_accel_time = min(self._min_accel_time, float(accel_time_chunks.min().item()))
        self._max_accel_time = max(self._max_accel_time, float(accel_time_chunks.max().item()))
        self._min_force = min(self._min_force, float(force_chunks.min().item()))
        self._max_force = max(self._max_force, float(force_chunks.max().item()))
        self._min_speed = min(self._min_speed, float(speed_chunks.min().item()))
        self._max_speed = max(self._max_speed, float(speed_chunks.max().item()))

    def _normalize_samples_inplace(self):
        eps = 1e-8
        f_den = (self._max_force - self._min_force) + eps
        s_den = (self._max_speed - self._min_speed) + eps
        a_den = (self._max_accel_time - self._min_accel_time) + eps

        for s in self.samples:
            s["force"] = (s["force"] - self._min_force) / f_den
            s["speed"] = (s["speed"] - self._min_speed) / s_den
            s["accel_time"] = (s["accel_time"] - self._min_accel_time) / a_den

        print("Input normalization applied using final global min/max values.")

    @staticmethod
    def lowpass_filter(data, cutoff=20.0, fs=1000.0, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, cutoff / nyq, btype="low", analog=False)
        return filtfilt(b, a, data)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = HapticDataLoader(
        use_n_materials=1,
        window_stride=100,
        max_windows_per_image_train=5,  # try 3..10
        max_windows_per_image_val=None,
        max_windows_per_image_test=None,
    )
    dl = DataLoader(ds.get_subset("train"), batch_size=4, shuffle=True)
    batch = next(iter(dl))
    print("Batch OK. accel_dft:", batch[0].shape, "image:", batch[5].shape)
