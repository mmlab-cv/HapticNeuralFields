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
    Loads paired (signal window j -> image frame j) samples.

    Key fixes vs the previous version:
      - No cartesian product between signal windows and images.
      - Images are paired to signal windows BEFORE splitting (split is driven by signal indices).
      - Train transforms are random; val/test are deterministic.
      - Force/speed/direction chunks are stacked consistently as tensors.
      - Optional normalization is applied ONCE after global min/max are known.

    Returns per sample:
        accel_dft_chunk, accel_time_chunk, force_chunk, speed_chunk, direction_chunk, image_tensor, material, accel_orig, norm_info
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
        split_ratios: dict | None = None,  # e.g. {"train": 80, "val": 10, "test": 10}
        seed: int = 1234,
        train_transform=None,
        eval_transform=None,
        train_image_augmentations=None,
        recon_split: bool = True,
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

        self.rng = random.Random(seed)

        # --- transforms ---
        # Keep transforms simple: augmentations controlled by train_image_augmentations + train_transform
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

        # Train-time geometric augments (applied deterministically based on aug_idx stored in samples)
        self.train_image_augmentations = train_image_augmentations or [
            (0, False),   # (rotation_degrees, horizontal_flip)
            # (90, False),
            # (180, False),
            # (270, False),
            # (0, True),
        ]

        # --- splitting ---
        if split_ratios is None:
            split_ratios = {"train": 80, "val": 10, "test": 10}
        self.split_ratios = split_ratios
        assert sum(self.split_ratios.values()) == 100, "split_ratios must sum to 100"

        # --- master storage ---
        self.samples = []  # list[dict]
        self.split_to_indices = defaultdict(list)  # split -> indices in self.samples

        # Store accel_orig once per material to avoid duplicating large arrays
        self.material_to_accel_orig = {}

        # Track global min/max (for optional normalization)
        self._min_force = float("inf")
        self._max_force = float("-inf")
        self._min_speed = float("inf")
        self._max_speed = float("-inf")
        self._min_accel_time = float("inf")
        self._max_accel_time = float("-inf")
        self._min_dft = float("inf")
        self._max_dft = float("-inf")

        # --- build dataset ---
        num_materials = 0
        for material in sorted(os.listdir(self.image_path)):
            mat_img_dir = os.path.join(self.image_path, material)
            mat_sig_dir = os.path.join(self.signals_path, material)
            if not os.path.isdir(mat_sig_dir) or not os.path.isdir(mat_img_dir):
                continue

            num_materials += 1
            if self.use_n_materials and num_materials > self.use_n_materials:
                print(f"Reached limit of {self.use_n_materials} materials. Stopping.")
                break

            print(f"Processing material {material} ({num_materials})...")

            accel_path, force_path, position_path = self._find_signal_files(mat_sig_dir)
            if accel_path is None or force_path is None or position_path is None:
                print(f"  Skipping {material}: missing one of Accel/Force/Position files.")
                continue

            # --- load images for this material (time-ordered as best-effort via sorted filenames) ---
            img_paths = self._list_images(mat_img_dir)
            if len(img_paths) == 0:
                print(f"  Skipping {material}: no images found.")
                continue

            # --- load + chunk signals ---
            accel_dft_chunks, accel_time_chunks, force_chunks, speed_chunks, direction_chunks, accel_orig = \
                self._load_and_chunk_signals(accel_path, force_path, position_path)

            N = accel_dft_chunks.shape[0]
            assert N > 0, f"No chunks produced for {material}"

            # Store accel_orig once
            self.material_to_accel_orig[material] = accel_orig  # tensor

            # Update global ranges (raw, before normalization)
            self._update_global_ranges(
                accel_dft_chunks=accel_dft_chunks,
                accel_time_chunks=accel_time_chunks,
                force_chunks=force_chunks,
                speed_chunks=speed_chunks,
            )

            # --- pair each chunk index j to exactly one image (aligned pairing, no shuffle) ---
            paired_img = self._pair_images_to_chunks(img_paths, N)

            # --- make signal splits (blocked) ---
            sig_idxs = self._blocked_split_indices(N)

            # optional recon split uses all indices
            if self.recon_split:
                sig_idxs["recon"] = list(range(N))

            # --- create samples (NO cartesian product) ---
            for split_name in ["train", "val", "test"] + (["recon"] if self.recon_split else []):
                aug_list = self.train_image_augmentations if split_name == "train" else [(0, False)]
                for j in sig_idxs[split_name]:
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

        print(f"Built raw dataset with {len(self.samples)} samples from {num_materials} materials in {time.time() - t0:.2f}s")

        # --- apply normalization ONCE (after global min/max are known) ---
        self.norm_info = {
            "min_force": self._min_force, "max_force": self._max_force,
            "min_speed": self._min_speed, "max_speed": self._max_speed,
            "min_accel": self._min_accel_time, "max_accel": self._max_accel_time,
            "min_dft": self._min_dft, "max_dft": self._max_dft,
        }
        if self.normalize_signals:
            self._normalize_samples_inplace()

        for s in ["train", "val", "test"] + (["recon"] if self.recon_split else []):
            print(f"  {s}: {len(self.split_to_indices[s])} samples")

    # -------------------------
    # public helpers
    # -------------------------
    def get_subset(self, split: str) -> Subset:
        if split not in self.split_to_indices:
            raise ValueError(f"Unknown split '{split}'. Available: {list(self.split_to_indices.keys())}")
        return Subset(self, self.split_to_indices[split])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        accel_chunk = s["accel_dft"]          # (bins,)
        accel_chunk_time = s["accel_time"]    # (T,)
        force_chunk = s["force"]              # (T,)
        speed_chunk = s["speed"]              # (T,)
        direction_chunk = s["direction"]      # (T, 2)
        material = s["material"]
        split = s["split"]

        # Retrieve accel_orig (once per material)
        accel_orig = self.material_to_accel_orig[material]  # (L,)

        # Load and transform image
        image = Image.open(s["img_path"]).convert("RGB")

        # Apply stored train-time geometric aug
        rotation_angle, do_flip = (self.train_image_augmentations if split == "train" else [(0, False)])[s["aug_idx"]]
        if do_flip:
            image = TF.hflip(image)
        if rotation_angle:
            image = TF.rotate(image, rotation_angle)

        # Apply transform (train random, eval deterministic)
        if split == "train":
            image = self.train_transform(image)
        else:
            image = self.eval_transform(image)

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
    # internal: dataset building
    # -------------------------
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
        # --- ACCEL ---
        aroot = ET.parse(accel_path).getroot()
        accel_np = np.array([float(v.text) for v in aroot.findall("Accel/value")], dtype=np.float32)
        accel_np = self.lowpass_filter(accel_np, cutoff=1000.0, fs=10000.0, order=5)

        # original (for recon / debug) - keep as tensor
        accel_orig = torch.from_numpy(np.ascontiguousarray(accel_np[500:-500])).float()

        accel = torch.from_numpy(np.ascontiguousarray(accel_np)).float()

        # Spectrogram
        spec = torchaudio.transforms.Spectrogram(
            window_fn=torch.hann_window,
            n_fft=1000,
            hop_length=self.window_stride,
            win_length=self.chunk_size,
            power=1,
            onesided=True,
            center=False,
        )
        stft = spec(accel)  # (freq, frames)
        accel_dft_chunks = stft.transpose(0, 1)[:, : self.dft_bins_to_keep].contiguous()  # (N, bins)

        # Temporal chunks
        accel_time_chunks = torch.stack(
            [accel[i : i + self.chunk_size] for i in range(0, accel.numel() - self.chunk_size + 1, self.window_stride)],
            dim=0,
        ).contiguous()  # (N, T)

        # --- FORCE ---
        froot = ET.parse(force_path).getroot()
        force_np = np.array([float(v.text) for v in froot.findall("ForceNormal/value")], dtype=np.float32)
        force_np = self.lowpass_filter(force_np, cutoff=20.0, fs=10000.0, order=5)
        force = torch.from_numpy(np.ascontiguousarray(force_np)).float()
        force_chunks = torch.stack(
            [force[i : i + self.chunk_size] for i in range(0, force.numel() - self.chunk_size + 1, self.window_stride)],
            dim=0,
        ).contiguous()

        # --- SPEED ---
        proot = ET.parse(position_path).getroot()
        speed_np = np.array([float(v.text) for v in proot.findall("Speed/value")], dtype=np.float32)
        speed_np = self.lowpass_filter(speed_np, cutoff=20.0, fs=10000.0, order=5)
        speed = torch.from_numpy(np.ascontiguousarray(speed_np)).float()
        speed_chunks = torch.stack(
            [speed[i : i + self.chunk_size] for i in range(0, speed.numel() - self.chunk_size + 1, self.window_stride)],
            dim=0,
        ).contiguous()

        # --- DIRECTION (from Position_x/y) ---
        pos_x_np = np.array([float(v.text) for v in proot.findall("Position_x/value")], dtype=np.float32)
        pos_y_np = np.array([float(v.text) for v in proot.findall("Position_y/value")], dtype=np.float32)
        pos_x_np = self.lowpass_filter(pos_x_np, cutoff=20.0, fs=10000.0, order=5)
        pos_y_np = self.lowpass_filter(pos_y_np, cutoff=20.0, fs=10000.0, order=5)

        dx = np.diff(pos_x_np, prepend=0.0).astype(np.float32)
        dy = np.diff(pos_y_np, prepend=0.0).astype(np.float32)
        ang = np.arctan2(dy, dx).astype(np.float32)
        dir_global = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)  # (L, 2)

        dir_global_t = torch.from_numpy(np.ascontiguousarray(dir_global)).float()
        direction_chunks = torch.stack(
            [dir_global_t[i : i + self.chunk_size] for i in range(0, dir_global_t.shape[0] - self.chunk_size + 1, self.window_stride)],
            dim=0,
        ).contiguous()  # (N, T, 2)

        # sanity
        N = accel_dft_chunks.shape[0]
        assert (
            N == accel_time_chunks.shape[0] == force_chunks.shape[0] == speed_chunks.shape[0] == direction_chunks.shape[0]
        ), "Signal chunk count mismatch"

        return accel_dft_chunks, accel_time_chunks, force_chunks, speed_chunks, direction_chunks, accel_orig

    def _pair_images_to_chunks(self, img_paths: list[str], N: int) -> list[str]:
        """Map each chunk index j in [0, N-1] to an image path in [0, M-1] in a time-consistent way."""
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
        # DFT
        self._min_dft = min(self._min_dft, float(accel_dft_chunks.min().item()))
        self._max_dft = max(self._max_dft, float(accel_dft_chunks.max().item()))
        # time accel
        self._min_accel_time = min(self._min_accel_time, float(accel_time_chunks.min().item()))
        self._max_accel_time = max(self._max_accel_time, float(accel_time_chunks.max().item()))
        # force
        self._min_force = min(self._min_force, float(force_chunks.min().item()))
        self._max_force = max(self._max_force, float(force_chunks.max().item()))
        # speed
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

        print("Input normalization applied ONCE using final global min/max values.")

    # -------------------------
    # signal processing
    # -------------------------
    def lowpass_filter(self, data, cutoff=20.0, fs=1000.0, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, cutoff / nyq, btype="low", analog=False)
        return filtfilt(b, a, data)


if __name__ == "__main__":
    # quick smoke test (requires your dataset folders)
    from torch.utils.data import DataLoader
    ds = HapticDataLoader(use_n_materials=1)
    dl = DataLoader(ds.get_subset("train"), batch_size=2, shuffle=True)
    batch = next(iter(dl))
    print("Batch shapes:")
    print("  accel_dft:", batch[0].shape)
    print("  accel_time:", batch[1].shape)
    print("  force:", batch[2].shape)
    print("  speed:", batch[3].shape)
    print("  direction:", batch[4].shape)
    print("  image:", batch[5].shape)
    print("  material:", batch[6])
