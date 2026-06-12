import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import librosa
import numpy as np
import torch
import torchaudio
import xml.etree.ElementTree as ET
from PIL import Image
from scipy.signal import butter, filtfilt
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.model import HNF


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an HNF spectrum from a material image and a novel action."
    )
    parser.add_argument("--action", default="Scratch_LeftRight_Strong")
    parser.add_argument("--image", default="Digit/BubbleEnvelope/00487.jpg")
    parser.add_argument("--checkpoint", help="Defaults to the newest local best_model.pth.")
    parser.add_argument("--actions-dir", default="novel_actions")
    parser.add_argument("--norm-stats", default="norm_stats.json")
    parser.add_argument("--output-dir", default="generated_novel_actions")
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-rotation", type=float, default=0.0)
    parser.add_argument("--direction-rotation", type=float, default=0.0)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--dft-cutoff-bins", type=int, default=100)
    parser.add_argument("--m-dim", type=int, default=512)
    parser.add_argument("--n-freq", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--sampling-rate", type=int, default=10000)
    parser.add_argument("--griffin-lim-iters", type=int, default=128)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    return parser.parse_args()


def resolve_device(value):
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_latest_checkpoint(output_dir=Path("output")):
    best = list(output_dir.glob("*/checkpoints/best_model.pth"))
    candidates = best or list(output_dir.glob("*/checkpoints/final_model.pth"))
    if not candidates:
        raise FileNotFoundError(
            "No local checkpoint found. Finish a training run or pass --checkpoint."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_xml_series(path, xpath):
    root = ET.parse(path).getroot()
    values = [float(value.text) for value in root.findall(xpath)]
    if not values:
        raise ValueError(f"No values matching {xpath!r} in {path}")
    return np.asarray(values, dtype=np.float32)


def lowpass_filter(values, cutoff, fs=10000.0, order=5):
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype="low", analog=False)
    return filtfilt(b, a, values)


def chunk_1d(values, window, step):
    if values.shape[0] < window:
        values = np.pad(values, (0, window - values.shape[0]), mode="edge")
    tensor = torch.from_numpy(np.ascontiguousarray(values)).float()
    return torch.stack(
        [tensor[i:i + window] for i in range(0, tensor.numel() - window + 1, step)]
    )


def chunk_direction(pos_x, pos_y, window, step):
    if pos_x.shape[0] < window:
        padding = window - pos_x.shape[0]
        pos_x = np.pad(pos_x, (0, padding), mode="edge")
        pos_y = np.pad(pos_y, (0, padding), mode="edge")

    dx = np.diff(pos_x, prepend=pos_x[0]).astype(np.float32)
    dy = np.diff(pos_y, prepend=pos_y[0]).astype(np.float32)
    angle = np.arctan2(dy, dx).astype(np.float32)
    direction = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    tensor = torch.from_numpy(np.ascontiguousarray(direction)).float()
    return torch.stack(
        [tensor[i:i + window] for i in range(0, tensor.shape[0] - window + 1, step)]
    )


def rotate_direction(direction, degrees):
    if degrees == 0:
        return direction
    radians = np.deg2rad(degrees)
    cos_angle = float(np.cos(radians))
    sin_angle = float(np.sin(radians))
    x, y = direction[..., 0], direction[..., 1]
    return torch.stack(
        [cos_angle * x - sin_angle * y, sin_angle * x + cos_angle * y], dim=-1
    )


def normalize_inputs(force, speed, stats_path):
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalization statistics not found: {stats_path}. Run training first."
        )
    with stats_path.open(encoding="utf-8") as file:
        stats = json.load(file)

    eps = 1e-8
    force = (force - stats["min_force"]) / (
        stats["max_force"] - stats["min_force"] + eps
    )
    speed = (speed - stats["min_speed"]) / (
        stats["max_speed"] - stats["min_speed"] + eps
    )
    return force, speed


def load_action(actions_dir, action, window, step):
    paths = {
        kind: actions_dir / f"{kind}_{action}.xml"
        for kind in ("Force", "Speed", "Position")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        available = sorted(
            path.stem.removeprefix("Position_")
            for path in actions_dir.glob("Position_*.xml")
        )
        raise FileNotFoundError(
            f"Missing action files: {missing}. Available actions: {', '.join(available)}"
        )

    force = lowpass_filter(load_xml_series(paths["Force"], "ForceNormal/value"), 20.0)
    speed = lowpass_filter(load_xml_series(paths["Speed"], "Speed/value"), 20.0)
    pos_x = lowpass_filter(load_xml_series(paths["Position"], "Position_x/value"), 20.0)
    pos_y = lowpass_filter(load_xml_series(paths["Position"], "Position_y/value"), 20.0)

    force_chunks = chunk_1d(force, window, step)
    speed_chunks = chunk_1d(speed, window, step)
    direction_chunks = chunk_direction(pos_x, pos_y, window, step)
    if not (len(force_chunks) == len(speed_chunks) == len(direction_chunks)):
        raise ValueError("Force, speed, and direction produced different chunk counts.")
    return force_chunks, speed_chunks, direction_chunks


def load_image(path, rotation):
    transform = transforms.Compose([
        transforms.CenterCrop(480),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])
    image = transform(Image.open(path).convert("RGB"))
    return TF.rotate(image, rotation) if rotation else image


def predict_spectra(model, image, direction, speed, force, args, device):
    spectra = []
    with torch.inference_mode():
        for start in range(0, len(force), args.inference_batch_size):
            end = min(start + args.inference_batch_size, len(force))
            batch_size = end - start
            image_batch = image.expand(batch_size, -1, -1, -1)
            acceleration = model.render(
                image_batch,
                direction[start:end].to(device),
                speed[start:end].to(device),
                force[start:end].to(device),
                n_samples=args.num_bins,
                stratified=False,
            )
            acceleration = acceleration - acceleration.mean(dim=-1, keepdim=True)
            spectrum = torchaudio.functional.spectrogram(
                acceleration,
                pad=0,
                window=torch.hann_window(acceleration.shape[-1], device=device),
                n_fft=acceleration.shape[-1],
                hop_length=1,
                win_length=acceleration.shape[-1],
                power=1,
                normalized=False,
                center=False,
            ).squeeze(-1)[:, :args.dft_cutoff_bins]
            spectra.append(spectrum.cpu())
    return torch.cat(spectra, dim=0)


def reconstruct_waveform(spectra, args):
    full_bins = args.window // 2 + 1
    if spectra.shape[1] > full_bins:
        raise ValueError(
            f"Cannot place {spectra.shape[1]} bins into an STFT with {full_bins} bins."
        )

    magnitude = np.zeros((full_bins, spectra.shape[0]), dtype=np.float32)
    magnitude[:spectra.shape[1], :] = spectra.numpy().T
    context_frames = args.window // (2 * args.step)
    magnitude = np.pad(
        magnitude,
        ((0, 0), (context_frames, context_frames)),
        mode="edge",
    )
    waveform = librosa.griffinlim(
        magnitude,
        n_iter=args.griffin_lim_iters,
        hop_length=args.step,
        win_length=args.window,
        n_fft=args.window,
        window="hann",
        momentum=0.99,
        init="random",
        random_state=0,
        center=True,
        length=args.window + args.step * (spectra.shape[0] - 1),
    )
    return lowpass_filter(waveform, cutoff=1000.0, fs=args.sampling_rate)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()

    force, speed, direction = load_action(
        Path(args.actions_dir), args.action, args.window, args.step
    )
    force, speed = normalize_inputs(force, speed, Path(args.norm_stats))

    if not 0 <= args.chunk_index < len(force):
        raise ValueError(
            f"--chunk-index must be between 0 and {len(force) - 1}, got {args.chunk_index}."
        )

    model = HNF(
        m_dim=args.m_dim,
        n_freq=args.n_freq,
        hidden=args.hidden,
        dropout_rate=args.dropout_rate,
    ).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    image = load_image(Path(args.image), args.image_rotation).unsqueeze(0).to(device)
    direction = rotate_direction(direction, args.direction_rotation)
    spectra = predict_spectra(model, image, direction, speed, force, args, device)
    spectrum = spectra[args.chunk_index]
    waveform = reconstruct_waveform(spectra, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.image).stem}_{args.action}_chunk{args.chunk_index}"
    spectrum_path = output_dir / f"{stem}.npy"
    figure_path = output_dir / f"{stem}.png"
    waveform_path = output_dir / f"{Path(args.image).stem}_{args.action}_waveform.npy"
    waveform_figure_path = output_dir / f"{Path(args.image).stem}_{args.action}_waveform.png"

    spectrum_np = spectrum.numpy()
    np.save(spectrum_path, spectrum_np)
    np.save(waveform_path, waveform.astype(np.float32))

    frequencies = np.fft.rfftfreq(args.window, d=1.0 / args.sampling_rate)
    frequencies = frequencies[:spectrum_np.shape[0]]

    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, spectrum_np, linewidth=3)
    plt.title(f"Predicted spectrum: {args.action}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

    time = np.arange(waveform.shape[0]) / args.sampling_rate
    plt.figure(figsize=(12, 6))
    plt.plot(time, waveform, linewidth=1)
    plt.title(f"Reconstructed acceleration: {args.action}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(waveform_figure_path)
    plt.close()

    print(f"Checkpoint: {checkpoint}")
    print(f"Saved spectrum: {spectrum_path}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved waveform: {waveform_path}")
    print(f"Saved waveform figure: {waveform_figure_path}")


if __name__ == "__main__":
    main()
