import os
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import HNF
from src.haptic_dataloader_aligned import HapticDataLoader
from src.utils import *
from src.config import FLAGS

from tqdm import tqdm
import wandb
from absl import app

from st_sim.ST_SIM import STSIMLoss as stsim
from st_sim.ST_SIM import STSIMMetric as stsimmetric

import warnings
warnings.filterwarnings("ignore")

def rms_lsd(spec1: torch.Tensor, spec2: torch.Tensor, eps=1e-12):
    """
    Compute Root Mean Square Log-Spectral Distance (RMS-LSD)
    Inputs:
        spec1, spec2: (B, F, T) magnitude or power spectrograms
    Returns:
        scalar RMS-LSD over batch
    """

    # Safety
    assert spec1.shape == spec2.shape, "Spectrograms must have same shape"

    # Convert to log domain
    log1 = torch.log(spec1 + eps)
    log2 = torch.log(spec2 + eps)

    # Per-frame log-spectral distance
    lsd = torch.sqrt(torch.mean((log1 - log2) ** 2, dim=1))   # shape (B, T)

    # RMS over time + batch
    rms_lsd = torch.sqrt(torch.mean(lsd ** 2))

    return rms_lsd


def train_one_epoch(FLAGS, model: nn.Module, dataloader: DataLoader, criterion, optimizer, scheduler, device: torch.device) -> float:
    model.train()

    # CRITICAL: keep frozen encoders in eval() so BatchNorm running stats do NOT drift
    if hasattr(model, "texture_encoder"):
        model.texture_encoder.eval()

    running_loss = 0.0
    pbar = tqdm(dataloader)
    pbar.set_description("Training")

    for (accel_chunk, accel_chunk_time, force_chunk, speed_chunk, direction_chunk, image, material, accel_orig, norm_info) in pbar:
        accel_chunk = accel_chunk.float().to(device)
        force_chunk = force_chunk.to(device)
        speed_chunk = speed_chunk.to(device)
        direction_chunk = direction_chunk.to(device)
        image = image.to(device)
        
        # optionally rotate image by 90 degrees and also direction by 90 degrees
        if torch.randint(0, 10, (1,)).item() == 1:
            import torchvision.transforms.functional as TF
            image = TF.rotate(image, angle=-90)
            x = direction_chunk[..., 0]
            y = direction_chunk[..., 1]
            direction_chunk = torch.stack([-y, x], dim=-1)  # CCW

        
        import json

        # after you have a norm_info dict available
        with open("norm_stats.json", "w") as f:
            json.dump({
                "min_force": float(torch.min(norm_info["min_force"])),
                "max_force": float(torch.max(norm_info["max_force"])),
                "min_speed": float(torch.min(norm_info["min_speed"])),
                "max_speed": float(torch.max(norm_info["max_speed"])),
            }, f, indent=2)


        # direction_chunk = torch.zeros_like(direction_chunk).to(device)
        # force_chunk = torch.zeros_like(force_chunk).to(device)
        # speed_chunk = torch.zeros_like(speed_chunk).to(device)

        acc_output = model.render(
            image, direction_chunk, speed_chunk, force_chunk,
            n_samples=FLAGS.num_of_bins, stratified=True
        )  # (B, C) where C = chunk_size / num_of_bins
        acc_output = acc_output - acc_output.mean(dim=-1, keepdim=True)

        # This produces (B, F, 1) because acc_output length == win_length == n_fft
        dft_predicted = torchaudio.functional.spectrogram(
            acc_output,
            pad=0,
            window=torch.hann_window(200, device=acc_output.device),
            n_fft=200,
            hop_length=1,
            win_length=200,
            power=1,
            normalized=False,
            center=False
        ).squeeze(-1)  # -> (B, F)

        dft_predicted = dft_predicted[:, :FLAGS.dft_cutoff_bins]  # -> (B, bins)

        loss = criterion(dft_predicted, accel_chunk)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Optional: monitor image_encoder grads (may be None if frozen/warmup)
        if hasattr(model, "image_encoder") and model.image_encoder[0].weight.grad is not None:
            pbar.set_postfix({
                "loss": running_loss/(pbar.n+1),
                "img_grad": float(model.image_encoder[0].weight.grad.norm().item())
            })
        else:
            pbar.set_postfix({"loss": running_loss/(pbar.n+1)})

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    save_signal_chart(dft_predicted[-1], accel_chunk[-1], os.path.join(FLAGS.output_dir, f"dft_chart"), f"train_dft_{material[-1]}.png", plot_type='freq')
    
    epoch_loss = running_loss / max(len(dataloader), 1)

    if scheduler is not None:
        scheduler.step()

    return epoch_loss

def evaluate(FLAGS, model: nn.Module, dataloader: DataLoader, criterion, device: torch.device, epoch: int) -> float:
    model.eval()
    running_loss = 0.0
    running_loss_1 = 0.0
    stsim_metric = stsimmetric(fs=1000, bl=10)
    with torch.no_grad():
        pbar = tqdm(dataloader)
        pbar.set_description(f"Evaluating")
        for (accel_chunk, accel_chunk_time, force_chunk, speed_chunk, direction_chunk, image, material, accel_orig, norm_info) in pbar:
            accel_chunk = accel_chunk.float().to(device)
            accel_chunk_time = accel_chunk_time.to(device)
            force_chunk = force_chunk.to(device)
            speed_chunk = speed_chunk.to(device)
            direction_chunk = direction_chunk.to(device)
            image = image.to(device)

            acc_output = model.render(image, direction_chunk, speed_chunk, force_chunk, n_samples=FLAGS.num_of_bins, stratified=False)
            acc_output = acc_output - acc_output.mean(dim=-1, keepdim=True)
 
            dft_predicted = torchaudio.functional.spectrogram(acc_output, pad=0, window=torch.hann_window(200).to(acc_output.device), n_fft=200, hop_length=1, win_length=200, power=1, normalized=False, center=False).squeeze(-1)
            dft_predicted = dft_predicted[:,:FLAGS.dft_cutoff_bins]

            loss = rms_lsd(dft_predicted, accel_chunk)
            stsim_value = stsim_metric.apply(accel_chunk, dft_predicted, dct=False).float()

            running_loss += loss.item()
            running_loss_1 += stsim_value.item()
            pbar.set_postfix({"loss": running_loss/(pbar.n+1), "st-sim": running_loss_1/(pbar.n+1)})

        save_signal_chart(dft_predicted[-1], accel_chunk[-1], os.path.join(FLAGS.output_dir, f"dft_chart"), f"eval_dft_{material[-1]}.png", plot_type='freq')

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def reconstruct_signals(FLAGS, model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    reconstruct_signals = []
    with torch.no_grad():
        pbar = tqdm(dataloader)
        pbar.set_description(f"Reconstructing")
        for idx, (accel_chunk, accel_chunk_time, force_chunk, speed_chunk, direction_chunk, image, material, accel_orig, norm_info) in enumerate(pbar):
            # chech if next material is different than current material
            if idx > 0 and material != prev_material:
                # save the reconstructed signal and ground truth signal
                recon = np.zeros((501, 991))
                gt = np.zeros((501, 991))
                reconstructed_signals = np.concatenate(reconstruct_signals).T
                for idx in range(reconstructed_signals.shape[0]):
                    recon[idx] = reconstructed_signals[idx]
                recon, gt = save_signal_chart(recon, accel_orig.cpu().numpy(), os.path.join(FLAGS.output_dir, f"recon_chart"), f"recon_signal_comparison_{prev_material}.png", use_wandb=FLAGS.wandb)
                
                # Temporal MSE between reconstructed signal and gt
                mse = np.mean((recon - gt) ** 2)
                print(f"MSE for material {prev_material}: {mse}")
                if FLAGS.wandb:
                    wandb.log({f"metrics/mse_material_{prev_material}": mse})
                reconstruct_signals = []
                
            accel_orig = accel_orig.float().to(device)
            force_chunk = force_chunk.to(device)
            speed_chunk = speed_chunk.to(device)
            direction_chunk = direction_chunk.to(device)
            image = image.to(device)
            prev_material = material

            acc_output = model.render(image, direction_chunk, speed_chunk, force_chunk, n_samples=FLAGS.num_of_bins, stratified=False)
            acc_output = acc_output - acc_output.mean(dim=-1, keepdim=True)
            
            dft_predicted = torchaudio.functional.spectrogram(acc_output, pad=0, window=torch.hann_window(200).to(acc_output.device), n_fft=200, hop_length=1, win_length=200, power=1, normalized=False, center=False).squeeze(-1)
            dft_predicted = dft_predicted[:,:FLAGS.dft_cutoff_bins]
            dft_predicted = dft_predicted.cpu().numpy()

            reconstruct_signals.append(dft_predicted)
            
        # save the reconstructed signal and ground truth signal
        recon = np.zeros((501, 991))
        gt = np.zeros((501, 991))
        reconstructed_signals = np.concatenate(reconstruct_signals).T
        for idx in range(reconstructed_signals.shape[0]):
            recon[idx] = reconstructed_signals[idx]
        recon, gt = save_signal_chart(recon, accel_orig.cpu().numpy(), os.path.join(FLAGS.output_dir, f"recon_chart"), f"recon_signal_comparison_{prev_material}.png")
        reconstruct_signals = []
        
        # MSE between recon and gt
        mse = np.mean((recon - gt) ** 2)
        print(f"MSE for material {prev_material}: {mse}")

        if FLAGS.wandb:
            wandb.log({f"metrics/mse_material_{prev_material}": mse})

    return reconstructed_signals, gt

def main(argv):
    # 1) Init experiment
    output_dir_path = init_experiment(FLAGS)
    FLAGS.output_dir = output_dir_path + '/'
    if FLAGS.wandb:
        wandb.init(project=FLAGS.project_name, config=FLAGS, dir=output_dir_path)
        wandb.run.name = os.path.basename(output_dir_path)

    # 2) Load Data
    ds = HapticDataLoader(use_n_materials=FLAGS.num_of_materials, dft_bins_to_keep=FLAGS.dft_cutoff_bins)
    train_loader = DataLoader(ds.get_subset('train'), batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)
    val_loader   = DataLoader(ds.get_subset('val'), batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers)
    test_loader  = DataLoader(ds.get_subset('test'), batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers)
    recon_loader = DataLoader(ds.get_subset('recon'), batch_size=1, shuffle=False, num_workers=FLAGS.num_workers)

    # 3) Init model
    model = HNF(m_dim=FLAGS.m_dim, n_freq=FLAGS.n_freq, hidden=FLAGS.hidden, dropout_rate=FLAGS.dropout_rate).to(FLAGS.device)

    # Freeze the ResNet backbone (texture_encoder)
    for p in model.texture_encoder.parameters():
        p.requires_grad = False
    model.texture_encoder.eval()  # important for BatchNorm stats stability

    warmup_epochs = getattr(FLAGS, "warmup_epochs", 0)

    # 4) Optimizer with param groups: small LR for image_encoder, larger LR for the NeRF MLPs
    base_lr = FLAGS.learning_rate
    img_lr = getattr(FLAGS, "image_lr", base_lr * 0.01)

    mlp_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("image_encoder."):
            continue
        mlp_params.append(p)

    param_groups = [
        {"params": mlp_params, "lr": base_lr},
        {"params": model.image_encoder.parameters(), "lr": img_lr},
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=FLAGS.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FLAGS.num_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    # 5) Training loop
    best_val_loss = float('inf')
    for epoch in range(FLAGS.num_epochs):
        # Warmup handling
        if warmup_epochs > 0:
            if epoch < warmup_epochs:
                for p in model.image_encoder.parameters():
                    p.requires_grad = False
            elif epoch == warmup_epochs:
                for p in model.image_encoder.parameters():
                    p.requires_grad = True
                print(f"[Warmup] Unfroze image_encoder at epoch {epoch}")
        # if warmup_epochs > 0:
        #     if epoch < warmup_epochs:
        #         for p in model.sigma_mlp.parameters():
        #             p.requires_grad = False
        #         for p in model.a_mlp.parameters():
        #             p.requires_grad = False
        #     elif epoch == warmup_epochs:
        #         for p in model.sigma_mlp.parameters():
        #             p.requires_grad = True
        #         for p in model.a_mlp.parameters():
        #             p.requires_grad = True
        #         print(f"[Warmup] Unfroze hnf at epoch {epoch}")

        train_loss = train_one_epoch(FLAGS, model, train_loader, criterion, optimizer, scheduler, FLAGS.device)
        print(f"Epoch {epoch+1}/{FLAGS.num_epochs} | train_loss={train_loss:.6f}")

        if FLAGS.wandb:
            wandb.log({"train_loss": train_loss, "epoch": epoch+1})
            
        if (epoch+1) % FLAGS.validation_interval == 0:
            val_loss = evaluate(FLAGS, model, val_loader, criterion, FLAGS.device, epoch)
            print(f"Epoch {epoch+1}/{FLAGS.num_epochs}, Validation Loss: {val_loss:.6f}")
            if FLAGS.wandb:
                wandb.log({"val_loss": val_loss, "epoch": epoch+1})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the best model
                save_dir = os.path.join(FLAGS.output_dir, 'checkpoints')
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                print(f"Best model saved at epoch {epoch+1} with val loss {best_val_loss:.6f}")
                
        if (epoch+1) % FLAGS.checkpoint_interval == 0:
            rec, orig = reconstruct_signals(FLAGS, model, recon_loader, FLAGS.device)

    # Save final model
    save_dir = os.path.join(FLAGS.output_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print(f"Saved final model to {save_dir}")
    
    # 5. Reconstruction on recon set
    rec, orig = reconstruct_signals(FLAGS, model, recon_loader, FLAGS.device)


if __name__ == "__main__":
    app.run(main)
