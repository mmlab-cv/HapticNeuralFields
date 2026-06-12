import os
import torch
import torch.nn as nn
import numpy as np
import random
from scipy.signal import butter, filtfilt
import datetime
import matplotlib.pyplot as plt
import librosa
import wandb

def _griffin_lim_reconstruct(spectrogram_mag, sampling_rate, window_duration_s, step_duration_s):
    """
    Reconstructs a temporal signal from a sequence of DFT magnitudes using the Griffin-Lim Algorithm.
    This implementation uses librosa, which is an external library not explicitly detailed in the sources.
    The sources mention using Průša's Matlab implementation of GLA [1, 4].

    Args:
        spectrogram_magnitudes_list (list of np.ndarray): A list where each element is
                                                        a NumPy array representing the DFT magnitude for a 0.1-second window,
                                                        filtered to include frequencies up to 1000 Hz (e.g., 100 bins).
        sampling_rate (int): The sampling rate of the original signal in Hz.
        window_duration_s (float): The duration of the FFT window in seconds.
        step_duration_s (float): The step size (hop length) between windows in seconds.

    Returns:
        np.ndarray: The reconstructed temporal acceleration signal.
    """

    n_iter = 128  # Number of Griffin-Lim iterations
    n_fft = int(sampling_rate * window_duration_s)  # FFT window size in samples
    hop_length = int(sampling_rate * step_duration_s)  # Hop length in samples
    win_length = n_fft  # Window length in samples

    reconstructed_signal = librosa.griffinlim(
        spectrogram_mag,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        n_iter=n_iter,
        momentum=0.99,
        init='random',
        random_state=0
    )
    return reconstructed_signal

def save_signal_chart(pred_signal, gt_signal, output_dir, filename, plot_type='recon',
                     sampling_rate=10000,
                     window_duration_s=0.1,
                     step_duration_s=0.01,
                     label1='Predicted Signal', label2='Ground Truth Signal', use_wandb=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert inputs to NumPy arrays if they are PyTorch tensors
    pred_signal_np = pred_signal.detach().cpu().numpy() if not isinstance(pred_signal, np.ndarray) else pred_signal
    gt_signal_np = gt_signal.detach().cpu().numpy() if not isinstance(gt_signal, np.ndarray) else gt_signal

    plt.figure(figsize=(12, 6))
    
    if plot_type == 'recon':
        
        current_data1_temporal = _griffin_lim_reconstruct(pred_signal_np, sampling_rate, window_duration_s, step_duration_s)
        current_data2_temporal = gt_signal_np.squeeze(0)
        current_data1_temporal = lowpass_filter(current_data1_temporal, cutoff=1000.0, fs=sampling_rate, order=5)

        # Plot the reconstructed temporal signals
        # Time axis for the full reconstructed signal
        total_time_s = len(current_data1_temporal) / sampling_rate
        time_axis = np.linspace(0, total_time_s, len(current_data1_temporal), endpoint=False)

        plt.plot(time_axis, current_data1_temporal, label=label1, alpha=0.7)
        plt.plot(time_axis, current_data2_temporal, label=label2, alpha=0.6)

        plt.title('Reconstructed Acceleration Signal (Temporal Domain via Griffin-Lim)', fontsize=14)
        plt.xlabel(f'Time (s)', fontsize=12)
        plt.ylabel('Acceleration (mm/s²)', fontsize=12)
        plt.xlim(0, total_time_s)

    elif plot_type == 'time':
        # --- Temporal Domain Plotting (as per previous conversation) ---
        num_samples = len(np.fft.irfft(pred_signal_np,n=1000))
        time_axis = np.linspace(0, window_duration_s, num_samples, endpoint=False) # Time in seconds [1, 4-8]

        plt.plot(time_axis, np.fft.irfft(pred_signal_np,n=1000), label=label1, alpha=0.7)
        plt.plot(time_axis, np.fft.irfft(gt_signal_np,n=1000), label=label2, alpha=0.7)

        plt.title('Predicted vs. Ground Truth Acceleration Signal (Temporal Domain)', fontsize=14)
        plt.xlabel(f'Time (s) for {window_duration_s}s Window', fontsize=18)
        plt.ylabel('Acceleration (m/s²)', fontsize=18) # Units based on source [1]
        plt.xlim(0, window_duration_s)

    elif plot_type == 'freq':
        frequencies = np.fft.rfftfreq(1000, d=1./sampling_rate)[:100]
        plt.plot(frequencies, pred_signal_np, label=label1, alpha=0.7, linewidth=7)
        plt.plot(frequencies, gt_signal_np, label=label2, alpha=0.7, linewidth=7)

        plt.title('Predicted vs. Ground Truth Acceleration DFT Magnitude (Spectral Domain)', fontsize=22)
        plt.xlabel('Frequency (Hz)', fontsize=20)
        plt.ylabel('Magnitude (mm/s²/Hz)', fontsize=20) # Matching Figure 5's label [Figure 5 in Source 2]
        plt.xlim(0, 1000)

    else:
        raise ValueError(f"Invalid plot_type: '{plot_type}'. Must be 'time' or 'frequency'.")

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    
    # Log plot to wandb if wandb is initialized
    try:
        if use_wandb:
            wandb.log({filename: wandb.Image(plt.gcf())})
    except ImportError:
        print("Logging to wandb failed.")
        pass
    
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    if plot_type == 'recon':
        return current_data1_temporal, current_data2_temporal
    else:
        return None, None
    

def lowpass_filter(data, cutoff=20.0, fs=1000.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low', analog=False)
    return filtfilt(b, a, data)

# def set_random_seed(seed):
#     """
#     Set the random seed for reproducibility.
#     """
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
        
def set_random_seed(seed):
    print(f"[Seed] Using seed {seed}")
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (CPU & CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
def create_output_dirs(FLAGS):
    """
    Create output directories for the experiment.
    """
    experiment_name = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
    output_dir_path = os.path.join(FLAGS.output_dir, experiment_name)
    os.makedirs(output_dir_path, exist_ok=True)
    os.makedirs(os.path.join(output_dir_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir_path, "logs"), exist_ok=True)
    return output_dir_path

def log_experiment_info(FLAGS):
    """
    Log experiment information.
    """
    print("Experiment Configuration:")
    print(f"Learning Rate: {FLAGS.learning_rate}")
    print(f"Batch Size: {FLAGS.batch_size}")
    print(f"Num Epochs: {FLAGS.num_epochs}")
    print(f"Weight Decay: {FLAGS.weight_decay}")
    print(f"Dropout Rate: {FLAGS.dropout_rate}")

    print("Output Directory:", FLAGS.output_dir)
    print("Seed:", FLAGS.seed)

def init_experiment(FLAGS):
    """
    Initialize the experiment with the given flags.
    """
    # Set random seeds for reproducibility
    set_random_seed(FLAGS.seed)

    # Create output directories
    output_dir_path = create_output_dirs(FLAGS)

    # Log experiment details
    log_experiment_info(FLAGS)

    return output_dir_path