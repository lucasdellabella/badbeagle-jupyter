
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import torch

from config import SAMPLING_RATE, WINDOW_SIZE

def visualize_audio_and_markers(audio, start_window, end_window, bitmask_labels, label_type='Label', color='blue'):
    # Convert window indices to audio sample indices
    start_idx = start_window * int(SAMPLING_RATE * WINDOW_SIZE / 1000)
    end_idx = end_window * int(SAMPLING_RATE * WINDOW_SIZE / 1000)
    
    # Extract the audio sample
    audio_segment = audio[start_idx:end_idx]
    
    # Calculate the duration of the audio segment
    duration = (end_window - start_window) * WINDOW_SIZE / 1000  # in seconds
    
    # Create the figure and subplot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the spectrogram
    D = librosa.stft(audio_segment.numpy(), n_fft=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='hz', sr=SAMPLING_RATE, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Spectrogram with Phoneme Markers')
    
    # Plot markers directly on the spectrogram using bitmask
    for i, is_start in enumerate(bitmask_labels[start_window:end_window]):
        if is_start:
            label_time = i * WINDOW_SIZE / 1000
            ax.axvline(x=label_time, color=color, linewidth=0.5, alpha=0.7, label=label_type)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    ax.set_xlabel('Time (s)')
    ax.set_xlim(0, duration)  # Set the x-axis limit to the duration of the audio segment
    plt.tight_layout()
    return fig, ax


def visualize_bitmask(labels, predictions_model1, predictions_model2):
    # Detach the tensors from the computation graph and move them to the CPU
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(predictions_model1, torch.Tensor):
        predictions_model1 = predictions_model1.detach().cpu().numpy()
    if isinstance(predictions_model2, torch.Tensor):
        predictions_model2 = predictions_model2.detach().cpu().numpy()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 3))

    # Create a 3-row table
    table_data = np.vstack((labels, predictions_model1, predictions_model2))

    # Create a colormap where 0 is white, 1 is green, and values in between are shades of green
    cmap = plt.cm.Greens

    # Display the table using imshow
    im = ax.imshow(table_data, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    # Set the x-tick labels to every 5th time increment
    ax.set_xticks(range(0, len(labels), 10))
    ax.set_xticklabels(range(1, len(labels) + 1, 10))

    # Set the y-tick labels to "Labels", "Model 1 Predictions", and "Model 2 Predictions"
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Labels", "Model 1 Predictions", "Model 2 Predictions"])

    # Add a colorbar to show the range of values
    cbar = ax.figure.colorbar(im, ax=ax)

    # Show the plot
    plt.tight_layout()
    plt.show()