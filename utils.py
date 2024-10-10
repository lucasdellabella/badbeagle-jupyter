
import torch

import soundfile as sf
import os

from config import SAMPLE_RATE, WINDOW_SIZE

def frame_pos_to_nth_window(frame_position):
    return torch.floor_divide(torch.divide(frame_position * 1000, SAMPLE_RATE) - 5, WINDOW_SIZE)

def get_audio_length_in_windows(audio):
    if len(audio.shape) != 1:
        raise ValueError("Need to pass a 1 dimensional vector in as arg")
    return int(frame_pos_to_nth_window(len(audio)).item())

def create_bit_mask_of_frame_start_positions(frame_positions_batch, num_of_windows):
    batch_size = frame_positions_batch.size(0)
    bitmask_batch = torch.zeros(batch_size, num_of_windows, dtype=torch.bool)

    for i in range(batch_size):
        bitmask_batch[i, frame_positions_batch[i].long() - 1] = True
        
    return bitmask_batch

def example_pred_and_target():
    pred_probs = torch.tensor([[0.4, 0.6, 0.3, 0.8, 0.3, 0.9, 0.5, 0.5, 0.5, 0.5, 0.9]])
    target = torch.tensor([[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]])

    return (pred_probs, target)


def save_audio(audio_data, filename="sample_audio.wav"):
    """audio_data needs to be in numpy form on cpu. call .cpu().numpy() if you have a tensor"""

    # Create a directory to save the audio file if it doesn't exist
    output_dir = 'audio_output'
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file path
    output_file = os.path.join(output_dir, filename)

    # Save the audio data to a WAV file
    sf.write(output_file, audio_data, SAMPLE_RATE)

    print(f"Audio saved to: {output_file}")