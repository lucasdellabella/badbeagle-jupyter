
import torch

import soundfile as sf
import os

from config import SAMPLE_RATE, WINDOW_SIZE
from datetime import datetime
from pathlib import Path

def generate_attention_mask(input_values):
    batch_size, sequence_length = input_values.shape
    attention_mask = torch.zeros((batch_size, sequence_length), dtype=torch.long, device=input_values.device)
    
    for i in range(batch_size):
        last_non_zero_idx = torch.nonzero(input_values[i], as_tuple=True)[0][-1].item()
        attention_mask[i, :last_non_zero_idx + 1] = 1
    
    return attention_mask

def frame_pos_to_nth_window(frame_position):
    return torch.floor_divide(torch.divide(frame_position * 1000, SAMPLE_RATE) - 5, WINDOW_SIZE).to(dtype=torch.int)

def get_audio_length_in_windows(audio):
    if len(audio.shape) != 1:
        raise ValueError("Need to pass a 1 dimensional vector in as arg")
    return int(frame_pos_to_nth_window(len(audio)).item())

def get_audio_length_in_windows_batched(batched_audio):
    if len(batched_audio.shape) != 2:
        raise ValueError("Need to pass a 2 dimensional vector in as arg")
    attn_mask = generate_attention_mask(batched_audio)
    audio_length_in_frames = attn_mask.sum(-1)
    return frame_pos_to_nth_window(audio_length_in_frames)

def create_bit_mask_of_frame_start_positions(frame_positions_batch, num_of_windows):
    batch_size = frame_positions_batch.size(0)
    bitmask_batch = torch.zeros(batch_size, num_of_windows, dtype=torch.bool)

    for i in range(batch_size):
        bitmask_batch[i, frame_positions_batch[i].long() - 1] = True
        
    return bitmask_batch


def get_class_id_for_each_frame(frame_start_indices, classes, input_length_in_frames):
    batch_size = frame_start_indices.shape[0]
    
    # Create a mask for valid classes (not -100)
    valid_mask = classes != -100
    label_lengths = valid_mask.sum(dim=-1)
    
    # prep frame_start_indices ahead of 
    additional_column = torch.zeros((batch_size, 1), dtype=torch.float, device=valid_mask.device)
    frame_start_indices = torch.cat([frame_start_indices, additional_column], dim=1)

    # Calculate the number of frames for each class
    for i in range(batch_size):
        batch_input_length = input_length_in_frames[i]
        batch_label_length = label_lengths[i]
        frame_start_indices[i][batch_label_length] = batch_input_length

    frame_start_indices[:, 0] = 0
    frame_counts = torch.diff(frame_start_indices, dim=1)
    frame_counts = torch.where(valid_mask, frame_counts, torch.zeros_like(frame_counts))
    
    # Create a tensor to hold the result
    max_frames = input_length_in_frames.max().item()
    result = torch.full((batch_size, max_frames), -100, device=classes.device, dtype=classes.dtype)
    
    for i in range(batch_size):
        valid_classes = classes[i][valid_mask[i]]
        valid_counts = frame_counts[i][valid_mask[i]]
        
        # Valid_counts is wrong, the last value is incorrect
        expanded = torch.repeat_interleave(valid_classes, valid_counts.long())
        
        # Trim or pad to match input_length_in_frames
        if len(expanded) < input_length_in_frames[i]:
            padding = torch.full((input_length_in_frames[i] - len(expanded),), valid_classes[-1], device=classes.device, dtype=classes.dtype)
            expanded = torch.cat([expanded, padding])
        else:
            expanded = expanded[:input_length_in_frames[i]]
        
        # Fill the result tensor
        result[i, :len(expanded)] = expanded

    return result


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


def save_model(model, filename):
    # Get current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a filename with the datetime prefix
    model_filename = f"{current_datetime}_{filename}.pth"

    # Create a Path object for the models directory and the full file path
    models_dir = Path("models")
    model_path = models_dir / model_filename

    # Create the models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), model_path)

    print(f"Model saved as: {model_path}")
