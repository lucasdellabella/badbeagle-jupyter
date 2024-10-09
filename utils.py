
import torch

from config import SAMPLE_RATE, WINDOW_SIZE

def frame_pos_to_nth_window(frame_position):
    return torch.floor_divide(torch.divide(frame_position * 1000, SAMPLE_RATE) - 5, WINDOW_SIZE)

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