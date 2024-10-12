import torch
from torch import nn
import torchmetrics

from config import NUM_PHONEMES

#### Loss Functions
ce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCELoss()

def calculate_phoneme_metrics(phoneme_logits, phoneme_labels):
    """
    Calculate precision, recall, and F1 score for phoneme predictions.

    Args:
        phoneme_logits (torch.Tensor): Tensor of shape (batch_size, sequence_length, NUM_PHONEMES)
            containing the logits for each phoneme class.
        phoneme_labels (torch.Tensor): Tensor of shape (batch_size, sequence_length)
            containing the true phoneme labels.

    Returns:
        tuple: (phoneme_precision, phoneme_recall, phoneme_f1)
    """
    phoneme_preds = torch.argmax(phoneme_logits, dim=-1)
    phoneme_precision = torchmetrics.functional.precision(phoneme_preds, phoneme_labels, task='multiclass', num_classes=NUM_PHONEMES)
    phoneme_recall = torchmetrics.functional.recall(phoneme_preds, phoneme_labels, task='multiclass', num_classes=NUM_PHONEMES)
    phoneme_f1 = torchmetrics.functional.f1_score(phoneme_preds, phoneme_labels, task='multiclass', num_classes=NUM_PHONEMES)
    return {
        "precision": phoneme_precision,
        "recall": phoneme_recall,
        "f1": phoneme_f1
    }

def calculate_frame_metrics(start_idx_pred, frame_bitmask_labels):
    frame_preds = (start_idx_pred > 0.5).float()
    frame_precision = torchmetrics.functional.precision(frame_preds, frame_bitmask_labels, task='binary')
    frame_recall = torchmetrics.functional.recall(frame_preds, frame_bitmask_labels, task='binary')
    frame_f1 = torchmetrics.functional.f1_score(frame_preds, frame_bitmask_labels, task='binary')
    return {
        "precision": frame_precision,
        "recall": frame_recall,
        "f1": frame_f1
    }