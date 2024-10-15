import torch

from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorForAudioWithPadding:
    """
    Data collator that dynamically pads both the inputs and labels received.
    This is specialized for audio processing tasks where labels include phoneme information and frame indices.

    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`):
            The processor used for processing the audio data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, optional, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index).
    """
    processor: Any  # Adjusted to generic 'Any' if not specifically using Wav2Vec2Processor
    padding: Union[bool, str] = True
    
    def __init__(self, processor, padding=True, tokenizer=None):
        self.processor = processor
        self.padding = padding
        self.tokenizer = tokenizer  # Added tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate audio input values from labels since they need different padding methods
        input_features = [{"input_values": feature["audio"]} for feature in features]
        labels = [{"labels": feature["labels"]} for feature in features]
        phoneme_labels_CE = [feature["phoneme_labels_CE"] for feature in features]
        frame_labels_CE = [feature["frame_labels_CE"] for feature in features]

        # Pad audio input features
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels without any processor since labels are numerical indices (no text tokenization)
        max_label_length = max(len(l["labels"]['phoneme_start_idx']) for l in labels)
        padded_labels = torch.full((len(labels), max_label_length), -100, dtype=torch.long)  # Using -100 as the padding value for labels
        padded_labels_start_idx = torch.full((len(labels), max_label_length), -100, dtype=torch.long)  # Using -100 as the padding value for labels
        padded_labels_utterance = torch.full((len(labels), max_label_length), -100, dtype=torch.long)  # Using -100 as the padding value for labels
        
        # Our labels are structured like:
        # [{
        #   "phoneme_start_idxs": [],
        #   "utterance": []
        # }]
        for i, label in enumerate(labels):
            label_length = len(label["labels"]['phoneme_start_idx'])

            padded_labels_start_idx[i, :label_length] = torch.tensor(label["labels"]['phoneme_start_idx'], dtype=torch.long)
            padded_labels_utterance[i, :label_length] = torch.tensor(self.tokenizer.convert_tokens_to_ids(label["labels"]['utterance']), dtype=torch.long)
            padded_labels = {'phoneme_start_idx': padded_labels_start_idx, 'utterance': padded_labels_utterance}


        max_ce_label_length = max(len(row) for row in frame_labels_CE)
        padded_phoneme_labels_CE = torch.full((len(features), max_ce_label_length), -100, dtype=torch.long)  # Using -100 as the padding value for labels
        padded_frame_labels_CE = torch.full((len(features), max_ce_label_length), -100, dtype=torch.long)  # Using -100 as the padding value for labels

        for i in range(len(features)):
            label_length = len(phoneme_labels_CE[i])
            padded_phoneme_labels_CE[i, :label_length] = torch.tensor(phoneme_labels_CE[i], dtype=torch.long)
            padded_frame_labels_CE[i, :label_length] = torch.tensor(frame_labels_CE[i])
        
        batch["labels"] = padded_labels
        batch["phoneme_labels_CE"] = padded_phoneme_labels_CE
        batch["frame_labels_CE"] = padded_frame_labels_CE

        return batch
    

class PhonemeTokenizer(PreTrainedTokenizer):
    def __init__(self, phoneme_to_id, id_to_phoneme):
        self.phoneme_to_id = phoneme_to_id
        self.id_to_phoneme = id_to_phoneme
        self.pad_token = "<pad>"

    def tokenize(self, phonemes):
        return [self.phoneme_to_id[ph] for ph in phonemes]

    def convert_tokens_to_ids(self, tokens):
        return [self.phoneme_to_id[token] if token in self.phoneme_to_id else self.phoneme_to_id["<unk>"] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_phoneme[id.item()] for id in ids if id.item() in self.id_to_phoneme]

    def pad(self, encoded_inputs, max_length=None, padding_strategy="longest"):
        max_len = max_length or max(len(inputs) for inputs in encoded_inputs)
        padded = [inp + [self.phoneme_to_id[self.pad_token]] * (max_len - len(inp)) for inp in encoded_inputs]
        return padded