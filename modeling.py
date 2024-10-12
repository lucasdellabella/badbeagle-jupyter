from transformers import Wav2Vec2PreTrainedModel
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import ( CausalLMOutput,)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from transformers import Wav2Vec2Model
from transformers import AutoConfig
from config import NUM_PHONEMES

#### LOSS FUNCTIONS
def distance_weighted_mse_loss(pred_probs, target, distance_weight=0.5, device='cpu'):
    # Find the indices of the actual start frames
    start_indices = torch.nonzero(target, as_tuple=True)[1].to(device)
    
    # Create a tensor of frame indices
    frame_indices = torch.arange(pred_probs.size(1)).to(device)
    
    # Calculate the distance matrix
    distance_matrix = torch.abs(frame_indices.unsqueeze(1) - start_indices)
    
    # Select the closest frame
    distance_to_closest_actual_marker = torch.min(distance_matrix, dim=-1).values
    
    weights = (1 + distance_to_closest_actual_marker * distance_weight) ** 2
    
    # Calculate the weighted mean squared error loss
    squared_error = (pred_probs - target.float()) ** 2
    weighted_squared_error = squared_error * weights.unsqueeze(0)

    loss = torch.mean(weighted_squared_error)

    return loss


def custom_loss_function(phoneme_logits, frame_start, phoneme_labels, frame_labels, input_lengths, phoneme_lengths, device='cpu'):
    
    phoneme_probs = nn.functional.softmax(phoneme_logits, dim=-1)
    phoneme_loss = nn.functional.ctc_loss(torch.log(phoneme_probs.transpose(0, 1)), phoneme_labels, input_lengths, phoneme_lengths)
    
    frame_start_loss = distance_weighted_mse_loss(frame_start, frame_labels.float(), device=device)
    
    total_loss = phoneme_loss + frame_start_loss * 20
    
    return total_loss, phoneme_loss, frame_start_loss


#### MODELS
class MyWav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """

        # Note that `tie_weights` is usually used to tie input and output embedding weights. The method is re-purposed to
        # correctly load adapter layers for Wav2Vec2 so that we do not have to introduce a new API to
        # [`PreTrainedModel`]. While slightly hacky, Wav2Vec2 never has to tie input and output embeddings, so that it is
        # ok to repurpose this function here.
        target_lang = self.target_lang

        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False


    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
    
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        log_probs = None
        loss = None
        if labels is not None:
            if labels.max() >= NUM_PHONEMES:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            loss = None
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        ), log_probs  # Modify this line


class Wav2Vec2ForPhonemeAndFramePrediction(nn.Module):
    def __init__(self, num_phonemes, dropout_rate: float = None, freeze_feature_encoder=False):
        super().__init__()

        config = AutoConfig.from_pretrained("facebook/wav2vec2-base-960h")
        config.vocab_size = num_phonemes + 1
        
        self.wav2vec2 = MyWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", 
                                                         ignore_mismatched_sizes=True,
                                                        config=config)
        
        if (freeze_feature_encoder):
            self.wav2vec2.freeze_feature_encoder()
        
        if (dropout_rate != None):
            self.dropout = nn.Dropout(dropout_rate)
            
        # self.phoneme_head = nn.Linear(self.wav2vec2.config.hidden_size, config.vocab_size)
        # self.frame_start_head = nn.Linear(self.wav2vec2.config.hidden_size, 1)
        
#         self.phoneme_head = nn.Sequential(
#             nn.Linear(self.wav2vec2.config.hidden_size, 512),  # First layer with increased units
#             nn.ReLU(),
#             nn.Linear(512, config.vocab_size)  # Output layer
#         )

#         self.frame_start_head = nn.Sequential(
#             nn.Linear(self.wav2vec2.config.hidden_size, 512),  # First layer with increased units
#             nn.ReLU(),
#             nn.Linear(512, 1)  # Output layer
#         )

        self.phoneme_head = nn.Sequential(
            PrintShape(),
            nn.Linear(self.wav2vec2.config.hidden_size, 1024),  # Increased units
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),  # Additional layer
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, config.vocab_size)  # Output layer
        )

        self.frame_start_head = nn.Sequential(
            PrintShape(),
            nn.Linear(self.wav2vec2.config.hidden_size, 1024),  # Increased units
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),  # Additional layer
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1)  # Output layer
        )

    def forward(self, input_values, phoneme_labels=None):
        causal_lm_output, _ = self.wav2vec2(input_values, labels=phoneme_labels, output_hidden_states=True)
        outputs = causal_lm_output.hidden_states[-1]
        
        phoneme_logits = self.phoneme_head(outputs)
        frame_start = self.frame_start_head(outputs).squeeze(-1)
        
        return phoneme_logits, frame_start

class PrintShape(nn.Module):
    def __init__(self):
        super(PrintShape, self).__init__()
    
    def forward(self, x):
        print(f"Shape: {x.shape}")
        return x
