import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class TinyBERT(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()

        # ---- 1. BERT configuration ----
        
        config = BertConfig(
            vocab_size=30522,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            layer_norm_eps=1e-12,
        )

        # ---- 2. BERT encoder from scratch ----
        # No pretrained weights are loaded.
        # No need BERT's original pooler. [CLS] pooling

        self.bert = BertModel(
            config,
            add_pooling_layer=False
        )

        # ---- 3. Classification head ----

        self.classifier = nn.Linear(
            config.hidden_size,
            num_classes
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        
        # ---- 1. BERT encoder ----

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # ---- 2. Extract the [CLS] representation ----

        cls_output = outputs.last_hidden_state[:, 0, :]

        # ---- 3. Classification ----
        # Shape: (batch_size, 32) -> (batch_size, num_classes)

        logits = self.classifier(cls_output)

        return logits