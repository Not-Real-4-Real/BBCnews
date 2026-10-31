
import torch
import torch.nn as nn

from transformers import BertModel


PRETRAINED_MODEL_NAME = "bert-base-uncased"


class FrozenBERT(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()

        # ---- 1. Load pretrained BERT-base ----
        self.bert = BertModel.from_pretrained(
            PRETRAINED_MODEL_NAME,
            add_pooling_layer=False,
        )

        # ---- 2. Freeze all BERT parameters ----
        for param in self.bert.parameters():
            param.requires_grad = False

        # ---- 3. New, randomly initialized classifier ----
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_classes,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):

        # The encoder is frozen, so gradients are unnecessary here.
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        # [CLS] representation: (B, 768)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Classification: (B, 768) -> (B, 5)
        logits = self.classifier(cls_output)

        return logits

    def train(self, mode=True):
        # Set the full module's mode first.
        super().train(mode)

        # Keep the frozen encoder in eval mode, so its dropout
        # layers do not randomly alter the extracted features.
        self.bert.eval()

        return self