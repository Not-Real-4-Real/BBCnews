import torch
import torch.nn as nn

from transformers import BertConfig, BertModel


class TinyBERTChunked(nn.Module):

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

        self.bert = BertModel(
            config,
            add_pooling_layer=False,
        )

        # ---- 3. Classification head ----

        self.classifier = nn.Linear(
            config.hidden_size,
            num_classes,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        chunk_to_article=None,
        num_articles=None,
    ):

        # ---- 1. Encode every chunk ----

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Shape: (num_chunks, sequence_length, 32)

        # ---- 2. Extract each chunk's [CLS] representation ----

        chunk_embeddings = outputs.last_hidden_state[:, 0, :]

        # Shape: (num_chunks, 32)

        # ---- 3. Aggregate chunk vectors by article ----

        if chunk_to_article is None or num_articles is None:
            raise ValueError(
                "chunk_to_article and num_articles are required."
            )

        # Start with one zero vector per article.
        article_embeddings = chunk_embeddings.new_zeros(
            num_articles,
            chunk_embeddings.size(-1),
        )

        # Add each chunk vector to its corresponding article.
        article_embeddings = article_embeddings.index_add(
            0,
            chunk_to_article,
            chunk_embeddings,
        )

        # Count how many chunks belong to each article.
        chunk_counts = torch.bincount(
            chunk_to_article,
            minlength=num_articles,
        ).to(
            device=chunk_embeddings.device,
            dtype=chunk_embeddings.dtype,
        )

        # Mean-pool: sum of vectors / number of chunks.
        article_embeddings = (
            article_embeddings / chunk_counts.unsqueeze(1)
        )

        # Shape: (num_articles, 32)

        # ---- 4. Classify each article ----

        logits = self.classifier(article_embeddings)

        # Shape: (num_articles, num_classes)

        return logits
