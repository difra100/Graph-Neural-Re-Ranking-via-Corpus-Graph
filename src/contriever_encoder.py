"""Contriever bi-encoder compatible with the TctColBert interface used in datamodule.py.

Contriever (Izacard et al. 2021) uses mean pooling over all token embeddings
(no special [Q]/[D] prefixes, no CLS extraction). Both variants are supported:
  facebook/contriever          — NOT fine-tuned on MS MARCO (weaker → more room for GNN)
  facebook/contriever-msmarco  — fine-tuned on MS MARCO (stronger baseline)
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from more_itertools import chunked


class ContrieverEncoder:
    """Mean-pool bi-encoder wrapping facebook/contriever or facebook/contriever-msmarco."""

    def __init__(self, model_name: str = 'facebook/contriever',
                 batch_size: int = 128, device: str = 'cuda'):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask_expanded).sum(1)
        count = mask_expanded.sum(1).clamp(min=1e-9)
        return summed / count

    def _encode(self, texts, max_length: int = 512) -> np.ndarray:
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, self.batch_size):
                inps = self.tokenizer(
                    list(chunk), padding=True, truncation=True,
                    max_length=max_length, return_tensors='pt'
                )
                inps = {k: v.to(self.device) for k, v in inps.items()}
                out = self.model(**inps).last_hidden_state
                emb = self._mean_pool(out, inps['attention_mask'])
                results.append(emb.cpu().numpy())
        return np.concatenate(results, axis=0) if results else np.empty((0, 768))

    def encode_queries(self, texts, batch_size=None) -> np.ndarray:
        return self._encode(texts, max_length=64)

    def encode_docs(self, texts, batch_size=None) -> np.ndarray:
        return self._encode(texts, max_length=512)

    def __repr__(self):
        return f'ContrieverEncoder({self.model_name!r})'
