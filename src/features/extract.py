"""Word-level feature extraction for GPT-2 (causal) and BERT (masked).

For a story we:
  1. Tokenize each surface word separately so we know which subword tokens
     belong to which word (alignment = list[list[int]]).
  2. Run the model over the concatenated subword ids (sliding window for
     contexts longer than the model's max length).
  3. Aggregate to one feature vector per word by mean-pooling subword
     hidden states across all layers, and (for causal) summing -log p over
     subwords to get word surprisal.

Outputs per story:
  embeddings : np.ndarray [num_words, num_layers, hidden_size]
  surprisal  : np.ndarray [num_words] or None
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm


def _align_subwords(words: list[str], tokenizer, prefix_space: bool) -> tuple[list[int], list[tuple[int, int]]]:
    """Return (input_ids, word_spans) where word_spans[i] = (start, end) in input_ids."""
    ids: list[int] = []
    spans: list[tuple[int, int]] = []
    for w in words:
        text = (" " + w) if prefix_space and len(ids) > 0 else w
        sub = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(sub) == 0:
            sub = [tokenizer.unk_token_id]
        start = len(ids)
        ids.extend(sub)
        spans.append((start, start + len(sub)))
    return ids, spans


@torch.no_grad()
def extract_causal(
    words_by_story: dict[int, list[str]],
    model_id: str,
    device: str,
    context_window: int = 1024,
    stride: int = 512,
    extract_layers: str | list[int] = "all",
    mixed_precision: bool = False,
) -> dict[int, dict[str, np.ndarray]]:
    """Extract GPT-2 surprisal and per-layer embeddings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, output_hidden_states=True).to(device).eval()
    dtype = torch.float16 if mixed_precision and device == "cuda" else torch.float32

    out: dict[int, dict[str, np.ndarray]] = {}
    for story, words in tqdm(words_by_story.items(), desc="causal stories"):
        ids, spans = _align_subwords(words, tok, prefix_space=True)
        n_tok = len(ids)

        # Sliding-window: for each window, keep predictions for tokens that were
        # not scored by an earlier window. Predictions are shifted by one.
        nll = np.zeros(n_tok, dtype=np.float64)
        nll_set = np.zeros(n_tok, dtype=bool)
        layer_cache: Optional[np.ndarray] = None  # [n_tok, n_layers, hidden]

        prev_end = 0
        start = 0
        while start < n_tok:
            end = min(start + context_window, n_tok)
            chunk = torch.tensor([ids[start:end]], device=device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=mixed_precision and device == "cuda"):
                outputs = model(chunk, output_hidden_states=True)
            logits = outputs.logits[0].float()  # [L, V]
            hidden = torch.stack(outputs.hidden_states, dim=0)[:, 0].float()  # [num_layers, L, H]

            if layer_cache is None:
                layer_cache = np.zeros((n_tok, hidden.shape[0], hidden.shape[-1]), dtype=np.float32)
            layer_cache[start:end] = hidden.permute(1, 0, 2).cpu().numpy()

            # p(token_t | token_<t) = softmax(logits[t-1])[token_t]
            lp = log_softmax(logits[:-1], dim=-1)
            tgt = chunk[0, 1:]
            token_nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu().numpy()
            # token_nll[i] corresponds to ids[start + 1 + i]
            for i, v in enumerate(token_nll):
                abs_pos = start + 1 + i
                if abs_pos < prev_end and start > 0:
                    continue  # already scored
                if abs_pos >= n_tok:
                    break
                nll[abs_pos] = v
                nll_set[abs_pos] = True
            prev_end = end
            if end == n_tok:
                break
            start += stride

        # First token has no left context -> surprisal undefined; leave as NaN.
        nll[~nll_set] = np.nan

        # Aggregate to word level.
        num_layers = layer_cache.shape[1]
        hidden_size = layer_cache.shape[2]
        n_words = len(words)
        embeds = np.zeros((n_words, num_layers, hidden_size), dtype=np.float32)
        surp = np.zeros(n_words, dtype=np.float32)
        for wi, (s, e) in enumerate(spans):
            embeds[wi] = layer_cache[s:e].mean(axis=0)
            surp[wi] = np.nansum(nll[s:e])  # summed -log p across subwords

        if extract_layers != "all":
            embeds = embeds[:, list(extract_layers), :]

        out[story] = {"embeddings": embeds, "surprisal": surp}
    return out


@torch.no_grad()
def extract_masked(
    words_by_story: dict[int, list[str]],
    model_id: str,
    device: str,
    context_window: int = 512,
    stride: int = 256,
    extract_layers: str | list[int] = "all",
    mixed_precision: bool = False,
) -> dict[int, dict[str, np.ndarray]]:
    """Extract BERT per-layer embeddings (no surprisal at this stage)."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, output_hidden_states=True).to(device).eval()
    dtype = torch.float16 if mixed_precision and device == "cuda" else torch.float32

    cls, sep = tok.cls_token_id, tok.sep_token_id
    max_body = context_window - 2  # room for [CLS] and [SEP]

    out: dict[int, dict[str, np.ndarray]] = {}
    for story, words in tqdm(words_by_story.items(), desc="masked stories"):
        ids, spans = _align_subwords(words, tok, prefix_space=False)
        n_tok = len(ids)

        hidden_accum: Optional[np.ndarray] = None  # [n_tok, num_layers, H]
        counts = np.zeros(n_tok, dtype=np.int32)

        start = 0
        while start < n_tok:
            end = min(start + max_body, n_tok)
            chunk_ids = [cls] + ids[start:end] + [sep]
            chunk = torch.tensor([chunk_ids], device=device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=mixed_precision and device == "cuda"):
                outputs = model(chunk, output_hidden_states=True)
            hidden = torch.stack(outputs.hidden_states, dim=0)[:, 0].float()  # [num_layers, L, H]
            # Drop [CLS] / [SEP]
            hidden = hidden[:, 1:-1, :].permute(1, 0, 2).cpu().numpy()  # [body_len, num_layers, H]

            if hidden_accum is None:
                hidden_accum = np.zeros((n_tok, hidden.shape[1], hidden.shape[2]), dtype=np.float32)
            hidden_accum[start:end] += hidden
            counts[start:end] += 1
            if end == n_tok:
                break
            start += stride

        hidden_accum /= counts[:, None, None].clip(min=1)

        num_layers = hidden_accum.shape[1]
        hidden_size = hidden_accum.shape[2]
        embeds = np.zeros((len(words), num_layers, hidden_size), dtype=np.float32)
        for wi, (s, e) in enumerate(spans):
            embeds[wi] = hidden_accum[s:e].mean(axis=0)

        if extract_layers != "all":
            embeds = embeds[:, list(extract_layers), :]

        out[story] = {"embeddings": embeds, "surprisal": None}
    return out
