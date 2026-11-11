"""Utilities for loading pretrained word embeddings."""

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def build_embedding_matrix(
    vocab: Mapping[str, int],
    glove_path,
    emb_dim: int = 300,
    seed: int = 42,
    cache_path=None,
) -> np.ndarray:
    """Build a vocabulary-indexed matrix initialized from GloVe when available."""
    if emb_dim < 1:
        raise ValueError("emb_dim must be at least 1.")
    if not vocab:
        raise ValueError("vocab must not be empty.")

    vocab_size = len(vocab)
    indices = list(vocab.values())
    if any(not isinstance(index, (int, np.integer)) for index in indices):
        raise ValueError("Vocabulary indices must be integers.")
    if sorted(indices) != list(range(vocab_size)):
        raise ValueError("Vocabulary indices must be unique and contiguous from 0.")

    glove_path = Path(glove_path)
    if not glove_path.is_file():
        raise FileNotFoundError(f"GloVe file not found: {glove_path}")
    if cache_path is None:
        cache_path = glove_path.with_name(
            f"glove_matrix_{vocab_size}x{emb_dim}.npy"
        )
    else:
        cache_path = Path(cache_path)

    cache_metadata = {
        "vocab_sha256": hashlib.sha256(
            json.dumps(sorted(vocab.items(), key=lambda pair: pair[1])).encode("utf-8")
        ).hexdigest(),
        "glove_path": str(glove_path.resolve()),
        "glove_size": glove_path.stat().st_size,
        "glove_mtime_ns": glove_path.stat().st_mtime_ns,
        "emb_dim": emb_dim,
        "seed": seed,
    }
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if cache_path.is_file() and metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as file:
            if json.load(file) == cache_metadata:
                cached = np.load(cache_path, allow_pickle=False)
                if (
                    cached.shape == (vocab_size, emb_dim)
                    and cached.dtype == np.float32
                    and np.count_nonzero(cached[vocab.get("<pad>", 0)]) == 0
                ):
                    print(f"[GloVe] Loaded cached matrix from {cache_path}")
                    return cached

    rng = np.random.default_rng(seed)
    embedding_matrix = rng.normal(
        0, 0.01, size=(vocab_size, emb_dim)
    ).astype(np.float32)
    pad_idx = vocab.get("<pad>", 0)
    embedding_matrix[pad_idx] = 0.0

    special_indices = {vocab[word] for word in ("<pad>", "<unk>") if word in vocab}
    real_word_count = vocab_size - len(special_indices)
    found = 0
    with open(glove_path, "r", encoding="utf-8") as file:
        for line in file:
            word, separator, values = line.rstrip().partition(" ")
            index = vocab.get(word)
            if not separator or index is None:
                continue
            vector = np.fromstring(values, sep=" ", dtype=np.float32)
            if vector.size != emb_dim or not np.isfinite(vector).all():
                continue
            embedding_matrix[index] = vector
            if index not in special_indices:
                found += 1

    coverage = found / real_word_count * 100 if real_word_count else 0.0
    print(f"[GloVe] Found {found}/{real_word_count} vocabulary words in GloVe")
    print(f"[GloVe] Coverage: {coverage:.2f}% of real words")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_temp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    metadata_temp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    try:
        with matrix_temp.open("wb") as file:
            np.save(file, embedding_matrix, allow_pickle=False)
        with metadata_temp.open("w", encoding="utf-8") as file:
            json.dump(cache_metadata, file, sort_keys=True)
        os.replace(matrix_temp, cache_path)
        os.replace(metadata_temp, metadata_path)
    finally:
        matrix_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
    return embedding_matrix
