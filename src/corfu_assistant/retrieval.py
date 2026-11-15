from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


class EmbeddingModel(Protocol):
    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return one fixed-width vector for each input text."""


class RetrievalError(RuntimeError):
    """Raised when an embedding model or semantic index cannot be used."""


class SentenceTransformerEmbedder:
    """Lazy-to-use adapter around a cached Sentence Transformers model."""

    def __init__(self, model_name: str, cache_folder: Path | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RetrievalError(
                "Semantic retrieval requires the 'retrieval' dependency extra"
            ) from error
        except Exception as error:
            raise RetrievalError(
                f"Unable to import the configured Sentence Transformers stack: {error}"
            ) from error

        try:
            self._model = SentenceTransformer(
                model_name,
                cache_folder=str(cache_folder) if cache_folder is not None else None,
            )
        except Exception as error:
            raise RetrievalError(
                f"Unable to initialize embedding model {model_name!r}: {error}"
            ) from error
        self._encode_lock = Lock()

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        try:
            with self._encode_lock:
                embeddings = self._model.encode(
                    list(texts),
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as error:
            raise RetrievalError(f"Embedding generation failed: {error}") from error
