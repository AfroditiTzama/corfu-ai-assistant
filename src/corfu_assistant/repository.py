from pathlib import Path
import logging
import re
import time
from threading import Lock
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from corfu_assistant.models import CorfuDataset, POI
from corfu_assistant.models import HybridSearchRequest, POIMetadataFilters, RetrievalResult
from corfu_assistant.retrieval import EmbeddingModel, RetrievalError, SentenceTransformerEmbedder


_LOGGER = logging.getLogger(__name__)


class InvalidMetadataFilter(ValueError):
    """Raised when a filter value is absent from the POI dataset."""


class POIRepository:
    def __init__(
        self,
        dataset: CorfuDataset,
        *,
        embedding_model: EmbeddingModel | None = None,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        embedding_cache_dir: Path | None = None,
    ) -> None:
        self._pois = tuple(dataset.pois)
        self._pois_by_id = {poi.id: poi for poi in self._pois}
        self._poi_indices_by_id = {poi.id: index for index, poi in enumerate(self._pois)}
        self._categories = {poi.category.casefold() for poi in self._pois}
        self._tags = {tag.casefold() for poi in self._pois for tag in poi.tags}
        self._embedding_model = embedding_model
        self._embedding_model_name = embedding_model_name
        self._embedding_cache_dir = embedding_cache_dir
        self._semantic_embeddings: NDArray[np.float32] | None = None
        self._semantic_index_initialization_ms: float | None = None
        self._semantic_index_lock = Lock()

    @property
    def count(self) -> int:
        return len(self._pois)

    @property
    def semantic_index_initialized(self) -> bool:
        return self._semantic_embeddings is not None

    @property
    def semantic_index_initialization_ms(self) -> float | None:
        return self._semantic_index_initialization_ms

    @property
    def metadata_categories(self) -> tuple[str, ...]:
        return tuple(sorted({poi.category for poi in self._pois}))

    @property
    def metadata_tags(self) -> tuple[str, ...]:
        return tuple(sorted({tag for poi in self._pois for tag in poi.tags}))

    def get_all(self) -> tuple[POI, ...]:
        return self._pois

    def get_by_id(self, poi_id: str) -> POI | None:
        return self._pois_by_id.get(poi_id)

    def filter_metadata(self, filters: POIMetadataFilters | Mapping[str, object] | None = None) -> tuple[POI, ...]:
        validated = POIMetadataFilters.model_validate({} if filters is None else filters)
        category = validated.category.casefold() if validated.category is not None else None
        tags = {tag.casefold() for tag in validated.tags}
        excluded_categories = {value.casefold() for value in validated.excluded_categories}
        excluded_tags = {value.casefold() for value in validated.excluded_tags}
        excluded_poi_ids = set(validated.excluded_poi_ids)

        if category is not None and category not in self._categories:
            raise InvalidMetadataFilter(f"Unknown POI category: {validated.category!r}")
        unknown_tags = tags - self._tags
        if unknown_tags:
            raise InvalidMetadataFilter(f"Unknown POI tags: {sorted(unknown_tags)}")
        unknown_excluded_categories = excluded_categories - self._categories
        if unknown_excluded_categories:
            raise InvalidMetadataFilter(
                f"Unknown excluded POI categories: {sorted(unknown_excluded_categories)}"
            )
        unknown_excluded_tags = excluded_tags - self._tags
        if unknown_excluded_tags:
            raise InvalidMetadataFilter(
                f"Unknown excluded POI tags: {sorted(unknown_excluded_tags)}"
            )
        unknown_poi_ids = excluded_poi_ids - self._pois_by_id.keys()
        if unknown_poi_ids:
            raise InvalidMetadataFilter(
                f"Unknown excluded POI IDs: {sorted(unknown_poi_ids)}"
            )

        match_unknown_family = (
            "family_suitability_fixture" in validated.model_fields_set
            and validated.family_suitability_fixture == "unknown"
        )
        family_value = validated.family_suitability_fixture

        return tuple(
            poi
            for poi in self._pois
            if (category is None or poi.category.casefold() == category)
            and poi.category.casefold() not in excluded_categories
            and (validated.setting is None or poi.setting == validated.setting)
            and tags.issubset({tag.casefold() for tag in poi.tags})
            and not excluded_tags.intersection(tag.casefold() for tag in poi.tags)
            and poi.id not in excluded_poi_ids
            and (
                family_value is None
                or (
                    poi.family_suitability_fixture is None
                    if match_unknown_family
                    else poi.family_suitability_fixture is family_value
                )
            )
        )

    def initialize_semantic_index(
        self,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        with self._semantic_index_lock:
            if self._semantic_embeddings is not None:
                if embedding_model is not None and embedding_model is not self._embedding_model:
                    raise RetrievalError("Semantic index is already initialized with another embedding model")
                return

            model = embedding_model or self._embedding_model
            if model is None:
                model = SentenceTransformerEmbedder(
                    self._embedding_model_name,
                    cache_folder=self._embedding_cache_dir,
                )

            started = time.perf_counter()
            documents = [self._semantic_document(poi) for poi in self._pois]
            if not documents:
                raise RetrievalError("Cannot initialize a semantic index without POIs")
            embeddings = self._encode(model, documents, expected_rows=len(documents))
            self._embedding_model = model
            self._semantic_embeddings = self._normalize(embeddings, "POI document")
            self._semantic_index_initialization_ms = (
                time.perf_counter() - started
            ) * 1000
            _LOGGER.info(
                "Embedding index initialized",
                extra={
                    "stage": "embedding_cold_start",
                    "poi_count": len(documents),
                    "duration_ms": round(self._semantic_index_initialization_ms, 2),
                },
            )

    def hybrid_search(
        self,
        query: str,
        filters: POIMetadataFilters | Mapping[str, object] | None = None,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        request = HybridSearchRequest.model_validate(
            {
                "query": query,
                "filters": {} if filters is None else filters,
                "top_k": top_k,
            }
        )
        candidates = self.filter_metadata(request.filters)
        if not candidates:
            return []

        self.initialize_semantic_index()
        if self._embedding_model is None or self._semantic_embeddings is None:
            raise RetrievalError("Semantic index initialization did not produce an index")

        query_embedding = self._encode(
            self._embedding_model,
            [request.query],
            expected_rows=1,
        )
        if query_embedding.shape[1] != self._semantic_embeddings.shape[1]:
            raise RetrievalError(
                "Embedding model returned inconsistent dimensions for documents and query"
            )
        normalized_query = self._normalize(query_embedding, "query")[0]
        candidate_indices = [self._poi_indices_by_id[poi.id] for poi in candidates]
        eligible_embeddings = self._semantic_embeddings[candidate_indices]
        semantic_scores = np.clip(eligible_embeddings @ normalized_query, -1.0, 1.0)
        lexical_scores = self._lexical_scores(request.query, candidates)
        combined_scores = (
            0.7 * ((semantic_scores + 1.0) / 2.0) + 0.3 * lexical_scores
        )
        scores = np.clip((2.0 * combined_scores) - 1.0, -1.0, 1.0)
        ranked_positions = np.argsort(-scores, kind="stable")[: request.top_k]

        return [
            RetrievalResult(
                poi_id=candidates[int(position)].id,
                relevance_score=float(scores[position]),
                source_urls=candidates[int(position)].source_urls,
                description_provenance=candidates[int(position)].description_provenance,
            )
            for position in ranked_positions
        ]

    @staticmethod
    def _semantic_document(poi: POI) -> str:
        return (
            f"Name: {poi.name}\n"
            f"Category: {poi.category}\n"
            f"Area: {poi.area}\n"
            f"Tags: {', '.join(poi.tags)}\n"
            f"Description: {poi.description}"
        )

    @classmethod
    def _lexical_scores(cls, query: str, candidates: tuple[POI, ...]) -> NDArray[np.float32]:
        query_terms = cls._tokens(query)
        if not query_terms:
            return np.zeros(len(candidates), dtype=np.float32)

        documents = [cls._tokens(cls._semantic_document(poi)) for poi in candidates]
        document_frequency = {
            term: sum(term in document for document in documents)
            for term in query_terms
        }
        term_weights = {
            term: float(np.log1p(
                (len(candidates) - frequency + 0.5) / (frequency + 0.5)
            ))
            for term, frequency in document_frequency.items()
        }
        weight_total = sum(term_weights.values())
        if weight_total == 0:
            return np.zeros(len(candidates), dtype=np.float32)

        normalized_query = cls._normalize_phrase(query)
        scores: list[float] = []
        for poi in candidates:
            name_terms = set(cls._tokens(poi.name))
            metadata_terms = set(cls._tokens(f"{poi.category} {' '.join(poi.tags)}"))
            area_terms = set(cls._tokens(poi.area))
            description_terms = set(cls._tokens(poi.description))

            def coverage(terms: set[str]) -> float:
                return sum(
                    term_weights[term]
                    for term in query_terms
                    if term in terms
                ) / weight_total

            name_score = coverage(name_terms)
            if cls._normalize_phrase(poi.name) in normalized_query:
                name_score = 1.0
            metadata_score = coverage(metadata_terms)
            area_score = coverage(area_terms)
            description_score = coverage(description_terms)
            scores.append(
                min(
                    1.0,
                    (0.50 * name_score)
                    + (0.25 * metadata_score)
                    + (0.10 * area_score)
                    + (0.15 * description_score),
                )
            )
        return np.asarray(scores, dtype=np.float32)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "can",
            "could", "for", "from", "i", "in", "is", "it", "me", "of", "on",
            "or", "place", "places", "please", "recommend", "show", "some",
            "the", "to", "what", "where", "which", "with", "near", "nearby",
            "around", "visit", "visiting",
        }
        return [
            token
            for token in re.findall(r"[a-z0-9]+", text.casefold())
            if token not in stop_words
        ]

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))

    @staticmethod
    def _encode(
        embedding_model: EmbeddingModel,
        texts: list[str],
        expected_rows: int,
    ) -> NDArray[np.float32]:
        try:
            embeddings = np.asarray(embedding_model.encode(texts), dtype=np.float32)
        except Exception as error:
            raise RetrievalError(f"Embedding model failed: {error}") from error
        if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows or embeddings.shape[1] == 0:
            raise RetrievalError(
                "Embedding model returned an invalid shape: "
                f"expected ({expected_rows}, dimension), got {embeddings.shape}"
            )
        if not np.isfinite(embeddings).all():
            raise RetrievalError("Embedding model returned non-finite values")
        return embeddings

    @staticmethod
    def _normalize(
        embeddings: NDArray[np.float32],
        kind: str,
    ) -> NDArray[np.float32]:
        scales = np.max(np.abs(embeddings), axis=1, keepdims=True)
        if np.any(scales == 0):
            raise RetrievalError(f"Embedding model returned a zero-length {kind} vector")
        scaled = embeddings / scales
        norms = np.linalg.norm(scaled, axis=1, keepdims=True)
        if not np.isfinite(norms).all():
            raise RetrievalError(f"Embedding model returned an invalid {kind} vector")
        return scaled / norms
