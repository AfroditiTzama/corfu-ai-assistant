from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from corfu_assistant.data import load_dataset
from corfu_assistant.models import POIMetadataFilters
from corfu_assistant.opening_hours import opening_hours_eligibility
from corfu_assistant.repository import InvalidMetadataFilter, POIRepository


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeterministicEmbeddings:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.vectors_by_prefix: dict[str, list[float]] = {}
        self.query_vector = [1.0, 0.0]

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        values = list(texts)
        self.batches.append(values)
        vectors = []
        for text in values:
            if text == "query":
                vectors.append(self.query_vector)
                continue
            vector = next(
                (
                    value
                    for prefix, value in self.vectors_by_prefix.items()
                    if text.startswith(prefix)
                ),
                [0.0, 1.0],
            )
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


def make_repository(embedding_model: DeterministicEmbeddings | None = None) -> POIRepository:
    dataset = load_dataset(PROJECT_ROOT / "corfu_dataset.json")
    return POIRepository(dataset, embedding_model=embedding_model)


def test_metadata_filtering_supports_actual_category_tags_setting_and_family_values() -> None:
    repository = make_repository()

    history_outdoors = repository.filter_metadata(
        POIMetadataFilters(category=" history ", setting="outdoor")
    )
    assert history_outdoors
    assert all(poi.category == "history" and poi.setting == "outdoor" for poi in history_outdoors)

    museum_tagged = repository.filter_metadata(POIMetadataFilters(tags=["museum"]))
    assert museum_tagged
    assert all("museum" in poi.tags for poi in museum_tagged)

    family_suitable = repository.filter_metadata(
        POIMetadataFilters(family_suitability_fixture=True)
    )
    assert family_suitable
    assert all(poi.family_suitability_fixture is True for poi in family_suitable)

    unspecified = repository.filter_metadata(
        POIMetadataFilters(family_suitability_fixture="unknown")
    )
    assert unspecified
    assert all(poi.family_suitability_fixture is None for poi in unspecified)


def test_mixed_setting_is_not_treated_as_indoor() -> None:
    repository = make_repository()

    indoor = repository.filter_metadata(POIMetadataFilters(setting="indoor"))
    mixed = repository.filter_metadata(POIMetadataFilters(setting="mixed"))

    assert indoor
    assert mixed
    assert all(poi.setting == "indoor" for poi in indoor)
    assert all(poi.setting == "mixed" for poi in mixed)
    assert not {poi.id for poi in indoor} & {poi.id for poi in mixed}


def test_invalid_filter_values_are_reported() -> None:
    repository = make_repository()

    with pytest.raises(InvalidMetadataFilter, match="Unknown POI category"):
        repository.filter_metadata({"category": "museum"})
    with pytest.raises(InvalidMetadataFilter, match="Unknown POI tags"):
        repository.filter_metadata({"tags": ["not-a-dataset-tag"]})
    with pytest.raises(ValidationError):
        repository.filter_metadata("")
    with pytest.raises(ValidationError):
        repository.filter_metadata({"setting": "inside"})
    with pytest.raises(ValidationError):
        repository.filter_metadata({"tags": [""]})


def test_valid_but_conflicting_filters_return_no_candidates() -> None:
    repository = make_repository()

    results = repository.filter_metadata({"category": "history", "tags": ["seafood"]})

    assert results == ()


def test_opening_hours_eligibility_preserves_unknown_closed_and_scheduled() -> None:
    repository = make_repository()

    unknown = opening_hours_eligibility(repository.get_by_id("CF-004"), "mon")
    closed = opening_hours_eligibility(repository.get_by_id("CF-002"), "tue")
    scheduled = opening_hours_eligibility(repository.get_by_id("CF-001"), "tue")

    assert unknown.status == "unknown" and unknown.opening_window is None
    assert closed.status == "closed" and closed.opening_window is None
    assert scheduled.status == "scheduled" and scheduled.opening_window is not None
    with pytest.raises(ValueError, match="Unknown fixture weekday"):
        opening_hours_eligibility(repository.get_by_id("CF-001"), "holiday")


def test_hybrid_ranking_is_stable_and_returns_record_references() -> None:
    encoder = DeterministicEmbeddings()
    repository = make_repository(encoder)
    pois = repository.get_all()
    encoder.vectors_by_prefix = {
        f"Name: {pois[1].name}\n": [1.0, 0.0],
        f"Name: {pois[0].name}\n": [0.6, 0.8],
    }

    results = repository.hybrid_search("query", top_k=2)

    assert [result.poi_id for result in results] == [pois[1].id, pois[0].id]
    assert results[0].relevance_score == pytest.approx(0.4)
    assert results[0].source_urls == pois[1].source_urls
    assert results[0].description_provenance == pois[1].description_provenance


def test_metadata_filtering_precedes_semantic_ranking() -> None:
    encoder = DeterministicEmbeddings()
    repository = make_repository(encoder)
    excluded = next(poi for poi in repository.get_all() if poi.setting != "outdoor")
    eligible = next(poi for poi in repository.get_all() if poi.setting == "outdoor")
    encoder.vectors_by_prefix = {f"Name: {excluded.name}\n": [1.0, 0.0]}

    results = repository.hybrid_search("query", {"setting": "outdoor"}, top_k=3)

    assert results
    assert all(repository.get_by_id(result.poi_id).setting == "outdoor" for result in results)
    assert excluded.id not in {result.poi_id for result in results}
    assert results[0].poi_id == eligible.id
    assert results[0].relevance_score == pytest.approx(-0.3)


def test_exact_name_lexical_signal_ranks_the_named_poi_first() -> None:
    repository = make_repository(DeterministicEmbeddings())

    results = repository.hybrid_search("Old Fortress", top_k=5)

    assert results[0].poi_id == "CF-001"


def test_exclusion_filters_remove_categories_tags_and_specific_pois() -> None:
    repository = make_repository(DeterministicEmbeddings())

    results = repository.filter_metadata(
        {
            "excluded_categories": ["nightlife"],
            "excluded_tags": ["museum"],
            "excluded_poi_ids": ["CF-023"],
        }
    )
    result_ids = {poi.id for poi in results}

    assert "CF-023" not in result_ids
    assert all(poi.category != "nightlife" for poi in results)
    assert all("museum" not in poi.tags for poi in results)


def test_unknown_exclusion_metadata_is_rejected() -> None:
    repository = make_repository(DeterministicEmbeddings())

    with pytest.raises(InvalidMetadataFilter, match="Unknown excluded POI IDs"):
        repository.filter_metadata({"excluded_poi_ids": ["CF-999"]})
    with pytest.raises(InvalidMetadataFilter, match="Unknown excluded POI tags"):
        repository.filter_metadata({"excluded_tags": ["not-a-dataset-tag"]})


def test_empty_hybrid_result_does_not_initialize_embeddings() -> None:
    encoder = DeterministicEmbeddings()
    repository = make_repository(encoder)

    results = repository.hybrid_search(
        "query",
        {"category": "history", "tags": ["seafood"]},
    )

    assert results == []
    assert not repository.semantic_index_initialized
    assert encoder.batches == []


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_is_rejected(top_k: int | float) -> None:
    repository = make_repository(DeterministicEmbeddings())

    with pytest.raises(ValidationError):
        repository.hybrid_search("query", top_k=top_k)


def test_query_must_not_be_blank() -> None:
    repository = make_repository(DeterministicEmbeddings())

    with pytest.raises(ValidationError, match="query must not be blank"):
        repository.hybrid_search("   ")


def test_semantic_index_is_initialized_once_and_only_queries_are_reembedded() -> None:
    encoder = DeterministicEmbeddings()
    repository = make_repository(encoder)

    repository.initialize_semantic_index()
    repository.initialize_semantic_index()
    assert len(encoder.batches) == 1
    assert len(encoder.batches[0]) == repository.count
    assert repository.semantic_index_initialized

    repository.hybrid_search("query")
    repository.hybrid_search("query", top_k=1)

    assert len(encoder.batches) == 3
    assert encoder.batches[1:] == [["query"], ["query"]]


def test_invalid_embedding_shape_fails_explicitly() -> None:
    class InvalidEmbeddings:
        def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
            return np.ones((1, 0), dtype=np.float32)

    repository = make_repository(InvalidEmbeddings())

    from corfu_assistant.retrieval import RetrievalError

    with pytest.raises(RetrievalError, match="invalid shape"):
        repository.initialize_semantic_index()
