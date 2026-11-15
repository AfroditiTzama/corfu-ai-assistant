import os
from pathlib import Path

import pytest

from corfu_assistant.data import load_dataset
from corfu_assistant.repository import POIRepository


pytestmark = pytest.mark.integration
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    os.getenv("CORFU_RUN_MODEL_SMOKE") != "1",
    reason="Set CORFU_RUN_MODEL_SMOKE=1 to download/load the real embedding model",
)
def test_real_sentence_transformer_model_can_retrieve_dataset_pois() -> None:
    repository = POIRepository(load_dataset(PROJECT_ROOT / "corfu_dataset.json"))

    results = repository.hybrid_search("Venetian fortifications and Corfu history", top_k=3)

    assert results
    assert len(results) <= 3
    assert all(repository.get_by_id(result.poi_id) is not None for result in results)
