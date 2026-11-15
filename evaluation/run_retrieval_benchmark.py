#!/usr/bin/env python3
"""Run the human-labeled retrieval benchmark against the configured embedder."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corfu_assistant.data import load_dataset  # noqa: E402
from corfu_assistant.models import POIMetadataFilters  # noqa: E402
from corfu_assistant.repository import POIRepository  # noqa: E402


BENCHMARK_PATH = ROOT / "evaluation" / "retrieval_benchmark.json"
DATASET_PATH = ROOT / "corfu_dataset.json"


def _query_text(case: dict[str, Any]) -> str:
    conversation = case.get("conversation")
    if conversation:
        return " ".join((conversation["first"], conversation["follow_up"]))
    return case["query"]


def _repo_filters(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: values[key]
        for key in (
            "category",
            "tags",
            "setting",
            "family_suitability_fixture",
            "excluded_categories",
            "excluded_tags",
            "excluded_poi_ids",
        )
        if key in values
    }


def _passes_all_filters(poi: Any, filters: dict[str, Any]) -> bool:
    if "category" in filters and poi.category.casefold() != filters["category"].casefold():
        return False
    if "tags" in filters and not {
        tag.casefold() for tag in filters["tags"]
    }.issubset({tag.casefold() for tag in poi.tags}):
        return False
    if "setting" in filters and poi.setting != filters["setting"]:
        return False
    if (
        "family_suitability_fixture" in filters
        and poi.family_suitability_fixture is not filters["family_suitability_fixture"]
    ):
        return False
    if poi.id in filters.get("excluded_poi_ids", []):
        return False
    excluded_tags = {tag.casefold() for tag in filters.get("excluded_tags", [])}
    if excluded_tags.intersection(tag.casefold() for tag in poi.tags):
        return False
    if poi.category.casefold() in {
        category.casefold() for category in filters.get("excluded_categories", [])
    }:
        return False
    return True


def _dcg(relevances: list[int]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances)
    )


def _query_metrics(
    case: dict[str, Any],
    ranked_ids: list[str],
    eligible_count: int,
) -> dict[str, Any]:
    relevance = case["relevant"]
    relevant_ids = set(relevance)
    reciprocal_rank = next(
        (1 / (rank + 1) for rank, poi_id in enumerate(ranked_ids) if poi_id in relevant_ids),
        0.0,
    )
    dcg = _dcg([relevance.get(poi_id, 0) for poi_id in ranked_ids[:5]])
    ideal = _dcg(sorted(relevance.values(), reverse=True)[:5])
    metrics: dict[str, Any] = {
        "id": case["id"],
        "category": case["category"],
        "recall_at_3": (
            len(set(ranked_ids[:3]) & relevant_ids) / len(relevant_ids)
            if relevant_ids else None
        ),
        "recall_at_5": (
            len(set(ranked_ids[:5]) & relevant_ids) / len(relevant_ids)
            if relevant_ids else None
        ),
        "precision_at_3": len(set(ranked_ids[:3]) & relevant_ids) / 3,
        "precision_at_5": len(set(ranked_ids[:5]) & relevant_ids) / 5,
        "mrr": reciprocal_rank,
        "ndcg_at_5": dcg / ideal if ideal else None,
        "duplicate_count": len(ranked_ids) - len(set(ranked_ids)),
        "hard_filter_violations": 0,
        "hard_filter_evaluated_count": 0,
        "returned_count": len(ranked_ids),
        "eligible_count": eligible_count,
    }

    if "requested_count" in case:
        requested = case["requested_count"]
        selected = ranked_ids[:requested]
        target_count = min(requested, eligible_count)
        metrics["requested_count"] = requested
        metrics["count_adherence"] = float(len(selected) == target_count)
        metrics["irrelevant_recommendation_rate"] = (
            sum(poi_id not in relevant_ids for poi_id in selected) / len(selected)
            if selected else 0.0
        )
        metrics["selected_count"] = len(selected)
    return metrics


def run_benchmark() -> dict[str, Any]:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    dataset = load_dataset(DATASET_PATH)
    pois_by_id = {poi.id: poi for poi in dataset.pois}
    cases = benchmark["queries"]
    if len(cases) < 40:
        raise ValueError(f"Benchmark must contain at least 40 queries, found {len(cases)}")
    for case in cases:
        missing = set(case["relevant"]) - pois_by_id.keys()
        if missing:
            raise ValueError(f"{case['id']} references unknown POI IDs: {sorted(missing)}")
        for poi_id in case["relevant"]:
            poi = pois_by_id[poi_id]
            filters = case.get("filters", {})
            if not _passes_all_filters(poi, filters):
                raise ValueError(
                    f"{case['id']} labels ineligible POI {poi_id} as relevant"
                )

    repository = POIRepository(dataset)
    query_results: list[dict[str, Any]] = []
    warm_latencies: list[float] = []
    cold_latency_ms: float | None = None

    for index, case in enumerate(cases):
        query = _query_text(case)
        filters = case.get("filters", {})
        started = time.perf_counter()
        results = repository.hybrid_search(
            query,
            filters=POIMetadataFilters.model_validate(_repo_filters(filters)),
            top_k=benchmark["top_k"],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        if index == 0:
            cold_latency_ms = elapsed_ms
        else:
            warm_latencies.append(elapsed_ms)

        ranked_ids = [result.poi_id for result in results]
        fully_eligible = [
            poi for poi in dataset.pois if _passes_all_filters(poi, filters)
        ]
        query_metrics = _query_metrics(case, ranked_ids, len(fully_eligible))
        invalid_ids = [
            poi_id
            for poi_id in ranked_ids
            if not _passes_all_filters(pois_by_id[poi_id], filters)
        ]
        query_metrics["hard_filter_violations"] = len(invalid_ids)
        query_metrics["hard_filter_evaluated_count"] = (
            len(ranked_ids) if filters else 0
        )
        query_metrics["hard_filter_violation_ids"] = invalid_ids
        query_metrics["retrieval_latency_ms"] = elapsed_ms
        query_metrics["top_ids"] = ranked_ids
        if "requested_count" in case:
            selected = ranked_ids[: case["requested_count"]]
            relevant = case["relevant"]
            query_metrics["irrelevant_recommendation_rate"] = (
                sum(poi_id not in relevant for poi_id in selected) / len(selected)
                if selected else 0.0
            )
        query_results.append(query_metrics)

    metric_names = (
        "recall_at_3",
        "recall_at_5",
        "precision_at_3",
        "precision_at_5",
        "mrr",
        "ndcg_at_5",
    )

    def average(rows: list[dict[str, Any]], key: str) -> float:
        values = [row[key] for row in rows if row.get(key) is not None]
        return statistics.fmean(values) if values else 0.0

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in query_results:
        categories[result["category"]].append(result)
    summary: dict[str, Any] = {
        key: average(query_results, key) for key in metric_names
    }
    summary.update(
        {
            "query_count": len(cases),
            "hard_filter_violation_rate": (
                sum(row["hard_filter_violations"] for row in query_results)
                / sum(row["hard_filter_evaluated_count"] for row in query_results)
                if any(row["hard_filter_evaluated_count"] for row in query_results)
                else 0.0
            ),
            "duplicate_recommendation_rate": (
                sum(row["duplicate_count"] for row in query_results)
                / sum(row["returned_count"] for row in query_results)
                if any(row["returned_count"] for row in query_results) else 0.0
            ),
            "recommendation_count_adherence": average(
                [row for row in query_results if "count_adherence" in row],
                "count_adherence",
            ),
            "irrelevant_recommendation_rate": average(
                [
                    row for row in query_results
                    if "irrelevant_recommendation_rate" in row
                ],
                "irrelevant_recommendation_rate",
            ),
            "cold_first_query_latency_ms": cold_latency_ms,
            "embedding_index_initialization_ms": (
                repository.semantic_index_initialization_ms
            ),
            "warm_latency_median_ms": (
                statistics.median(warm_latencies) if warm_latencies else None
            ),
            "warm_latency_p95_ms": (
                sorted(warm_latencies)[int((len(warm_latencies) - 1) * 0.95)]
                if warm_latencies else None
            ),
            "by_category": {
                category: {
                    **{key: average(rows, key) for key in metric_names},
                    "query_count": len(rows),
                    "hard_filter_violation_count": sum(
                        row["hard_filter_violations"] for row in rows
                    ),
                    "hard_filter_evaluated_count": sum(
                        row["hard_filter_evaluated_count"] for row in rows
                    ),
                }
                for category, rows in sorted(categories.items())
            },
            "query_failures": [
                {
                    "id": row["id"],
                    "category": row["category"],
                    "recall_at_5": row["recall_at_5"],
                    "violations": row["hard_filter_violation_ids"],
                    "top_ids": row["top_ids"],
                }
                for row in query_results
                if (
                    (
                        row["recall_at_5"] is not None
                        and row["recall_at_5"] < 0.2
                        and row["eligible_count"] > 0
                    )
                    or row["hard_filter_violations"] > 0
                )
            ],
        }
    )
    return {"benchmark_version": benchmark["version"], "summary": summary, "queries": query_results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write full machine-readable results to JSON")
    args = parser.parse_args()
    report = run_benchmark()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
