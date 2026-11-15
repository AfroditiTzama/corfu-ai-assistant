#!/usr/bin/env python3
"""Reproducible coordinate, area-cluster, and walking-estimate audit."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "corfu_dataset.json"
TOWN_ANCHOR_ID = "CF-033"  # Spianada Square, town-centre reference point
ROUTING_MULTIPLIER = 1.4
WALKING_SPEED_KMH = 4.5


def distance_km(a: dict[str, float], b: dict[str, float]) -> float:
    lat_a = math.radians(a["latitude"])
    lat_b = math.radians(b["latitude"])
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(b["longitude"] - a["longitude"])
    haversine = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, haversine)))


def run_audit() -> dict[str, object]:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    pois = dataset["pois"]
    by_id = {poi["id"]: poi for poi in pois}
    anchor = by_id[TOWN_ANCHOR_ID]["location"]
    area_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    rows: list[dict[str, object]] = []
    invalid_coordinates: list[str] = []
    for poi in pois:
        loc = poi["location"]
        if not (-90 <= loc["latitude"] <= 90 and -180 <= loc["longitude"] <= 180):
            invalid_coordinates.append(poi["id"])
        nearest = min(
            (
                (distance_km(loc, other["location"]), other)
                for other in pois if other["id"] != poi["id"]
            ),
            key=lambda pair: pair[0],
        )
        direct_from_town = distance_km(anchor, loc)
        route_km = nearest[0] * ROUTING_MULTIPLIER
        rows.append(
            {
                "id": poi["id"],
                "name": poi["name"],
                "area": poi["area"],
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "distance_from_spianada_km": round(direct_from_town, 3),
                "nearest_poi_id": nearest[1]["id"],
                "nearest_straight_line_km": round(nearest[0], 3),
                "nearest_estimated_walking_km": round(route_km, 3),
                "nearest_estimated_walking_minutes": math.ceil(route_km / WALKING_SPEED_KMH * 60),
            }
        )
        area_groups[poi["area"]].append(poi)

    all_pairs = [
        distance_km(a["location"], b["location"])
        for index, a in enumerate(pois)
        for b in pois[index + 1 :]
    ]
    group_spreads = []
    for area, members in sorted(area_groups.items()):
        maximum = max(
            distance_km(a["location"], b["location"])
            for index, a in enumerate(members)
            for b in members[index + 1 :]
        ) if len(members) > 1 else 0.0
        group_spreads.append(
            {"area": area, "poi_count": len(members), "max_pair_spread_km": round(maximum, 3)}
        )

    return {
        "poi_count": len(pois),
        "coordinate_range": {
            "latitude": [min(p["location"]["latitude"] for p in pois), max(p["location"]["latitude"] for p in pois)],
            "longitude": [min(p["location"]["longitude"] for p in pois), max(p["location"]["longitude"] for p in pois)],
        },
        "invalid_coordinates": invalid_coordinates,
        "town_anchor": {"poi_id": TOWN_ANCHOR_ID, "name": by_id[TOWN_ANCHOR_ID]["name"]},
        "all_pair_distances_km": {
            "pair_count": len(all_pairs),
            "minimum": round(min(all_pairs), 3),
            "median": round(sorted(all_pairs)[len(all_pairs) // 2], 3),
            "maximum": round(max(all_pairs), 3),
        },
        "walking_estimation_assumptions": {
            "route_to_straight_line_multiplier": ROUTING_MULTIPLIER,
            "walking_speed_kmh": WALKING_SPEED_KMH,
            "minutes_formula": "ceil(estimated_walking_distance_km / speed_kmh * 60)",
            "limitation": "No pedestrian network, elevation, gates, stairs, crossings, or accessibility routing is modeled.",
        },
        "area_cluster_spreads": group_spreads,
        "pois": sorted(rows, key=lambda row: row["id"]),
        "manual_verification": {
            "CF-026": "Coordinates and Old Town label align with 25 Nikiforou Theotoki Street in the Old Town; see docs/geographic-audit.md.",
            "CF-031": "Coordinates match the official museum map pin; area corrected to Garitsa / Corfu Town.",
            "CF-021": "The palace is at Gastouri, southwest of town; area corrected from Perama to Gastouri.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON audit output")
    args = parser.parse_args()
    report = run_audit()
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "pois"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
