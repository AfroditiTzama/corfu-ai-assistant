#!/usr/bin/env python3
"""Run the exact DOTSOFT A–E demo prompts through the submitted HTTP API."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


PROMPTS: dict[str, tuple[str, ...]] = {
    "A": (
        "Plan a four-hour visit to Corfu focused on art and museums on Monday, October 5, 2026, from 13:00 to 17:00.",
    ),
    "B": (
        "Plan a four-hour walking itinerary in Corfu focused on historical attractions on Tuesday, October 6, 2026, from 11:00 to 15:00.",
        "Replace the second stop with something suitable for my 10-year-old daughter. Keep the rest of my itinerary and make sure we still finish by 15:00.",
    ),
    "C": (
        "Can I visit Old Fortress, New Fortress, Achilleion Palace, Mon Repos Museum of Palaiopolis, and Museum of Asian Art on foot in just 90 minutes on Tuesday, October 6, 2026, between 11:00 and 12:30? Create an itinerary if possible.",
    ),
    "D": (
        "Plan a four-hour itinerary in Corfu focused on outdoor historical attractions on Wednesday, October 7, 2026, from 13:00 to 17:00.",
        "Can you check the weather for Wednesday, October 7, 2026, afternoon and adapt my existing itinerary if needed while keeping the original 13:00 - 17:00 time window?",
    ),
    "E": (
        "What is the weather forecast for Corfu for the next 24 hours? Should I expect rain?",
    ),
}


def post_chat(base_url: str, message: str, conversation_id: str | None) -> dict[str, object]:
    payload: dict[str, object] = {"message": message, "request_id": str(uuid4())}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    request = Request(
        f"{base_url.rstrip('/')}/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"Unable to reach {base_url}: {error.reason}") from error


def weather_source(result: dict[str, object]) -> str | None:
    forecast = result.get("weather_forecast")
    if not isinstance(forecast, dict):
        replanning = result.get("weather_replanning")
        if isinstance(replanning, dict):
            forecast = replanning.get("forecast")
    return forecast.get("source") if isinstance(forecast, dict) else None


def run_scenario(base_url: str, scenario: str) -> None:
    print(f"\n=== Scenario {scenario} ===")
    conversation_id: str | None = None
    last_result: dict[str, object] | None = None
    for turn, prompt in enumerate(PROMPTS[scenario], start=1):
        print(f"\nUser {turn}: {prompt}")
        result = post_chat(base_url, prompt, conversation_id)
        conversation_id = str(result["conversation_id"])
        last_result = result
        print(f"Status: {result.get('status')}")
        print(f"Assistant: {result.get('response')}")
        planning = result.get("planning_result")
        if not isinstance(planning, dict):
            replanning = result.get("weather_replanning")
            planning = replanning.get("planning_result") if isinstance(replanning, dict) else None
        if isinstance(planning, dict):
            print(f"Selected POIs: {', '.join(planning.get('selected_poi_ids') or []) or '(none)'}")

    assert last_result is not None
    source = weather_source(last_result)
    if scenario == "D" and source != "synthetic_fixture":
        raise RuntimeError(f"Scenario D requires synthetic_fixture weather; received {source!r}")
    if scenario == "E" and (source is None or source == "synthetic_fixture"):
        raise RuntimeError(f"Scenario E requires a real live-weather response; received {source!r}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(PROMPTS),
        default=["A", "B", "C"],
        help="Scenarios to run. Run D on the fixture backend and E on the live backend.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if "D" in args.scenarios and "E" in args.scenarios:
        print(
            "Scenarios D and E require different backend weather modes; run them separately.",
            file=sys.stderr,
        )
        return 2
    try:
        for scenario in args.scenarios:
            run_scenario(args.base_url, scenario)
    except (KeyError, RuntimeError, ValueError) as error:
        print(f"Demo failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
