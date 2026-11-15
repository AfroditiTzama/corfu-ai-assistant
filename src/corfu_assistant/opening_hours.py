from corfu_assistant.models import OpeningHoursEligibility, POI, Weekday


def opening_hours_eligibility(poi: POI, weekday: Weekday) -> OpeningHoursEligibility:
    if weekday not in ("mon", "tue", "wed", "thu", "fri", "sat", "sun"):
        raise ValueError(f"Unknown fixture weekday: {weekday!r}")

    schedule = poi.opening_hours_fixture
    if schedule is None:
        return OpeningHoursEligibility(status="unknown")

    window = getattr(schedule, weekday)
    if window is None:
        return OpeningHoursEligibility(status="closed")
    return OpeningHoursEligibility(status="scheduled", opening_window=window)
