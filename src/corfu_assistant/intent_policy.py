"""Normalize accidental literal theme tags while preserving explicit filters."""
from __future__ import annotations

import logging
import re
from collections.abc import Collection, Mapping, Sequence

from corfu_assistant.models import UserIntent

_LOGGER = logging.getLogger(__name__)

_FILTER_FIELDS = {
    "filter_category": "category", "filter_tags": "tags", "filter_setting": "setting",
    "filter_family_suitability": "family_suitability_fixture",
    "filter_excluded_categories": "excluded_categories", "filter_excluded_tags": "excluded_tags",
    "filter_excluded_poi_ids": "excluded_poi_ids",
}
_RESTRICTION = re.compile(r"\b(?:only|must|require[ds]?|restrict\w*|exclusively|strictly|nothing but|solely|every|each|all)\b", re.I)
_NEGATION = re.compile(r"\b(?:no|not|without|avoid|exclude\w*|except|don't|do not|leave out|omit|skip|keep out)\b", re.I)
_SOFT_CUE = re.compile(r"\b(?:focus(?:ed|ing)? on|interested in|prefer|enjoy|would like|inspired by)\b", re.I)
_ACTUAL_WEATHER_REQUEST = re.compile(
    r"\b(?:forecast|current(?:ly)?|right now|today|tomorrow|temperature|"
    r"is it raining|will it rain|expect rain|check (?:the )?weather|"
    r"(?:what|how) (?:is )?(?:the )?weather)\b",
    re.I,
)


def _text(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalize_contextual_activity_request(intent: UserIntent, message: str) -> UserIntent:
    """Route hypothetical weather-suitability requests without calling live weather."""
    scope = intent.weather_request
    if (
        intent.intent != "live_weather_query"
        or scope is None
        or not scope.activity_recommendation
        or scope.relative_horizon is not None
        or scope.duration_hours is not None
        or scope.forecast_date is not None
        or scope.start_time is not None
        or scope.end_time is not None
        or _ACTUAL_WEATHER_REQUEST.search(message)
    ):
        return intent
    data = intent.model_dump(mode="python")
    data["intent"] = "poi_recommendation"
    _LOGGER.info(
        "Hypothetical weather activity request routed to POI recommendations",
        extra={"stage": "intent_normalization", "decision": "contextual_recommendation"},
    )
    return UserIntent.model_validate(data)


def _value_pattern(value: str) -> re.Pattern[str]:
    """Match a metadata value and its regular English plural form."""
    phrase = _text(value.replace("_", " "))
    head, separator, word = phrase.rpartition(" ")
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        plural = word[:-1] + "ies"
    elif word.endswith(("s", "x", "z", "ch", "sh")):
        plural = word + "es"
    else:
        plural = word + "s"
    forms = [phrase, f"{head}{separator}{plural}" if separator else plural]
    return re.compile(r"\b(?:" + "|".join(re.escape(item) for item in sorted(set(forms), key=len, reverse=True)) + r")\b")


def _explicitly_restricts_value(message: str, value: str) -> bool:
    """Conservatively identify restriction language scoped to a metadata value."""
    normalized = _text(message)
    words = [(match.group(), match.start()) for match in re.finditer(r"\b[\w]+\b", normalized)]
    subject = value.replace("_", " ").casefold()
    subject_words = subject.split()
    positions = [index for index in range(len(words) - len(subject_words) + 1)
                 if [word for word, _ in words[index:index + len(subject_words)]] == subject_words
                 or (len(subject_words) == 1 and words[index][0] == subject + "s")]
    cues = [index for index, (_, start) in enumerate(words)
            if _RESTRICTION.match(normalized, start) or _EXPLICIT.match(normalized, start)]
    soft_cues = [index for index, (_, start) in enumerate(words) if _SOFT_CUE.match(normalized, start)]
    # A restrictive request that uses a synonym the model maps to a metadata
    # value cannot safely be softened solely because its literal value is absent.
    if not positions:
        return bool(cues)
    negations = [index for index, (_, start) in enumerate(words) if _NEGATION.match(normalized, start)]
    return any(-2 <= position - cue <= 4
               and not any(cue < soft <= position for soft in soft_cues)
               and not any(cue - 1 <= negation <= position for negation in negations)
               for position in positions for cue in cues)


def _mentions_value(message: str, value: str) -> bool:
    return bool(_value_pattern(value).search(_text(message)))


def _explicitly_excludes_value(message: str, value: str) -> bool:
    normalized = _text(message)
    target = _value_pattern(value)
    before = any(
        0 <= len(re.findall(r"\b\w+\b", normalized[cue.end():match.start()])) <= 16
        and not re.search(r"[.;!?]|\b(?:but|however)\b", normalized[cue.end():match.start()])
        for cue in _NEGATION.finditer(normalized)
        for match in target.finditer(normalized)
        if cue.end() <= match.start()
    )
    after = any(
        0 <= len(re.findall(r"\b\w+\b", normalized[match.end():cue.start()])) <= 3
        and not any(mark in normalized[match.end():cue.start()] for mark in ",;.")
        and not re.search(r"\bbut\b", normalized[match.end():cue.start()])
        for match in target.finditer(normalized)
        for cue in _NEGATION.finditer(normalized)
        if match.end() <= cue.start()
    )
    return before or after


def _literal_tag_request(message: str, value: str) -> bool:
    normalized = _text(message)
    target = _value_pattern(value)
    return any(
        0 <= len(re.findall(r"\b\w+\b", normalized[cue.end():match.start()])) <= 3
        for cue in _EXPLICIT.finditer(normalized)
        for match in target.finditer(normalized)
        if cue.end() <= match.start()
    )


def _conditional_exclusion_satisfied(
    message: str, excluded_value: str, required_values: Collection[str],
) -> bool:
    """Recognize an excluded entity qualified as not matching a required entity."""
    normalized = _text(message)
    for excluded_match in _value_pattern(excluded_value).finditer(normalized):
        tail = re.split(r"[.;!?]", normalized[excluded_match.end():], maxsplit=1)[0]
        qualifier = re.search(r"\b(?:that|which)\s+(?:are\s+|is\s+)?not\s+", tail)
        if (qualifier is None
                or len(re.findall(r"\b\w+\b", tail[:qualifier.start()])) > 3
                or any(mark in tail[:qualifier.start()] for mark in ",;")):
            continue
        remainder = tail[qualifier.end():]
        if any(_value_pattern(required).match(remainder) for required in required_values):
            return True
    return False


def _resolve_metadata_predicates(
    intent: UserIntent, message: str,
    category_vocabulary: Collection[str], tag_vocabulary: Collection[str],
) -> UserIntent:
    """Map a requested entity type to the metadata field that actually has it."""
    if intent.hard_filters is None:
        return intent
    categories = {value.casefold() for value in category_vocabulary}
    tags = {value.casefold() for value in tag_vocabulary}
    data = intent.model_dump(mode="python")
    filters, actions = data["hard_filters"], data["field_actions"]
    category = filters.get("category")
    if actions["filter_category"] == "set" and category and category.casefold() in tags:
        if (category.casefold() not in categories or _literal_tag_request(message, category)) and _request_supports_filter(
            "filter_category", category.casefold(), "required", message, None
        ):
            values = list(filters.get("tags") or [])
            if category.casefold() not in {value.casefold() for value in values}:
                values.append(category)
            filters["tags"] = values
            filters["category"] = None
            actions["filter_tags"] = "set"
            actions["filter_category"] = "preserve"
    if actions["filter_excluded_categories"] == "set":
        kept = []
        moved = list(filters.get("excluded_tags") or [])
        for category in filters.get("excluded_categories") or []:
            if category.casefold() in tags and (
                category.casefold() not in categories or _literal_tag_request(message, category)
            ) and _explicitly_excludes_value(message, category):
                if category.casefold() not in {value.casefold() for value in moved}:
                    moved.append(category)
            else:
                kept.append(category)
        if len(kept) != len(filters.get("excluded_categories") or []):
            filters["excluded_categories"] = kept or None
            filters["excluded_tags"] = moved
            actions["filter_excluded_categories"] = "set" if kept else "preserve"
            actions["filter_excluded_tags"] = "set"
    required_values = [value for value in [filters.get("category"), *(filters.get("tags") or [])] if value]
    conditional_drops = 0
    for field, action_name in (("excluded_categories", "filter_excluded_categories"),
                               ("excluded_tags", "filter_excluded_tags")):
        if actions[action_name] != "set":
            continue
        original = list(filters.get(field) or [])
        retained = [value for value in original if not _conditional_exclusion_satisfied(
            message, value, required_values
        )]
        conditional_drops += len(original) - len(retained)
        filters[field] = retained or None
        actions[action_name] = "set" if retained else "preserve"
    if conditional_drops:
        _LOGGER.info(
            "Conditional exclusions resolved by mandatory inclusion",
            extra={"stage": "intent_normalization", "conditional_exclusion_drop_count": conditional_drops},
        )
    return UserIntent.model_validate(data)


def _request_supports_filter(
    field: str, value: str, kind: str, message: str,
    poi_names: Mapping[str, str] | None,
) -> bool:
    """Check explicit current-request wording independently of model annotations."""
    if kind == "required" and field in {"filter_category", "filter_tags"}:
        return (
            _mentions_value(message, value)
            and _explicitly_restricts_value(message, value)
        )
    if kind == "required" and field == "filter_setting":
        if not _mentions_value(message, value) or _explicitly_excludes_value(message, value):
            return False
        return not bool(re.search(
            r"\b(?:prefer|enjoy|interested in|would like)\s+(?:\w+\s+){0,2}"
            + re.escape(value) + r"\b", _text(message)
        ))
    if kind == "excluded":
        subject = (poi_names or {}).get(value.upper(), value) if field == "filter_excluded_poi_ids" else value
        return _explicitly_excludes_value(message, subject)
    return False


def validate_constraint_provenance(
    intent: UserIntent, message: str, poi_names: Mapping[str, str] | None = None
) -> UserIntent:
    """Admit supported constraints; discard unsupported positive filters only when safe."""
    evidence = intent.constraint_evidence or []
    expected = []
    for action_name, field in _FILTER_FIELDS.items():
        action = getattr(intent.field_actions, action_name)
        if action == "preserve":
            continue
        value = getattr(intent.hard_filters, field) if intent.hard_filters else None
        values = value if isinstance(value, list) else [value]
        if action == "clear" or values == []:
            expected.append((action_name, "*", "clear"))
        else:
            for item in values:
                expected.append((action_name, str(item).lower(), "excluded" if field.startswith("excluded_") else "required"))
    invalid = []
    if intent.hard_filters is not None:
        positive_tags = {value.casefold() for value in intent.hard_filters.tags or []}
        excluded_tags = {value.casefold() for value in intent.hard_filters.excluded_tags or []}
        for value in positive_tags & excluded_tags:
            invalid.append(("filter_tags", value, "conflict"))
        category = intent.hard_filters.category
        if category and category.casefold() in {
            value.casefold() for value in intent.hard_filters.excluded_categories or []
        }:
            invalid.append(("filter_category", category.casefold(), "conflict"))
    request_admitted = 0
    for field, value, kind in expected:
        # The current user request, not the model's evidence annotation, is
        # authoritative for positive categories/tags and exclusions. A broad
        # or mislabeled quote cannot authorize an invented restriction.
        if (field in {"filter_category", "filter_tags", "filter_setting"} and kind == "required") or kind == "excluded":
            if _request_supports_filter(field, value, kind, message, poi_names):
                request_admitted += 1
            else:
                invalid.append((field, value, kind))
            continue
        matches = [e for e in evidence if e.field == field and e.value.casefold() == value]
        valid = False
        if len(matches) == 1:
            e = matches[0]
            quote = _text(e.quote)
            valid = bool(quote and quote in _text(message) and e.kind != "ambiguous")
            valid &= e.kind == kind
            if kind == "clear":
                valid &= bool(re.search(r"\b(?:remove|clear|drop|lift|allow|any|no longer|forget)\b", quote))
        valid |= _request_supports_filter(field, value, kind, message, poi_names)
        if not valid:
            invalid.append((field, value, kind))
    if not invalid:
        _LOGGER.info(
            "Constraint provenance accepted",
            extra={"stage": "intent_normalization", "decision": "accepted",
                   "constraint_count": len(expected), "request_admitted_count": request_admitted},
        )
        return intent
    # A missing or malformed quote from the model is not itself ambiguity in
    # the user's request. A current, ordinary interest can still be served
    # without promoting the model's unsupported category/tag to a requirement.
    # Restrictive wording is kept conservative: a possibly explicit request
    # needs clarification rather than silent relaxation.
    softenable = all(
        field in {"filter_category", "filter_tags", "filter_setting"} and kind == "required"
        and not _explicitly_restricts_value(message, value)
        for field, value, kind in invalid
    )
    if softenable:
        data = intent.model_dump(mode="python")
        rejected = {(field, value) for field, value, _ in invalid}
        current_interests = list(
            intent.semantic_preferences.interests
            if intent.field_actions.semantic_interests == "set" and intent.semantic_preferences
            else []
        )
        if not current_interests:
            for field, value, _ in invalid:
                if (
                    field in {"filter_category", "filter_tags"}
                    and _mentions_value(message, value)
                    and not _explicitly_excludes_value(message, value)
                    and value.casefold() not in {item.casefold() for item in current_interests}
                ):
                    current_interests.append(value)
        for action_name, metadata_field in (("filter_category", "category"), ("filter_tags", "tags")):
            if action_name == "filter_category" and (action_name, str(data["hard_filters"].get(metadata_field)).lower()) in rejected:
                data["hard_filters"][metadata_field] = None
                data["field_actions"][action_name] = "preserve"
            elif action_name == "filter_tags" and data["field_actions"][action_name] == "set":
                kept = [tag for tag in data["hard_filters"][metadata_field] or []
                        if (action_name, tag.lower()) not in rejected]
                data["hard_filters"][metadata_field] = kept or None
                data["field_actions"][action_name] = "set" if kept else "preserve"
        setting = data["hard_filters"].get("setting")
        if ("filter_setting", str(setting).lower()) in rejected:
            data["hard_filters"]["setting"] = None
            data["field_actions"]["filter_setting"] = "preserve"
        if current_interests:
            data["semantic_preferences"] = {"interests": current_interests}
            data["field_actions"]["semantic_interests"] = "set"
        data["constraint_evidence"] = [e.model_dump() for e in evidence
                                       if (e.field, e.value.casefold()) not in rejected]
        _LOGGER.info(
            "Unsupported positive constraints retained as preferences only",
            extra={"stage": "intent_normalization", "decision": "softened",
                   "rejected_constraint_count": len(rejected)},
        )
        return UserIntent.model_validate(data)
    # A clarification carries no updates; inherited constraints and itinerary
    # remain authoritative until the user resolves the ambiguity.
    actions = dict.fromkeys(type(intent.field_actions).model_fields, "preserve")
    actions["clarification_question"] = "set"
    _LOGGER.info(
        "Constraint provenance requires clarification",
        extra={"stage": "intent_normalization", "decision": "clarification",
               "invalid_constraint_count": len(invalid),
               "invalid_constraint_fields": ",".join(sorted({field for field, _, _ in invalid}))},
    )
    return UserIntent.model_validate({
        "intent": "clarification", "field_actions": actions, "constraint_evidence": [],
        "clarification_question": "Should these interests be preferences, or must every place meet the proposed restrictions? Please specify which restrictions to require or exclude.",
    })

_EXPLICIT = re.compile(r"\b(?:tags?|tagged|metadata\s+tags?|labels?)\b", re.I)
_THEMES = {"history": ("history", "historical", "historic")}
_FOCUS = re.compile(r"\b(?:focus(?:ed|ing)? on|interested in|emphasis on)\s+(.+)", re.I)
_MANDATORY = re.compile(r"\b(?:only|exclusively|strictly|must|required|require)\b", re.I)
_FOCUS_TAGS = {
    "museum": ("museum", "museums"),
    "art": ("art", "arts", "gallery", "galleries"),
    "gallery": ("gallery", "galleries"),
}
_EXCLUSIVE = re.compile(r"\b(?:only|exclusively|strictly)\s+(?:(?:indoor|outdoor)\s+)?(?:historical|historic|history)\b", re.I)


def normalize_thematic_filters(
    intent: UserIntent, *, message: str,
    category_vocabulary: Collection[str], tag_vocabulary: Collection[str],
    prior_interests: Sequence[str] = (),
    poi_names: Mapping[str, str] | None = None,
) -> tuple[UserIntent, tuple[str, ...]]:
    """Move implicit category-like tags to soft interests; literal tags stay hard."""
    intent = _normalize_contextual_activity_request(intent, message)
    intent = _retain_explicit_soft_focus(
        intent,
        message,
        category_vocabulary=category_vocabulary,
        tag_vocabulary=tag_vocabulary,
    )
    if intent.constraint_evidence is not None:
        resolved = _resolve_metadata_predicates(intent, message, category_vocabulary, tag_vocabulary)
        return validate_constraint_provenance(resolved, message, poi_names), ()
    if intent.intent not in {"itinerary_creation", "itinerary_modification", "poi_recommendation"}:
        return intent, ()
    if intent.hard_filters is None or intent.field_actions.filter_tags != "set":
        return intent, ()
    if intent.field_actions.semantic_interests == "clear" or _EXPLICIT.search(message):
        return intent, ()
    shared = {x.casefold() for x in category_vocabulary} & {x.casefold() for x in tag_vocabulary}
    focus_match = _FOCUS.search(message)
    focus_phrase = focus_match.group(1).casefold() if focus_match else ""
    thematic_aliases = {
        canonical
        for canonical, forms in _FOCUS_TAGS.items()
        if any(re.search(r"\b" + re.escape(alias) + r"\b", focus_phrase) for alias in forms)
    }
    demoted = tuple(
        tag for tag in (intent.hard_filters.tags or [])
        if tag.casefold() in shared or (
            tag.casefold() in thematic_aliases and not _MANDATORY.search(message)
        )
    )
    if not demoted:
        return intent, ()
    data = intent.model_dump(mode="python")
    remaining = [tag for tag in intent.hard_filters.tags or [] if tag not in demoted]
    data["hard_filters"]["tags"] = remaining or None
    data["field_actions"]["filter_tags"] = "set" if remaining else "preserve"
    # An explicit exclusive theme is a category-level restriction, not a
    # literal tag requirement. Keep other explicit hard constraints unchanged.
    if _EXCLUSIVE.search(message) and len(demoted) == 1 and intent.field_actions.filter_category == "preserve":
        data["hard_filters"]["category"] = demoted[0]
        data["field_actions"]["filter_category"] = "set"
    current = list(intent.semantic_preferences.interests or []) if intent.field_actions.semantic_interests == "set" and intent.semantic_preferences else list(prior_interests)
    for tag in demoted:
        if tag.casefold() not in {item.casefold() for item in current}:
            current.append(tag)
    data["semantic_preferences"] = {"interests": current}
    data["field_actions"]["semantic_interests"] = "set"
    return UserIntent.model_validate(data), demoted


def _retain_explicit_soft_focus(
    intent: UserIntent,
    message: str,
    *,
    category_vocabulary: Collection[str],
    tag_vocabulary: Collection[str],
) -> UserIntent:
    """Resolve a clear current-turn soft focus against dataset vocabulary.

    A textual positive focus can repair an omitted or empty extractor field.
    The user's current request remains authoritative, while a clear action
    without a new focus is still handled by the normal patch contract.
    """
    if intent.intent not in {
        "itinerary_creation", "itinerary_modification", "poi_recommendation"
    }:
        return intent
    match = _FOCUS.search(message)
    if match is None:
        return intent
    focus = match.group(1).casefold()
    # Setting is a separate planning constraint, not a thematic interest.
    vocabulary = {
        value.casefold()
        for value in (*category_vocabulary, *tag_vocabulary)
        if value.casefold() not in {"indoor", "outdoor"}
    }
    aliases = {
        "history": ("history", "historical", "historic", "heritage"),
        "art": ("art", "arts", "artistic", "gallery", "galleries"),
        "nature": ("nature", "natural"),
        "food": ("food", "dining", "culinary", "restaurants"),
    }
    interests: list[str] = []
    for canonical, forms in aliases.items():
        if canonical not in vocabulary and not any(form in vocabulary for form in forms):
            continue
        if any(re.search(r"\b" + re.escape(form) + r"\b", focus) for form in forms):
            interests.append(canonical)
    for value in sorted(vocabulary):
        if value in interests:
            continue
        if _value_pattern(value).search(focus):
            interests.append(value)
    if not interests:
        return intent
    data = intent.model_dump(mode="python")
    # A positive, explicit focus in the current user message is stronger
    # evidence than an empty/clear action proposed by the extractor. Merge
    # any non-empty extracted soft interests with terms grounded in the
    # user's focus. A later explicit clear without a new focus remains clear.
    extracted = (
        list(intent.semantic_preferences.interests or [])
        if intent.field_actions.semantic_interests == "set"
        and intent.semantic_preferences is not None
        else []
    )
    merged = list(extracted)
    seen = {value.casefold() for value in merged}
    for value in interests:
        if value.casefold() not in seen:
            merged.append(value)
            seen.add(value.casefold())
    data["semantic_preferences"] = {"interests": merged}
    data["field_actions"]["semantic_interests"] = "set"
    _LOGGER.info(
        "Explicit soft focus recovered from dataset vocabulary",
        extra={
            "stage": "intent_normalization",
            "decision": "soft_focus_retained",
            "interest_count": len(interests),
        },
    )
    return UserIntent.model_validate(data)
