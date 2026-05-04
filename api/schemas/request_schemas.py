from __future__ import annotations

from models.landmark import Day

# ---------------------------------------------------------------------------
# Validation error
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when an incoming request body fails validation.

    Attributes:
        message: Human-readable description of what is wrong.
        field: The request field that caused the error, if applicable.
    """

    def __init__(self, message: str, field: str = "") -> None:
        """Initialises a ValidationError.

        Args:
            message: Description of the validation failure.
            field: The name of the field that failed (empty if global).
        """
        self.message = message
        self.field = field
        super().__init__(f"{field}: {message}" if field else message)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ALGORITHMS = {"grasp", "greedy", "sa", "genetic", "tabu", "cplex"}

VALID_DAYS = {d.name.lower() for d in Day}

MIN_TIME_BUDGET = 60    # 1 hour minimum
MAX_TIME_BUDGET = 1440  # 24 hours maximum

MIN_WEIGHT = 0.0
MAX_WEIGHT = 5.0

VALID_CATEGORIES = {
    "historical",
    "religious",
    "attraction",
    "tradition_art",
    "shopping",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require(body: dict, field: str) -> object:
    """Asserts that a required field is present and not None.

    Args:
        body: The parsed JSON request body.
        field: The field name to check.

    Returns:
        The field value.

    Raises:
        ValidationError: If the field is absent or None.
    """
    if field not in body or body[field] is None:
        raise ValidationError(f"This field is required.", field=field)
    return body[field]


def _validate_time_budget(value: object) -> int:
    """Validates and coerces the time_budget field.

    Args:
        value: Raw value from the request body.

    Returns:
        Validated integer time budget in minutes.

    Raises:
        ValidationError: If value is not a valid integer in range.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(
            "Must be an integer (minutes).", field="time_budget"
        )
    if not (MIN_TIME_BUDGET <= value <= MAX_TIME_BUDGET):
        raise ValidationError(
            f"Must be between {MIN_TIME_BUDGET} and {MAX_TIME_BUDGET} minutes.",
            field="time_budget",
        )
    return value


def _validate_tour_day(value: object) -> str:
    """Validates the tour_day field.

    Args:
        value: Raw value from the request body.

    Returns:
        Lowercased day string.

    Raises:
        ValidationError: If value is not a recognised day name.
    """
    if not isinstance(value, str):
        raise ValidationError("Must be a string day name.", field="tour_day")
    normalised = value.strip().lower()
    if normalised not in VALID_DAYS:
        raise ValidationError(
            f"Must be one of: {sorted(VALID_DAYS)}.", field="tour_day"
        )
    return normalised


def _validate_start_time(value: object) -> int:
    """Validates and converts a start_time string to minutes since midnight.

    Args:
        value: Raw value from the request body. Expected format: "HH:MM".

    Returns:
        Start time in minutes since midnight.

    Raises:
        ValidationError: If value is not a valid "HH:MM" string.
    """
    if not isinstance(value, str):
        raise ValidationError(
            'Must be a string in "HH:MM" format.', field="start_time"
        )
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValidationError(
            'Must be in "HH:MM" format (e.g. "09:00").', field="start_time"
        )
    try:
        hours, minutes = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValidationError(
            "Hours and minutes must be integers.", field="start_time"
        )
    if not (0 <= hours <= 23 and 0 <= minutes <= 59):
        raise ValidationError(
            "Hours must be 0–23 and minutes must be 0–59.", field="start_time"
        )
    return hours * 60 + minutes


def _validate_hotel_id(value: object) -> str:
    """Validates the hotel_id field.

    Args:
        value: Raw value from the request body.

    Returns:
        Stripped hotel ID string.

    Raises:
        ValidationError: If value is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            "Must be a non-empty string.", field="hotel_id"
        )
    return value.strip()


def _validate_category_weights(value: object) -> dict[str, float]:
    """Validates and normalises the category_weights field.

    Each key must be a recognised category and each value must be a
    float in [MIN_WEIGHT, MAX_WEIGHT].

    Args:
        value: Raw value from the request body. Must be a dict.

    Returns:
        Validated dict mapping category names to float weights.

    Raises:
        ValidationError: If structure, keys, or values are invalid.
    """
    if not isinstance(value, dict):
        raise ValidationError(
            "Must be a JSON object mapping category names to weights.",
            field="category_weights",
        )
    validated: dict[str, float] = {}
    for key, weight in value.items():
        if key not in VALID_CATEGORIES:
            raise ValidationError(
                f"Unknown category '{key}'. "
                f"Valid categories: {sorted(VALID_CATEGORIES)}.",
                field="category_weights",
            )
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ValidationError(
                f"Weight for '{key}' must be a number.",
                field="category_weights",
            )
        weight_f = float(weight)
        if not (MIN_WEIGHT <= weight_f <= MAX_WEIGHT):
            raise ValidationError(
                f"Weight for '{key}' must be between "
                f"{MIN_WEIGHT} and {MAX_WEIGHT}.",
                field="category_weights",
            )
        validated[key] = weight_f
    return validated


def _validate_algorithm(value: object) -> str:
    """Validates the algorithm field.

    Args:
        value: Raw value from the request body.

    Returns:
        Lowercased algorithm key string.

    Raises:
        ValidationError: If value is not a recognised algorithm.
    """
    if not isinstance(value, str):
        raise ValidationError("Must be a string.", field="algorithm")
    normalised = value.strip().lower()
    if normalised not in VALID_ALGORITHMS:
        raise ValidationError(
            f"Must be one of: {sorted(VALID_ALGORITHMS)}.",
            field="algorithm",
        )
    return normalised


def _validate_algorithm_params(value: object) -> dict:
    """Validates the algorithm_params field.

    Only checks structure — individual parameter values are left to
    the solver constructors, which raise ValueError on bad values.

    Args:
        value: Raw value from the request body.

    Returns:
        The params dict (may be empty).

    Raises:
        ValidationError: If value is not a dict.
    """
    if not isinstance(value, dict):
        raise ValidationError(
            "Must be a JSON object of algorithm hyperparameters.",
            field="algorithm_params",
        )
    return value


# ---------------------------------------------------------------------------
# Public validators
# ---------------------------------------------------------------------------


def validate_solve_request(body: object) -> dict:
    """Validates a POST /api/solve request body.

    Required fields: algorithm, hotel_id, time_budget, tour_day.
    Optional fields: start_time, category_weights, algorithm_params.

    Args:
        body: Parsed JSON from flask request.get_json(). Must be a dict.

    Returns:
        Validated and normalised parameter dict with keys:
            - algorithm (str)
            - hotel_id (str)
            - time_budget (int, minutes)
            - tour_day (str, lowercase day name)
            - start_time (int, minutes since midnight, default 540)
            - category_weights (dict[str, float], default {})
            - algorithm_params (dict, default {})

    Raises:
        ValidationError: If any required field is missing or invalid.
    """
    if not isinstance(body, dict):
        raise ValidationError("Request body must be a JSON object.")

    algorithm        = _validate_algorithm(_require(body, "algorithm"))
    hotel_id         = _validate_hotel_id(_require(body, "hotel_id"))
    time_budget      = _validate_time_budget(_require(body, "time_budget"))
    tour_day         = _validate_tour_day(_require(body, "tour_day"))

    start_time = _validate_start_time(body["start_time"]) if "start_time" in body else 540
    category_weights = _validate_category_weights(body["category_weights"]) if "category_weights" in body else {}
    algorithm_params = _validate_algorithm_params(body["algorithm_params"]) if "algorithm_params" in body else {}

    return {
        "algorithm":        algorithm,
        "hotel_id":         hotel_id,
        "time_budget":      time_budget,
        "tour_day":         tour_day,
        "start_time":       start_time,
        "category_weights": category_weights,
        "algorithm_params": algorithm_params,
    }


def validate_solve_all_request(body: object) -> dict:
    """Validates a POST /api/solve/all request body.

    Same as validate_solve_request but without the algorithm field,
    since all algorithms are run and compared automatically.

    Required fields: hotel_id, time_budget, tour_day.
    Optional fields: start_time, category_weights.

    Args:
        body: Parsed JSON from flask request.get_json().

    Returns:
        Validated parameter dict without the algorithm key.

    Raises:
        ValidationError: If any required field is missing or invalid.
    """
    if not isinstance(body, dict):
        raise ValidationError("Request body must be a JSON object.")

    hotel_id         = _validate_hotel_id(_require(body, "hotel_id"))
    time_budget      = _validate_time_budget(_require(body, "time_budget"))
    tour_day         = _validate_tour_day(_require(body, "tour_day"))

    start_time       = _validate_start_time(body["start_time"]) if "start_time" in body else 540
    category_weights = _validate_category_weights(body["category_weights"]) if "category_weights" in body else {}

    return {
        "hotel_id":         hotel_id,
        "time_budget":      time_budget,
        "tour_day":         tour_day,
        "start_time":       start_time,
        "category_weights": category_weights,
    }