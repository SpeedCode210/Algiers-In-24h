from __future__ import annotations

# All day names accepted by the API (lowercase).
VALID_DAYS: set[str] = {
    "sunday", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday",
}

# Each key is a frontend-facing algorithm ID.
# Value is the human-readable label shown in the UI.
ALGORITHM_LABELS: dict[str, str] = {
    # Greedy variants
    "greedy":              "Greedy (Score Priority)",
    "greedy_ratio":        "Greedy (Score/Time Ratio)",
    "greedy_nearest":      "Greedy (Nearest Neighbor)",
    "greedy_random":       "Greedy (Random)",
    # Simulated Annealing variants
    "sa":                  "Simulated Annealing",
    # GRASP
    "grasp":               "GRASP",
    # Tabu Search
    "tabu":                "Tabu Search",
    # Genetic Algorithm variants (one per fitness function)
    "ga":          "Genetic Algorithm",
    "ga_tailored":    "Tailored Genetic Algorithm",
    # Exact solver
    "cplex":               "CPLEX (Exact)",
}

VALID_ALGORITHMS: set[str] = set(ALGORITHM_LABELS.keys())

# Algorithms included in the /api/solve/all comparison.
# One representative per conceptual family.
COMPARISON_ALGORITHMS: list[str] = [
    "greedy",
    "sa",
    "grasp",
    "tabu",
    "ga",
]

VALID_CATEGORIES: set[str] = {
    "historical",
    "religious",
    "attraction",
    "tradition_art",
    "shopping",
}

MIN_TIME_BUDGET = 60     # 1 hour
MAX_TIME_BUDGET = 1440   # 24 hours
MIN_WEIGHT      = 0.4
MAX_WEIGHT      = 1.6


# ---------------------------------------------------------------------------
# Validation error
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """Raised when a request body fails validation.

    Attributes:
        message: Human-readable description of the failure.
        field: The field name that caused the error. Empty for global errors.
    """

    def __init__(self, message: str, field: str = "") -> None:
        """Initialises a ValidationError.

        Args:
            message: What is wrong with the value.
            field: Which request field is invalid.
        """
        self.message = message
        self.field   = field
        prefix = f"[{field}] " if field else ""
        super().__init__(f"{prefix}{message}")


# ---------------------------------------------------------------------------
# Low-level field validators
# ---------------------------------------------------------------------------


def _require(body: dict, field: str) -> object:
    """Returns a required field or raises ValidationError if missing.

    Args:
        body: Parsed JSON request body.
        field: Field name to look up.

    Returns:
        The field value.

    Raises:
        ValidationError: If the field is absent or None.
    """
    if field not in body or body[field] is None:
        raise ValidationError("This field is required.", field=field)
    return body[field]


def _validate_algorithm(value: object) -> str:
    """Validates and normalises the algorithm field.

    Args:
        value: Raw value from the request.

    Returns:
        Lowercase algorithm ID string.

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


def _validate_hotel_id(value: object) -> str:
    """Validates the hotel_id field.

    Args:
        value: Raw value from the request.

    Returns:
        Stripped hotel ID string.

    Raises:
        ValidationError: If value is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("Must be a non-empty string.", field="hotel_id")
    return value.strip()


def _validate_time_budget(value: object) -> int:
    """Validates the time_budget field.

    Args:
        value: Raw value. Must be a positive integer (minutes).

    Returns:
        Validated integer time budget.

    Raises:
        ValidationError: If value is out of range or wrong type.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError("Must be an integer (minutes).", field="time_budget")
    if not (MIN_TIME_BUDGET <= value <= MAX_TIME_BUDGET):
        raise ValidationError(
            f"Must be between {MIN_TIME_BUDGET} and {MAX_TIME_BUDGET} minutes.",
            field="time_budget",
        )
    return value


def _validate_tour_day(value: object) -> str:
    """Validates the tour_day field.

    Args:
        value: Raw value. Must be a lowercase day name string.

    Returns:
        Lowercase day name.

    Raises:
        ValidationError: If value is not a recognised day.
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
    """Validates start_time in HH:MM format and converts to minutes.

    Args:
        value: Raw value. Expected format: "HH:MM".

    Returns:
        Start time in minutes since midnight.

    Raises:
        ValidationError: If value is not a valid HH:MM string.
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
            "Hours must be 0-23 and minutes must be 0-59.", field="start_time"
        )
    return hours * 60 + minutes


def _validate_category_weights(value: object) -> dict[str, float]:
    """Validates the category_weights dict.

    Each key must be a known category and each value must be a float
    in [MIN_WEIGHT, MAX_WEIGHT].

    Args:
        value: Raw value. Must be a JSON object.

    Returns:
        Validated dict of {category: weight}.

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
        w = float(weight)
        if not (MIN_WEIGHT <= w <= MAX_WEIGHT):
            raise ValidationError(
                f"Weight for '{key}' must be between {MIN_WEIGHT} and {MAX_WEIGHT}.",
                field="category_weights",
            )
        validated[key] = w
    return validated


def _validate_algorithm_params(value: object) -> dict:
    """Validates the algorithm_params dict.

    Only checks that it is a dict — individual parameter values are
    validated by the solver constructors themselves.

    Args:
        value: Raw value. Must be a JSON object.

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

    Required fields:
        algorithm (str): One of VALID_ALGORITHMS.
        hotel_id (str): ID of the selected hotel.
        time_budget (int): Tour duration cap in minutes.
        tour_day (str): Lowercase day name.

    Optional fields:
        start_time (str): "HH:MM" format. Default: "09:00" (540 minutes).
        category_weights (dict): Category name → float weight multiplier.
        algorithm_params (dict): Algorithm-specific hyperparameters.

    Args:
        body: Parsed JSON from flask request.get_json(). Must be a dict.

    Returns:
        Validated parameter dict with keys:
            algorithm, hotel_id, time_budget, tour_day,
            start_time (int minutes), category_weights (dict),
            algorithm_params (dict).

    Raises:
        ValidationError: If any required field is missing or invalid.
    """
    if not isinstance(body, dict):
        raise ValidationError("Request body must be a JSON object.")

    algorithm   = _validate_algorithm(_require(body, "algorithm"))
    hotel_id    = _validate_hotel_id(_require(body, "hotel_id"))
    time_budget = _validate_time_budget(_require(body, "time_budget"))
    tour_day    = _validate_tour_day(_require(body, "tour_day"))

    start_time       = _validate_start_time(body["start_time"]) if "start_time" in body else 540
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

    Runs all comparison algorithms on the same configuration.
    Same rules as validate_solve_request but without the algorithm field.

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

    hotel_id    = _validate_hotel_id(_require(body, "hotel_id"))
    time_budget = _validate_time_budget(_require(body, "time_budget"))
    tour_day    = _validate_tour_day(_require(body, "tour_day"))

    start_time       = _validate_start_time(body["start_time"]) if "start_time" in body else 540
    category_weights = _validate_category_weights(body["category_weights"]) if "category_weights" in body else {}

    return {
        "hotel_id":         hotel_id,
        "time_budget":      time_budget,
        "tour_day":         tour_day,
        "start_time":       start_time,
        "category_weights": category_weights,
    }