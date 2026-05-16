"""
Utilities for converting between time strings and minutes since midnight.
"""

def time_in_minutes(time_str: str) -> int:
    """
    Convert a time string in HH:MM format to total minutes since midnight.

    Args:
        time_str (str): Time in "HH:MM" format.

    Returns:
        int: Total minutes since midnight.

    Raises:
        ValueError: If the time format is invalid or values are out of range.
    """
    try:
        hours, minutes = map(int, time_str.split(':'))
        if 0 <= hours < 24 and 0 <= minutes < 60:
            return hours * 60 + minutes
        else:
            raise ValueError("Invalid time format")
    except ValueError:
        raise ValueError("Invalid time format")

def time_in_string(minutes: int) -> str:
    """
    Convert total minutes since midnight to a time string in HH:MM format.

    Args:
        minutes (int): Total minutes since midnight (0-1439).

    Returns:
        str: Time in "HH:MM" format.

    Raises:
        ValueError: If minutes is negative or exceeds 1439.
    """
    try:
        print(minutes)
        if minutes < 0:
            raise ValueError("Minutes cannot be negative")
        if minutes >= 24 * 60:
            raise ValueError("Minutes cannot exceed 1439")
        hours = int(minutes // 60)
        remaining_minutes = int(minutes % 60)
        return f"{hours:02d}:{remaining_minutes:02d}"
    except ValueError:
        raise ValueError("Invalid minutes value")