def time_in_minutes(time_str: str) -> int:
    try:
        hours, minutes = map(int, time_str.split(':'))
        if 0 <= hours < 24 and 0 <= minutes < 60:
            return hours * 60 + minutes
        else:
            raise ValueError("Invalid time format")
    except ValueError:
        raise ValueError("Invalid time format")

def time_in_string(minutes: int) -> str:
    try:
        if minutes < 0:
            raise ValueError("Minutes cannot be negative")
        if minutes >= 24 * 60:
            raise ValueError("Minutes cannot exceed 1439")
        hours = minutes // 60
        remaining_minutes = minutes % 60
        return f"{hours:02d}:{remaining_minutes:02d}"
    except ValueError:
        raise ValueError("Invalid minutes value")