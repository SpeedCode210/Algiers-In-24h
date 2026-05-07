from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

class Day(IntEnum):
    """Enumeration for days of the week, starting from Sunday as 0."""

    SUNDAY = 0
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6

    @classmethod
    def from_string(cls, day_str: str) -> "Day":
        """Convert a string representation of a day to the Day enum.

        Args:
            day_str (str): The day name (case-insensitive).

        Returns:
            Day: The corresponding Day enum value.

        Raises:
            ValueError: If the day_str is not a valid day name.
        """
        try:
            return cls[day_str.strip().upper()]
        except KeyError:
            valid = [d.name.lower() for d in cls]
            raise ValueError(
                f"'{day_str}' is not a valid day name. "
                f"Expected one of: {valid}.")


@dataclass(frozen=True)
class TimeSlot:
    """Represents a time slot with opening and closing times in minutes since midnight.

    Attributes:
        open_time (int): Opening time in minutes.
        close_time (int): Closing time in minutes.
    """

    open_time: int
    close_time: int

    def __post_init__(self) -> None:
        if self.open_time >= self.close_time:
            raise ValueError(
                f"open_time ({self.open_time}) must be strictly less than "
                f"close_time ({self.close_time}).")

    def contains(self, arrival: float, duration: float) -> bool:
        """Check if a visit starting at arrival time with given duration fits within the slot.

        Args:
            arrival (float): Arrival time in minutes.
            duration (float): Visit duration in minutes.

        Returns:
            bool: True if the visit fits, False otherwise.
        """
        return self.open_time <= arrival and (arrival + duration) <= self.close_time


@dataclass
class WeeklySchedule:
    """Represents the weekly schedule of time slots for each day.

    Attributes:
        schedule (dict[Day, list[TimeSlot]]): Mapping of days to lists of time slots.
    """

    schedule: dict[Day, list[TimeSlot]] = field(default_factory=dict)

    def is_open_on(self, day: Day) -> bool:
        """Check if the schedule has any slots on the given day.

        Args:
            day (Day): The day to check.

        Returns:
            bool: True if there are slots on that day.
        """
        return bool(self.schedule.get(day))

    def get_slots(self, day: Day) -> list[TimeSlot]:
        """Get the list of time slots for the given day.

        Args:
            day (Day): The day to get slots for.

        Returns:
            list[TimeSlot]: List of slots, empty if none.
        """
        return self.schedule.get(day, [])

    def earliest_valid_start(self, day: Day, arrival: float, duration: float) -> Optional[float]:# this has been changed
        for slot in self.get_slots(day):
            start = max(arrival, slot.open_time)
            if slot.contains(start, duration):
                return start
        return None  # visit ca't achieved
    

@dataclass(frozen=True)
class Landmark:
    """Represents a landmark with location, interest, and schedule information.

    Attributes:
        id (str): Unique identifier.
        name (str): Name of the landmark.
        latitude (float): Latitude coordinate.
        longitude (float): Longitude coordinate.
        interest_score (float): Interest score.
        visit_duration (int): Estimated Visit duration in minutes.
        schedule (WeeklySchedule): Weekly schedule.
        category (str): Category of the landmark.
    """

    id: str
    name: str
    latitude: float
    longitude: float
    interest_score: float
    visit_duration: int
    schedule: WeeklySchedule
    category: str

    @property
    def coordinates(self) -> tuple[float, float]:
        """Get the coordinates as a tuple (latitude, longitude).

        Returns:
            tuple[float, float]: The coordinates.
        """
        return (self.latitude, self.longitude)
    
    def __str__(self) -> str: #helps for printing
        """Return a string representation of the landmark."""
        return (
            f"{self.name} "
            f"[{self.category}] "
            f"score={self.interest_score:.1f} "
            f"visit={self.visit_duration} min"
        )
    

import pandas as pd
from utils.time import time_in_minutes


def loadLandmarks(filepath: str = "../data/data.csv") -> list[Landmark]:
    """Load landmarks from a CSV file.

    Args:
        filepath (str): Path to the CSV file. Defaults to "../data/data.csv".

    Returns:
        list[Landmark]: List of loaded landmarks.
    """

    df = pd.read_csv(filepath)

    landmarks = []

    for landmark_id, group in df.groupby("id"):

        slots: dict[Day, list[TimeSlot]] = {}

        for _, row in group.iterrows():
            day = Day.from_string(row["day"])
            slot = TimeSlot(
                open_time=time_in_minutes(row["open_time"]),
                close_time=time_in_minutes(row["close_time"]),
            )
            slots.setdefault(day, []).append(slot)

        first = group.iloc[0]

        landmark = Landmark(
            id=str(first["id"]),
            name=str(first["name"]),
            latitude=float(first["latitude"]),
            longitude=float(first["longitude"]),
            interest_score=float(first["interest_score"]),
            visit_duration=int(first["visit_duration_minutes"]),
            schedule=WeeklySchedule(schedule=slots),
            category=str(first["category"]),
        )
        landmarks.append(landmark)

    return landmarks

def loadAllHotels(hotel_path: str = "../data/hotel.csv") -> list[Landmark]:
    """Loads all hotels from a CSV file.

    Each hotel gets a 24/7 open schedule, zero interest score, and
    zero visit duration — matching the format used throughout the codebase.
    This function does NOT replace loadHotel() which remains unchanged.

    Args:
        hotel_path: Path to the hotel CSV file.

    Returns:
        List of Landmark instances, one per hotel row.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(hotel_path)
    required = {"id", "name", "latitude", "longitude"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"hotel.csv is missing columns: {missing}")

    full_slot     = TimeSlot(open_time=0, close_time=1439)
    full_schedule = WeeklySchedule(schedule={day: [full_slot] for day in Day})

    hotels = []
    for _, row in df.iterrows():
        hotels.append(Landmark(
            id             = str(row["id"]).strip(),
            name           = str(row["name"]).strip(),
            latitude       = float(row["latitude"]),
            longitude      = float(row["longitude"]),
            interest_score = 0.0,
            visit_duration = 0,
            schedule       = full_schedule,
            category       = "hotel",
        ))
    return hotels

def loadHotel(hotel_path: str = "../data/hotel.csv") -> Landmark:
    """Load the hotel landmark from a CSV file.

    Args:
        hotel_path (str): Path to the hotel CSV file. Defaults to "../data/hotel.csv".

    Returns:
        Landmark: The hotel landmark.
    """
    df = pd.read_csv(hotel_path)
    row = df.iloc[0]

    full_day_slot = TimeSlot(open_time=0, close_time=1439)
    slots = {day: [full_day_slot] for day in Day}
    schedule = WeeklySchedule(schedule=slots)

    return Landmark(
        id=str(row["id"]).strip(),
        name=str(row["name"]).strip(),
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        interest_score=0.0,
        visit_duration=0,
        schedule=schedule,
        category="hotel",
    )