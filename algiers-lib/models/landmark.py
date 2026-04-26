from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

class Day(IntEnum):
    SUNDAY = 0
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6

    @classmethod
    def from_string(cls, day_str: str) -> "Day":
        try:
            return cls[day_str.strip().upper()]
        except KeyError:
            valid = [d.name.lower() for d in cls]
            raise ValueError(
                f"'{day_str}' is not a valid day name. "
                f"Expected one of: {valid}.")


@dataclass(frozen=True)
class TimeSlot:

    open_time: int
    close_time: int

    def __post_init__(self) -> None:
        if self.open_time >= self.close_time:
            raise ValueError(
                f"open_time ({self.open_time}) must be strictly less than "
                f"close_time ({self.close_time}).")

    def contains(self, arrival: int, duration: int) -> bool:
        return self.open_time <= arrival and (arrival + duration) <= self.close_time


@dataclass
class WeeklySchedule:

    schedule: dict[Day, list[TimeSlot]] = field(default_factory=dict)

    def is_open_on(self, day: Day) -> bool:
        return bool(self.schedule.get(day))

    def get_slots(self, day: Day) -> list[TimeSlot]:
        return self.schedule.get(day, [])

    def earliest_valid_start(self, day: Day, arrival: float, duration: float) -> Optional[float]:# this has been changed
        for slot in self.get_slots(day):
            start = max(arrival, slot.open_time)
            if slot.contains(start, duration):
                return start
        return None  # visit ca't achieved
    

@dataclass(frozen=True)
class Landmark:
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
        return (self.latitude, self.longitude)
    
    def __str__(self) -> str: #helps for printing
        return (
            f"{self.name} "
            f"[{self.category}] "
            f"score={self.interest_score:.1f} "
            f"visit={self.visit_duration} min"
        )
    

import pandas as pd
from utils.time import time_in_minutes


def loadLandmarks(filepath: str = "../data/data.csv") -> list[Landmark]:

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


def loadHotel(hotel_path: str = "../data/hotel.csv") -> Landmark:
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