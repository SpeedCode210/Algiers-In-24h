"""
BenchmarkProblem: Problem subclass for Solomon OPTW benchmark instances.

Key differences from the standard Problem:
- Travel time uses Euclidean distance (x,y grid coords) instead of Haversine.
- Provides parse_solomon_file() to load .txt benchmark instances directly.
- All landmarks are scheduled open on Day.MONDAY for simplicity.
"""

from __future__ import annotations

import math
from pathlib import Path

from models.landmark import Day, Landmark, TimeSlot, WeeklySchedule
from models.problem import Problem


# ---------------------------------------------------------------------------
# BenchmarkProblem
# ---------------------------------------------------------------------------

class BenchmarkProblem(Problem):
    """Problem subclass that uses Euclidean distance for travel times.

    Solomon benchmark coordinates are planar (x, y) with travel time equal
    to the Euclidean distance between nodes (implicit speed = 1 unit/time-unit).
    """

    def _precompute_travel_matrix(self) -> None:
        """Override: use Euclidean distance instead of Haversine."""
        all_locs: list[Landmark] = [self.hotel] + self.landmarks
        for a in all_locs:
            for b in all_locs:
                if a.id == b.id:
                    continue
                dx = a.latitude - b.latitude   # latitude stores x
                dy = a.longitude - b.longitude  # longitude stores y
                self._travel_matrix[(a.id, b.id)] = math.sqrt(dx * dx + dy * dy)

    # -----------------------------------------------------------------------
    # Static factory
    # -----------------------------------------------------------------------

    @staticmethod
    def parse_solomon_file(filepath: str | Path) -> "BenchmarkProblem":
        """Parse a Solomon OPTW .txt file and return a BenchmarkProblem.

        File format::

            <nVeh> <maxRoute> <nNodes> <nDays>
            <dayId> <timeBudget>
            <id> <x> <y> <score> <duration> <f1> <f2> <f3> <open> <close>
            ...
            # Node 0 (depot) has only one trailing value (time-horizon), no open/close.

        Args:
            filepath: Path to the .txt benchmark file.

        Returns:
            A BenchmarkProblem configured with the file's data.
        """
        filepath = Path(filepath)
        lines = [ln.strip() for ln in filepath.read_text().splitlines() if ln.strip()]

        # Extract true time budget from the depot line
        time_budget = 0
        for raw in lines[2:]:
            parts = raw.split()
            if parts and int(parts[0]) == 0:
                # Depot line: parts[8] is the close time (time horizon)
                time_budget = int(float(parts[8]))
                break
        
        if time_budget <= 0:
            # Fallback
            time_budget = int(lines[1].split()[1])

        # Build a single full-day slot for the hotel (open 0 .. time_budget)
        HORIZON = max(time_budget, 4000)
        hotel_slot = TimeSlot(open_time=0, close_time=HORIZON)
        hotel_schedule = WeeklySchedule(
            schedule={day: [hotel_slot] for day in Day}
        )

        hotel: Landmark | None = None
        landmarks: list[Landmark] = []

        tour_day = Day.MONDAY  # fixed reference day for all benchmark instances

        for raw in lines[2:]:
            parts = raw.split()
            if not parts:
                continue
            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            score = float(parts[3])
            duration = int(float(parts[4]))

            if node_id == 0:
                # Depot / hotel — no time window columns
                lm = Landmark(
                    id="depot",
                    name="Depot",
                    latitude=x,
                    longitude=y,
                    interest_score=0.0,
                    visit_duration=0,
                    schedule=hotel_schedule,
                    category="hotel",
                )
                hotel = lm
            else:
                # Regular landmark with [open, close] columns
                open_t = int(float(parts[8]))
                close_t = int(float(parts[9]))

                # Ensure valid slot (close > open)
                if close_t <= open_t:
                    close_t = open_t + max(duration, 1)

                slot = TimeSlot(open_time=open_t, close_time=close_t)
                schedule = WeeklySchedule(schedule={tour_day: [slot]})

                lm = Landmark(
                    id=str(node_id),
                    name=f"Node_{node_id}",
                    latitude=x,
                    longitude=y,
                    interest_score=score,
                    visit_duration=duration,
                    schedule=schedule,
                    category="benchmark",
                )
                landmarks.append(lm)

        if hotel is None:
            raise ValueError(f"No depot (node 0) found in {filepath}")

        return BenchmarkProblem(
            hotel=hotel,
            landmarks=landmarks,
            time_budget=time_budget,
            tour_day=tour_day,
            start_time=0,
        )
