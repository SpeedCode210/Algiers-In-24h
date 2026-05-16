"""
BenchmarkProblem: Problem subclass for Solomon OPTW benchmark instances.

Key differences from the standard Problem:
- Travel time uses Euclidean distance (x,y grid coords) instead of Haversine.
- Provides parse_solomon_file() to load .txt benchmark instances directly.
- All landmarks are scheduled open on Day.MONDAY for simplicity.

TOPTW file format (one line per node):
    i  x  y  d  S  f  a  list  O  C
    0  1  2  3  4  5  6   7    8  9

    i = vertex number
    x = x coordinate        -> stored in latitude
    y = y coordinate        -> stored in longitude
    d = service duration    -> visit_duration
    S = profit / score      -> interest_score
    f, a, list = not relevant (list length depends on a; for Solomon a=1 so
                 exactly one extra column, giving indices 5,6,7 for f,a,list)
    O = opening time        -> parts[8]
    C = closing time        -> parts[9]

Node 0 is the depot: no O/C columns, last column is the time horizon (Tmax).
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
        """Parse a Solomon TOPTW .txt file and return a BenchmarkProblem.

        Column mapping (0-indexed after splitting each data line):
            0  i        vertex number
            1  x        x coordinate
            2  y        y coordinate
            3  d        service duration  (visit_duration)
            4  S        profit / score    (interest_score)
            5  f        not used
            6  a        not used
            7  list     not used  (one extra column for Solomon where a=1)
            8  O        opening time of time window
            9  C        closing time of time window

        Node 0 (depot) has no O/C columns; its last column is the time
        horizon, used as the time budget (Tmax).

        Args:
            filepath: Path to the .txt benchmark file.

        Returns:
            A BenchmarkProblem configured with the file's data.
        """
        filepath = Path(filepath)
        lines = [ln.strip() for ln in filepath.read_text().splitlines() if ln.strip()]

        # Extract true time budget from the depot line (node 0)
        # Depot format: 0  x  y  d  S  f  a  list  <time_horizon>
        # The time horizon is the last (9th, index 8) column for node 0.
        time_budget = 0
        for raw in lines[2:]:
            parts = raw.split()
            if parts and int(parts[0]) == 0:
                # Depot: last column is close time / time horizon
                time_budget = int(float(parts[-1]))
                break

        if time_budget <= 0:
            # Fallback to header line if depot parse failed
            time_budget = int(lines[1].split()[1])

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

            node_id  = int(parts[0])
            x        = float(parts[1])
            y        = float(parts[2])
            duration = int(float(parts[3]))    # d = service duration
            score    = float(parts[4])         # S = profit / interest score

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
                # Regular landmark: columns 8 and 9 are open/close times
                open_t  = int(float(parts[8]))
                close_t = int(float(parts[9]))

                # Solomon TOPTW defines close_t as the latest *start* time.
                # Our TimeSlot.contains() checks (arrival + duration <= close_time),
                # so we extend close_time by duration at parse time to match that
                # convention without modifying the shared landmark structure.
                close_t = close_t + duration

                # Ensure valid slot (close must be strictly after open)
                if close_t <= open_t:
                    close_t = open_t + max(duration, 1)

                slot     = TimeSlot(open_time=open_t, close_time=close_t)
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