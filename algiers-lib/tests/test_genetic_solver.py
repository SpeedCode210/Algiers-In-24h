import random
import sys
from pathlib import Path   

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.landmark import Day, Landmark, TimeSlot, WeeklySchedule, loadLandmarks ,loadHotel
from models.problem import Problem
from models.tour import Tour
from utils.time import time_in_string
from solvers.genetic_mutation import Mutation
from solvers.genetic_crossover import Crossover
from solvers.genetic_fitness import (FeasibilityFitnessFunction, InfeasibilityFitnessFunction, PenaltyFitnessFunction)
from solvers.genetic_selection import Selection 
from solvers.genetic_augmented_representation import AugmentedRepresentation
from solvers.genetic_solver import GeneticSolver , TailoredGeneticSolver


def load_dataset_problem() -> Problem:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    landmarks = loadLandmarks(str(data_dir / "data.csv"))
    hotel = loadHotel(str(data_dir / "hotel.csv"))

    return Problem(hotel=hotel, landmarks=landmarks, time_budget=500, tour_day=Day.MONDAY)


def format_route(tour: Tour) -> str:
    if not tour.visited_landmarks:
        return "Hotel -> Hotel"
    return "Hotel -> " + " -> ".join(
        landmark.name for landmark in tour.visited_landmarks
    ) + " -> Hotel"


def format_time_window_feedback(entry, tour_day: Day) -> str:
    day_slots = entry.landmark.schedule.get_slots(tour_day)
    if not day_slots:
        return "closed on tour day"

    matching_slot = next(
        (
            slot for slot in day_slots
            if slot.contains(entry.visit_start_time, entry.landmark.visit_duration)
        ),
        None,
    )

    if matching_slot is None:
        return "no valid slot for this visit"

    arrival = round(entry.arrival_time)
    if arrival < matching_slot.open_time:
        wait = int(entry.visit_start_time - entry.arrival_time)
        return (
            f"waited {wait} min until slot {time_in_string(matching_slot.open_time)}-"
            f"{time_in_string(matching_slot.close_time)}"
        )

    if matching_slot.contains(arrival, entry.landmark.visit_duration):
        return f"arrived inside slot {time_in_string(matching_slot.open_time)}-{time_in_string(matching_slot.close_time)}"

    return f"started in slot {time_in_string(matching_slot.open_time)}-{time_in_string(matching_slot.close_time)}"


def format_route_details(tour: Tour) -> str:
    simulation = tour.simulation_cache()
    if not simulation.entries:
        return "No visited landmarks."

    lines = []
    for entry in simulation.entries:
        landmark = entry.landmark
        day_slots = landmark.schedule.get_slots(tour.problem.tour_day)
        slot_info = ", ".join(
            f"{slot.open_time}-{slot.close_time}"
            for slot in day_slots
        ) or "closed"

        lines.append(
            f"{landmark.name} | arrival: {entry.arrival_time} "
            f"| start: {entry.visit_start_time} "
            f"| depart: {entry.departure_time} "
            f"| slots: {slot_info} "
            f"| window: {format_time_window_feedback(entry, tour.problem.tour_day)}"
        )

    return "\n".join(lines)


def test_genetic_solver_runs_with_penalty_fitness() -> None:
    problem = load_dataset_problem()
    solver = GeneticSolver(
        problem=problem,
        fitness_function=PenaltyFitnessFunction(),
        regenerations=100,
        population_size=20,
        mutation_rate=0.1,
        crossover_method="order",
    )

    best = solver.solve()

    assert isinstance(best, Tour)
    assert best.problem is problem
    assert len(best.visited_landmarks) > 0

    print(
        f"test_genetic_solver_runs_with_penalty_fitness: PASSED - best route length={len(best.visited_landmarks)} - total score={best.total_score():.1f} - total time={best.simulation_cache().total_duration:.1f} min\n"
        f"route: {format_route(best)}\n"
        f"details:\n{format_route_details(best)}"
    )


def test_genetic_solver_runs_with_infeasibility_fitness() -> None:
    problem = load_dataset_problem()
    solver = GeneticSolver(
        problem=problem,
        fitness_function=InfeasibilityFitnessFunction(),
        regenerations=1,
        population_size=2,
        mutation_rate=0.1,
        crossover_method="order",
    )

    best = solver.solve()

    assert isinstance(best, Tour)
    assert best.problem is problem
    assert len(best.visited_landmarks) > 0
    assert best.is_valid()

    print(
        f"test_genetic_solver_runs_with_infeasibility_fitness: PASSED - best route length={len(best.visited_landmarks)} - total score={best.total_score():.1f} - total time={best.simulation_cache().total_duration:.1f} min\n"
        f"route: {format_route(best)}\n"
        f"details:\n{format_route_details(best)}"
    )


def test_genetic_solver_runs_with_feasibility_fitness() -> None:
    problem = load_dataset_problem()
    solver = TailoredGeneticSolver(
        problem=problem,
        fitness_function=FeasibilityFitnessFunction(),
        regenerations=4,
        population_size=2,
        mutation_rate=0.5,
        crossover_method="tailored",
    )

    best = solver.solve()

    assert isinstance(best, Tour)
    assert best.problem is problem
    assert len(best.visited_landmarks) > 0
    #assert best.is_valid()

    """print(
        f"test_genetic_solver_runs_with_feasibility_fitness: PASSED - best route length={len(best.visited_landmarks)} - total score={best.total_score():.1f} - total time={best.simulation_cache().total_duration:.1f} min\n"
        f"route: {format_route(best)}\n"
        f"details:\n{format_route_details(best)}"
    )"""


if __name__ == "__main__":
    print ("-------------------this is the start of the second test-------------------")
    test_genetic_solver_runs_with_feasibility_fitness()
    print("ALL genetic solver tests executed.")

