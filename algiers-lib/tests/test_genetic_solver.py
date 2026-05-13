import random
import sys
from pathlib import Path   

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import pytest
from models.landmark import Day, Landmark, TimeSlot, WeeklySchedule, loadLandmarks, loadHotel
from models.problem import Problem
from models.tour import Tour
from utils.time import time_in_string
from solvers.genetic_mutation import Mutation
from solvers.genetic_crossover import Crossover
from solvers.genetic_fitness import (FeasibilityFitnessFunction, InfeasibilityFitnessFunction, PenaltyFitnessFunction)
from solvers.genetic_selection import Selection 
from solvers.genetic_augmented_representation import AugmentedRepresentation
from solvers.genetic_solver import GeneticSolver, TailoredGeneticSolver


def _build_landmark(landmark_id, name, lat, lon, score, duration, schedule):
    return Landmark(
        id=landmark_id, name=name, latitude=lat, longitude=lon,
        interest_score=score, visit_duration=duration,
        schedule=schedule, category="Tourist Site"
    )


def _build_hotel(schedule):
    return Landmark(
        id="hotel", name="Hotel", latitude=36.7, longitude=3.1,
        interest_score=0, visit_duration=0,
        schedule=schedule, category="Hotel"
    )


@pytest.fixture
def sample_problem():
    schedule = WeeklySchedule()
    schedule.schedule[Day.MONDAY] = [TimeSlot(540, 960)]
    hotel = _build_hotel(schedule)
    landmarks = [
        _build_landmark("1", "Alpha", 36.50, 3.00, 8.0, 30, schedule),
        _build_landmark("2", "Beta",  36.51, 3.01, 6.0, 30, schedule),
        _build_landmark("3", "Gamma", 36.52, 3.02, 9.0, 45, schedule),
        _build_landmark("4", "Delta", 36.53, 3.03, 4.0, 20, schedule),
    ]
    return Problem(hotel=hotel, landmarks=landmarks,
                   time_budget=480, tour_day=Day.MONDAY, start_time=540)


@pytest.fixture
def infeasible_landmark_problem(sample_problem):
    tuesday_schedule = WeeklySchedule()
    tuesday_schedule.schedule[Day.TUESDAY] = [TimeSlot(540, 720)]
    closed_lm = _build_landmark("closed", "Closed", 36.50, 3.00, 5.0, 30, tuesday_schedule)

    full_schedule = WeeklySchedule()
    full_schedule.schedule[Day.MONDAY] = [TimeSlot(0, 1439)]
    hotel = _build_hotel(full_schedule)
    return Problem(hotel=hotel, landmarks=[closed_lm],
                   time_budget=480, tour_day=Day.MONDAY, start_time=540)


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


def format_route_details(tour: Tour) -> str:
    simulation = tour.simulation_cache()
    if not simulation.entries:
        return "No visited landmarks."
    lines = []
    for entry in simulation.entries:
        landmark = entry.landmark
        day_slots = landmark.schedule.get_slots(tour.problem.tour_day)
        slot_info = ", ".join(
            f"{time_in_string(slot.open_time)}-{time_in_string(slot.close_time)}"
            for slot in day_slots
        ) or "closed"
        lines.append(
           f"{landmark.name} | arrival: {safe_time(entry.arrival_time)} "
          f"| start: {safe_time(entry.visit_start_time)} "
          f"| depart: {safe_time(entry.departure_time)} "
         f"| slots: {slot_info}"
      )
    return "\n".join(lines)

def safe_time(minutes):
    try:
        return time_in_string(minutes)
    except ValueError:
        return f"INVALID({minutes})"

class TestSelection:

    def test_tournament_returns_two_tours_from_population(self, sample_problem):
        pop = [sample_problem.random_tour() for _ in range(6)]
        sel = Selection(PenaltyFitnessFunction())
        p1, p2 = sel.tournament_selection(pop, k=3)
        assert isinstance(p1, Tour) and isinstance(p2, Tour)
        assert p1 in pop and p2 in pop

    def test_tournament_k_larger_than_population_raises(self, sample_problem):
        pop = [sample_problem.random_tour() for _ in range(4)]
        sel = Selection(PenaltyFitnessFunction())
        with pytest.raises(ValueError):
            sel.tournament_selection(pop, k=10)

    def test_tournament_k_less_than_2_raises(self, sample_problem):
        pop = [sample_problem.random_tour() for _ in range(4)]
        sel = Selection(PenaltyFitnessFunction())
        with pytest.raises(ValueError):
            sel.tournament_selection(pop, k=1)

    def test_tournament_always_picks_best_when_k_equals_pop_size(self, sample_problem):
        ff  = PenaltyFitnessFunction()
        lms = sample_problem.landmarks
        t_best   = Tour(sample_problem, [lms[2]])
        t_second = Tour(sample_problem, [lms[0]])
        t_worst  = Tour(sample_problem, [lms[3]])
        pop = [t_worst, t_best, t_second]
        sel = Selection(ff)
        p1, p2 = sel.tournament_selection(pop, k=3)
        selected_scores = sorted([ff.fitness(p1), ff.fitness(p2)], reverse=True)
        assert selected_scores[0] == pytest.approx(ff.fitness(t_best), abs=1e-6)
        assert selected_scores[1] == pytest.approx(ff.fitness(t_second), abs=1e-6)

    def test_random_selection_returns_two_distinct_tours(self, sample_problem):
        pop = [sample_problem.random_tour() for _ in range(5)]
        sel = Selection(PenaltyFitnessFunction())
        p1, p2 = sel.random_selection(pop)
        assert p1 in pop and p2 in pop
        assert p1 is not p2

    def test_random_selection_too_small_raises(self, sample_problem):
        sel = Selection(PenaltyFitnessFunction())
        with pytest.raises(ValueError):
            sel.random_selection([sample_problem.random_tour()])

    def test_random_selection_all_tours_reachable(self, sample_problem):
        pop = [sample_problem.random_tour() for _ in range(4)]
        sel = Selection(PenaltyFitnessFunction())
        seen = set()
        for _ in range(200):
            p1, p2 = sel.random_selection(pop)
            seen.add(id(p1))
            seen.add(id(p2))
        assert len(seen) == len(pop)

    def test_probability_selection_p0_always_returns_top_two(self, sample_problem):
        ff  = PenaltyFitnessFunction()
        lms = sample_problem.landmarks
        t1  = Tour(sample_problem, [lms[2]])
        t2  = Tour(sample_problem, [lms[0]])
        t3  = Tour(sample_problem, [lms[3]])
        sel = Selection(ff)
        for _ in range(20):
            p1, p2 = sel.probability_selection([t3, t1, t2], p=0.0)
            scores = {ff.fitness(p1), ff.fitness(p2)}
            assert ff.fitness(t1) in scores
            assert ff.fitness(t2) in scores

    def test_probability_selection_p1_best_always_included(self, sample_problem):
        ff  = PenaltyFitnessFunction()
        lms = sample_problem.landmarks
        t1  = Tour(sample_problem, [lms[2]])
        t2  = Tour(sample_problem, [lms[0]])
        t3  = Tour(sample_problem, [lms[3]])
        sel = Selection(ff)
        for _ in range(20):
            p1, p2 = sel.probability_selection([t3, t1, t2], p=1.0)
            assert ff.fitness(t1) in {ff.fitness(p1), ff.fitness(p2)}

    def test_probability_selection_invalid_p_raises(self, sample_problem):
        pop = [sample_problem.random_tour() for _ in range(3)]
        sel = Selection(PenaltyFitnessFunction())
        with pytest.raises(ValueError):
            sel.probability_selection(pop, p=1.5)

    def test_fps_returns_two_distinct_tours_from_population(self, sample_problem):
        pop = [sample_problem.random_tour() for _ in range(5)]
        sel = Selection(PenaltyFitnessFunction())
        p1, p2 = sel.fitness_proportionate_selection(pop)
        assert p1 in pop and p2 in pop
        assert p1 is not p2

    def test_fps_too_small_raises(self, sample_problem):
        sel = Selection(PenaltyFitnessFunction())
        with pytest.raises(ValueError):
            sel.fitness_proportionate_selection([sample_problem.random_tour()])

    def test_fps_higher_fitness_selected_more_often(self, sample_problem):
        ff  = PenaltyFitnessFunction()
        lms = sample_problem.landmarks
        t_best  = Tour(sample_problem, [lms[2]])
        t_worst = Tour(sample_problem, [lms[3]])
        sel = Selection(ff)
        best_count = sum(
            1 for _ in range(200)
            if sel.fitness_proportionate_selection([t_worst, t_best])[0] is t_best
        )
        assert best_count > 100


class TestMutation:

    def test_insert_adds_landmark_to_empty_tour(self, sample_problem):
        tour = sample_problem.create_empty_tour()
        result = Mutation()._insert(tour)
        assert len(result.visited_landmarks) == 1

    def test_insert_no_duplicates(self, sample_problem):
        tour = sample_problem.create_empty_tour()
        tour.add_landmark(sample_problem.landmarks[0])
        Mutation()._insert(tour)
        ids = [lm.id for lm in tour.visited_landmarks]
        assert len(ids) == len(set(ids))

    def test_insert_falls_back_to_delete_when_all_visited(self, sample_problem):
        tour = sample_problem.create_empty_tour()
        for lm in sample_problem.landmarks:
            tour.add_landmark(lm)
        before = len(tour.visited_landmarks)
        result = Mutation()._insert(tour)
        assert len(result.visited_landmarks) <= before

    def test_delete_removes_one_landmark(self, sample_problem):
        tour = sample_problem.create_empty_tour()
        tour.add_landmark(sample_problem.landmarks[0])
        tour.add_landmark(sample_problem.landmarks[1])
        before = len(tour.visited_landmarks)
        result = Mutation()._delete(tour)
        assert len(result.visited_landmarks) == before - 1

    def test_delete_on_empty_tour_returns_unchanged(self, sample_problem):
        tour   = sample_problem.create_empty_tour()
        result = Mutation()._delete(tour)
        assert len(result.visited_landmarks) == 0

    def test_tailored_insert_result_is_tour(self, sample_problem):
        tour   = sample_problem.random_tour()
        result = Mutation()._tailored_insert(tour)
        assert isinstance(result, Tour)

    def test_tailored_insert_no_duplicates(self, sample_problem):
        tour = sample_problem.random_tour()
        Mutation()._tailored_insert(tour)
        ids = [lm.id for lm in tour.visited_landmarks]
        assert len(ids) == len(set(ids))

    def test_tailored_insert_empty_tour_falls_back(self, sample_problem):
        tour   = sample_problem.create_empty_tour()
        result = Mutation()._tailored_insert(tour)
        assert isinstance(result, Tour)

    def test_mutate_empty_tour_always_inserts(self, sample_problem):
        mut  = Mutation(insertion_probability=0.0)
        tour = sample_problem.create_empty_tour()
        result = mut.mutate(tour)
        assert isinstance(result, Tour)
        assert len(result.visited_landmarks) >= 1

    def test_mutate_does_not_modify_original(self, sample_problem):
        tour       = sample_problem.create_empty_tour()
        tour.add_landmark(sample_problem.landmarks[0])
        ids_before = list(lm.id for lm in tour.visited_landmarks)
        Mutation().mutate(tour)
        ids_after  = list(lm.id for lm in tour.visited_landmarks)
        assert ids_before == ids_after


class TestCrossover:

    def test_order_crossover_both_empty(self, sample_problem):
        cx = Crossover(method="order")
        p1 = sample_problem.create_empty_tour()
        p2 = sample_problem.create_empty_tour()
        c1, c2 = cx.order_crossover(p1, p2)
        assert len(c1.visited_landmarks) == 0
        assert len(c2.visited_landmarks) == 0

    def test_order_crossover_one_empty(self, sample_problem):
        cx    = Crossover(method="order")
        empty = sample_problem.create_empty_tour()
        full  = sample_problem.create_empty_tour()
        full.add_landmark(sample_problem.landmarks[0])
        full.add_landmark(sample_problem.landmarks[1])
        c1, c2 = cx.order_crossover(empty, full)
        assert isinstance(c1, Tour) and isinstance(c2, Tour)

    def test_order_crossover_no_duplicates_in_children(self, sample_problem):
        cx = Crossover(method="order")
        lms = sample_problem.landmarks
        p1 = Tour(sample_problem, [lms[0], lms[1], lms[2]])
        p2 = Tour(sample_problem, [lms[2], lms[3], lms[0]])
        c1, c2 = cx.order_crossover(p1, p2)
        for child in (c1, c2):
            ids = [lm.id for lm in child.visited_landmarks]
            assert len(ids) == len(set(ids)), f"Duplicates in child: {ids}"

    def test_order_crossover_different_problems_raises(self, sample_problem):
        other_schedule = WeeklySchedule()
        other_schedule.schedule[Day.MONDAY] = [TimeSlot(540, 960)]
        other_hotel = _build_hotel(other_schedule)
        other_lm    = _build_landmark("x", "X", 36.5, 3.0, 5.0, 30, other_schedule)
        other_prob  = Problem(hotel=other_hotel, landmarks=[other_lm],
                              time_budget=480, tour_day=Day.MONDAY, start_time=540)
        p1 = sample_problem.random_tour()
        p2 = other_prob.random_tour()
        with pytest.raises(ValueError):
            Crossover(method="order").order_crossover(p1, p2)

    def test_tailored_crossover_no_valid_cuts_returns_two_tours(self, sample_problem):
        cx  = Crossover(method="tailored")
        ar1 = AugmentedRepresentation(landmarks=[], problem=sample_problem)
        ar2 = AugmentedRepresentation(landmarks=[], problem=sample_problem)
        c1, c2 = cx.tailored_crossover(ar1, ar2)
        assert isinstance(c1, Tour)
        assert isinstance(c2, Tour)

    def test_tailored_crossover_no_problem_raises(self):
        cx  = Crossover(method="tailored")
        ar1 = AugmentedRepresentation(landmarks=[], problem=None)
        ar2 = AugmentedRepresentation(landmarks=[], problem=None)
        with pytest.raises(ValueError):
            cx.tailored_crossover(ar1, ar2)

    def test_tailored_crossover_child_no_duplicates(self, sample_problem):
        cx = Crossover(method="tailored")
        t1 = sample_problem.random_tour()
        t2 = sample_problem.random_tour()
        if not t1.visited_landmarks or not t2.visited_landmarks:
            pytest.skip("random_tour returned empty; skipping")
        ar1 = AugmentedRepresentation.from_tour(t1)
        ar2 = AugmentedRepresentation.from_tour(t2)
        c1, c2 = cx.tailored_crossover(ar1, ar2)
        for child in (c1, c2):
            ids = ([lm.id for lm in child.landmarks]
                   if isinstance(child, AugmentedRepresentation)
                   else [lm.id for lm in child.visited_landmarks])
            assert len(ids) == len(set(ids))


class TestFitnessFunctions:

    def test_penalty_feasible_tour_no_penalty(self, sample_problem):
        ff   = PenaltyFitnessFunction(invalid_penalty=2.0, overtime_penalty=1.0)
        tour = sample_problem.random_tour()
        if not tour.is_valid() or len(tour.visited_landmarks) == 0:
            pytest.skip("Could not build a non-empty valid tour")
        invalid_count, total_dur = ff._evaluate_tour(tour)
        expected = (
            tour.total_score()
            - 2.0 * invalid_count
            - 1.0 * max(total_dur - sample_problem.time_budget, 0)
        )
        assert ff.fitness(tour) == pytest.approx(expected, abs=1e-6)

    def test_penalty_infeasible_landmark_reduces_fitness(self, infeasible_landmark_problem):
        ff   = PenaltyFitnessFunction(invalid_penalty=2.0, overtime_penalty=1.0)
        lm   = infeasible_landmark_problem.landmarks[0]
        tour = Tour(infeasible_landmark_problem, [lm])
        invalid_count, _ = ff._evaluate_tour(tour)
        assert invalid_count >= 1
        assert ff.fitness(tour) < tour.total_score()

    def test_penalty_custom_values_applied_correctly(self, sample_problem):
        ff   = PenaltyFitnessFunction(invalid_penalty=5.0, overtime_penalty=3.0)
        tour = sample_problem.random_tour()
        invalid_count, total_dur = ff._evaluate_tour(tour)
        expected = (
            tour.total_score()
            - 5.0 * invalid_count
            - 3.0 * max(total_dur - sample_problem.time_budget, 0)
        )
        assert ff.fitness(tour) == pytest.approx(expected, abs=1e-6)

    def test_infeasibility_empty_tour_returns_negative_infinity(self, sample_problem):
        ff   = InfeasibilityFitnessFunction()
        tour = sample_problem.create_empty_tour()
        assert ff.fitness(tour) == float("-inf")

    def test_infeasibility_infeasible_tour_is_heavily_penalised(self, infeasible_landmark_problem):
        ff   = InfeasibilityFitnessFunction()
        lm   = infeasible_landmark_problem.landmarks[0]
        tour = Tour(infeasible_landmark_problem, [lm])
        invalid_count       = ff._evaluate_tour(tour)[0]
        total_possible      = sum(l.interest_score for l in infeasible_landmark_problem.landmarks)
        expected            = tour.total_score() - total_possible * invalid_count
        assert ff.fitness(tour) == pytest.approx(expected, abs=1e-6)

    def test_infeasibility_feasible_tour_delegates_to_penalty(self, sample_problem):
        tour = sample_problem.random_tour()
        if not tour.is_valid() or len(tour.visited_landmarks) == 0:
            pytest.skip("Could not build a valid non-empty tour")
        assert InfeasibilityFitnessFunction().fitness(tour) == pytest.approx(
            PenaltyFitnessFunction().fitness(tour), abs=1e-6
        )

    def test_feasibility_formula(self, sample_problem):
        ff   = FeasibilityFitnessFunction()
        tour = sample_problem.random_tour()
        _, total_dur = ff._evaluate_tour(tour)
        reward   = sum(lm.interest_score for lm in tour.visited_landmarks)
        expected = reward + (sample_problem.time_budget - total_dur) / sample_problem.time_budget
        assert ff.fitness(tour) == pytest.approx(expected, abs=1e-6)

    def test_feasibility_empty_tour_bonus_is_one(self, sample_problem):
        ff   = FeasibilityFitnessFunction()
        tour = sample_problem.create_empty_tour()
        assert ff.fitness(tour) == pytest.approx(1.0, abs=1e-6)


class TestAugmentedRepresentation:

    def test_empty_tour_gives_empty_timeline(self, sample_problem):
        ar = AugmentedRepresentation.from_tour(sample_problem.create_empty_tour())
        assert ar.timeline == []
        assert ar.landmarks == []

    def test_timeline_length_matches_visited_landmarks(self, sample_problem):
        tour = sample_problem.random_tour()
        if not tour.visited_landmarks:
            pytest.skip("random_tour returned empty")
        ar = AugmentedRepresentation.from_tour(tour)
        assert len(ar.timeline) == len(tour.visited_landmarks)

    def test_timeline_entry_is_five_tuple(self, sample_problem):
        tour = sample_problem.random_tour()
        if not tour.visited_landmarks:
            pytest.skip("random_tour returned empty")
        ar = AugmentedRepresentation.from_tour(tour)
        for entry in ar.timeline:
            assert len(entry) == 5

    def test_departure_equals_start_plus_duration(self, sample_problem):
        tour = sample_problem.random_tour()
        if not tour.visited_landmarks:
            pytest.skip("random_tour returned empty")
        ar = AugmentedRepresentation.from_tour(tour)
        for i, (arrival, wait, start, departure, max_shift) in enumerate(ar.timeline):
            expected = start + ar.landmarks[i].visit_duration
            assert departure == pytest.approx(expected, abs=1e-6)

    def test_wait_equals_start_minus_arrival(self, sample_problem):
        tour = sample_problem.random_tour()
        if not tour.visited_landmarks:
            pytest.skip("random_tour returned empty")
        ar = AugmentedRepresentation.from_tour(tour)
        for (arrival, wait, start, departure, max_shift) in ar.timeline:
            assert wait == pytest.approx(start - arrival, abs=1e-6)

    def test_max_shift_nonnegative_for_valid_tour(self, sample_problem):
        tour = sample_problem.random_tour()
        if not tour.visited_landmarks:
            pytest.skip("random_tour returned empty")
        ar = AugmentedRepresentation.from_tour(tour)
        for (arrival, wait, start, departure, max_shift) in ar.timeline:
            assert max_shift >= -1e-9

    def test_compute_closing_term_within_slot(self):
        slot   = TimeSlot(540, 660)
        result = AugmentedRepresentation._compute_closing_term([slot], 570.0, 30.0, 100.0)
        assert result == pytest.approx(60.0, abs=1e-6)

    def test_compute_closing_term_next_limit_is_binding(self):
        slot   = TimeSlot(540, 1200)
        result = AugmentedRepresentation._compute_closing_term([slot], 540.0, 30.0, 10.0)
        assert result == pytest.approx(10.0, abs=1e-6)

    def test_compute_closing_term_no_slots_returns_next_limit(self):
        result = AugmentedRepresentation._compute_closing_term([], 540.0, 30.0, 45.0)
        assert result == pytest.approx(45.0, abs=1e-6)

    def test_compute_closing_term_start_outside_slot(self):
        slot   = TimeSlot(600, 700)
        result = AugmentedRepresentation._compute_closing_term([slot], 540.0, 30.0, 50.0)
        assert result == pytest.approx(50.0, abs=1e-6)


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


if __name__ == "__main__":
    print("-------------------this is the start of the second test-------------------")
    test_genetic_solver_runs_with_feasibility_fitness()
    print("ALL genetic solver tests executed.")