from __future__ import annotations

from models.problem import Problem
from models.tour import Tour
from models.landmark import Landmark
from .solver import Solver


class CPLEXSolver(Solver):
    """
    Exact MILP solver for the Orienteering Problem using IBM CPLEX (via docplex).

    Formulation
    -----------
    Nodes  : 0 = hotel, 1..n = candidate landmarks open on tour_day
    x[i,j] : binary — arc (i→j) is used
    y[i]   : binary — landmark i is visited  (i ≥ 1)
    t[i]   : continuous — visit *start* time at node i (minutes since midnight)
    w[i,k] : binary — landmark i is served in its k-th time-window slot

    Objective  : maximise Σ score_i · y[i]

    Constraints
    -----------
    (C1) Depart hotel exactly once        : Σ_j x[0,j] = 1
    (C2) Return to hotel exactly once     : Σ_i x[i,0] = 1
    (C3) Flow conservation per landmark   : Σ_j x[i,j] = y[i]  and  Σ_j x[j,i] = y[i]
    (C4) MTZ time propagation             : t[j] ≥ t[i] + dur[i] + travel[i,j] - M(1-x[i,j])
    (C5) Time-budget on return arcs       : t[i] + dur[i] + travel[i,0] ≤ t_end  when x[i,0]=1
    (C6) Time-window slot selection       : Σ_k w[i,k] = y[i]
    (C7) Lower bound per slot             : t[i] ≥ open[k]  - M(1-w[i,k])
    (C8) Upper bound per slot             : t[i] + dur[i] ≤ close[k] + M(1-w[i,k])
    (C9) Hotel fixed start                : t[0] = start_time
    """

    def __init__(
        self,
        problem: Problem,
        time_limit: float = 120.0,
        mip_gap: float = 0.0,
        log_output: bool = False,
    ) -> None:
        """
        Initialise the CPLEX solver.

        Args:
            problem    (Problem): The problem instance to solve.
            time_limit (float):  Wall-clock time limit for the solver in seconds.
                                 Defaults to 120 s.
            mip_gap    (float):  Relative MIP optimality gap tolerance (0 = optimal).
                                 Defaults to 0.0.
            log_output (bool):   Whether to print CPLEX solver log. Defaults to False.
        """
        if time_limit <= 0:
            raise ValueError("time_limit must be positive.")
        if not (0.0 <= mip_gap < 1.0):
            raise ValueError("mip_gap must be in [0, 1).")

        super().__init__(problem)
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.log_output = log_output

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tour:
        """
        Build and solve the MILP model, then reconstruct the best Tour found.

        Returns:
            Tour: The best feasible tour found within the time limit,
                  or an empty tour if the model is infeasible.

        Raises:
            ImportError: If docplex / CPLEX is not installed.
        """
        try:
            from docplex.mp.model import Model
        except ImportError:
            raise ImportError(
                "docplex is required for CPLEXSolver.\n"
                "Install with:  pip install docplex\n"
                "A licensed CPLEX installation must also be available."
            )

        # ---- Nodes -------------------------------------------------------
        # Only keep landmarks that are open on the tour day.
        candidates: list[Landmark] = [
            lm for lm in self.problem.landmarks
            if lm.schedule.is_open_on(self.problem.tour_day)
        ]

        nodes: list[Landmark] = [self.problem.hotel] + candidates  # index 0 = hotel
        N = len(nodes)
        n = N - 1  # number of candidate landmarks

        # Pre-fetch slots for each landmark node (index 1..N-1)
        day_slots: dict[int, list] = {
            i: nodes[i].schedule.get_slots(self.problem.tour_day)
            for i in range(1, N)
        }

        # Big-M — an upper bound on any time value in the problem
        t_end = self.problem.start_time + self.problem.time_budget
        BIG_M = t_end + 1440  # 1 extra day; never tight in practice

        # ---- Model -------------------------------------------------------
        mdl = Model(name="OrienteeringProblem")
        mdl.parameters.timelimit = self.time_limit
        mdl.parameters.mip.tolerances.mipgap = self.mip_gap

        # ---- Decision variables ------------------------------------------
        # x[i,j]  arc variables (no self-loops)
        x = {
            (i, j): mdl.binary_var(name=f"x_{i}_{j}")
            for i in range(N)
            for j in range(N)
            if i != j
        }

        # y[i]  visit indicators for landmarks only
        y = {i: mdl.binary_var(name=f"y_{i}") for i in range(1, N)}

        # t[i]  visit start time  (hotel = departure time, fixed)
        t = {
            i: mdl.continuous_var(lb=0, ub=t_end, name=f"t_{i}")
            for i in range(N)
        }

        # w[i,k]  time-window slot selector
        w = {
            (i, k): mdl.binary_var(name=f"w_{i}_{k}")
            for i in range(1, N)
            for k in range(len(day_slots[i]))
        }

        # ---- Objective ---------------------------------------------------
        mdl.maximize(
            mdl.sum(nodes[i].interest_score * y[i] for i in range(1, N))
        )

        # ---- Constraints -------------------------------------------------

        # (C9) Hotel departure time is fixed
        mdl.add_constraint(t[0] == self.problem.start_time, ctname="hotel_start")

        # (C1) Leave the hotel exactly once
        mdl.add_constraint(
            mdl.sum(x[(0, j)] for j in range(1, N)) == 1,
            ctname="depart_hotel"
        )

        # (C2) Return to the hotel exactly once
        mdl.add_constraint(
            mdl.sum(x[(i, 0)] for i in range(1, N)) == 1,
            ctname="return_hotel"
        )

        # (C3) Flow conservation at each landmark
        for i in range(1, N):
            mdl.add_constraint(
                mdl.sum(x[(i, j)] for j in range(N) if j != i) == y[i],
                ctname=f"flow_out_{i}"
            )
            mdl.add_constraint(
                mdl.sum(x[(j, i)] for j in range(N) if j != i) == y[i],
                ctname=f"flow_in_{i}"
            )

        # (C4) MTZ time-propagation (eliminates sub-tours and fixes timing)
        for i in range(N):
            for j in range(1, N):
                if i == j:
                    continue
                travel = self.problem.travel_time(nodes[i], nodes[j])
                dur_i = nodes[i].visit_duration  # 0 for hotel
                mdl.add_constraint(
                    t[j] >= t[i] + dur_i + travel - BIG_M * (1 - x[(i, j)]),
                    ctname=f"mtz_{i}_{j}"
                )

        # (C5) Return-arc timing must respect the overall time budget
        for i in range(1, N):
            travel_back = self.problem.travel_time(nodes[i], self.problem.hotel)
            mdl.add_constraint(
                t[i] + nodes[i].visit_duration + travel_back
                <= t_end + BIG_M * (1 - x[(i, 0)]),
                ctname=f"budget_{i}"
            )

        # (C6) Exactly one slot is selected iff the landmark is visited
        for i in range(1, N):
            slots_i = day_slots[i]
            mdl.add_constraint(
                mdl.sum(w[(i, k)] for k in range(len(slots_i))) == y[i],
                ctname=f"slot_select_{i}"
            )

        # (C7) & (C8) Time-window bounds per slot
        for i in range(1, N):
            dur_i = nodes[i].visit_duration
            for k, slot in enumerate(day_slots[i]):
                # Lower bound: t[i] >= open_k  when slot k is active
                mdl.add_constraint(
                    t[i] >= slot.open_time - BIG_M * (1 - w[(i, k)]),
                    ctname=f"tw_lb_{i}_{k}"
                )
                # Upper bound: t[i] + dur <= close_k  when slot k is active
                mdl.add_constraint(
                    t[i] + dur_i <= slot.close_time + BIG_M * (1 - w[(i, k)]),
                    ctname=f"tw_ub_{i}_{k}"
                )

        # ---- Solve -------------------------------------------------------
        solution = mdl.solve(log_output=self.log_output)

        if solution is None:
            # No feasible solution found within the time limit
            return self.problem.create_empty_tour()

        # ---- Extract tour ------------------------------------------------
        visited_ordered = self._extract_route(solution, x, nodes, N)
        return Tour(self.problem, visited_ordered)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_route(
        self,
        solution,
        x: dict,
        nodes: list[Landmark],
        N: int,
    ) -> list[Landmark]:
        """
        Follow the active arcs from the hotel to reconstruct the ordered visit list.

        Args:
            solution : CPLEX solution object.
            x        : Arc decision variables dict.
            nodes    : Full node list (index 0 = hotel).
            N        : Total number of nodes.

        Returns:
            list[Landmark]: Ordered list of visited landmarks (hotel excluded).
        """
        # Build adjacency from solution values
        arc_used: dict[int, int] = {}  # successor[i] = j
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if solution.get_value(x[(i, j)]) > 0.5:
                    arc_used[i] = j

        # Walk from hotel (0) until we return to hotel
        route: list[Landmark] = []
        current = 0
        visited: set[int] = {0}

        for _ in range(N):  # at most N steps before returning to 0
            nxt = arc_used.get(current)
            if nxt is None or nxt == 0:
                break
            if nxt in visited:
                break  # safety guard against malformed solution
            route.append(nodes[nxt])
            visited.add(nxt)
            current = nxt

        return route