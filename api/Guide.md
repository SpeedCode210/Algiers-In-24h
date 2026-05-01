
## The Folder Structure First

```
api/
├── main.py
├── routes/
│   ├── __init__.py
│   ├── landmarks.py
│   └── solve.py
├── services/
│   ├── __init__.py
│   ├── problem_loader.py
│   ├── solver_service.py
│   ├── routing_service.py
│   └── scoring_service.py
└── schemas/
    ├── __init__.py
    └── request_schemas.py
```

---

## `api/main.py`

**What it is:** The entry point of the entire backend. The single file you run to start the server.

**What it does:**

It creates the Flask application object, enables CORS (Cross-Origin Resource Sharing) so the Next.js frontend running on a different port can call it without the browser blocking it, registers the route blueprints (which tells Flask "these URLs exist"), and starts the server on port 5000.

**Why it's minimal:** It contains zero logic. Its only job is assembly — connecting all the pieces together and saying "go." If someone asks "where does the backend start?", the answer is always this file.

```python
# When you run: python api/main.py
# Flask starts listening on http://localhost:5000
# Every request to /api/... gets routed to the right handler
```

---

## `routes/__init__.py`

**What it is:** An empty file.

**What it does:** Nothing by itself. Its presence tells Python "treat the `routes/` folder as a package so other files can import from it." Without this file, `from routes.landmarks import landmarks_bp` would fail with an import error.

Same applies to `services/__init__.py` and `schemas/__init__.py` — all three are empty files that exist purely for Python's import system.

---

## `routes/landmarks.py`

**What it is:** A Flask Blueprint that handles two read-only endpoints.

**What it does:** It answers two questions the frontend asks before anything else:

**Question 1 — `GET /api/landmarks`:** "Give me every landmark and the hotel so I can display them on the map." This is called once when the page loads. The frontend uses the response to place pins on the map for every landmark — before the user has run any algorithm.

**Question 2 — `GET /api/categories`:** "Give me the list of all categories so I can display the category filter panel." The frontend uses this to build the panel where the user assigns weights to categories like Historical, Nature, Religious.

**What it does NOT do:** No algorithm runs here. No routing happens here. No heavy computation. It just reads the loaded problem and formats the data into JSON. This endpoint should respond in under 5ms.

**Why a Blueprint?** Flask Blueprints let you split routes across multiple files instead of putting everything in `main.py`. `landmarks_bp` is registered in `main.py` — that's how Flask knows these routes exist.

---

## `routes/solve.py`

**What it is:** A Flask Blueprint that handles the two most important endpoints — running algorithms.

**What it does:**

**`POST /api/solve`:** The user has selected an algorithm (say GRASP), set a time budget (720 minutes), chosen categories, and clicked Run. The frontend sends all of this as JSON. This route:
1. Reads and validates the incoming JSON
2. Passes everything to `solver_service.run_solver()`
3. Returns the full result — the tour stops, road geometry, scores, execution time — as JSON

**`POST /api/solve/all`:** The user clicks "Compare All Algorithms." Same configuration but no specific algorithm. This route:
1. Reads and validates the incoming JSON
2. Passes to `solver_service.run_all_solvers()`
3. Returns the ranked comparison of every algorithm

**What it does NOT do:** No algorithm logic. No map calls. No score calculations. It purely handles HTTP — reading the request and writing the response. All real work happens in `services/`.

**Why keep routes dumb?** Because if you put logic in routes, when you need to change how algorithms run, you have to hunt through HTTP-handling code to find it. Keeping routes as thin wrappers makes the codebase navigable.

---

## `services/problem_loader.py`

**What it is:** The file responsible for loading your CSV data into a `Problem` object.

**What it does:**

Loading a `Problem` is expensive — it reads two CSV files and builds the full travel time matrix (computing Haversine distance for every pair of landmarks). With 25 landmarks, that's 625 distance calculations every time.

If you loaded the Problem on every single API request, a user clicking "Compare All Algorithms" would load the CSV 4 times in a row. Wasteful and slow.

This file solves that with a **module-level cache** — a variable `_BASE_PROBLEM` that starts as `None`. The first time any code calls `get_base_problem()`, it loads from CSV and stores the result in `_BASE_PROBLEM`. Every subsequent call just returns the cached object. This is called a singleton pattern.

It also provides `build_problem()` — a function that takes the user's chosen time budget, tour day, and start time, and creates a new `Problem` using the cached landmarks but with the user's specific parameters. This is necessary because the cached base problem has a default configuration, but each API request might specify "Saturday, 8 hours, starting at 09:00" or "Sunday, 6 hours, starting at 10:00."

**Why not just import Problem directly in routes?** Because every route file would need to know the CSV paths, handle loading errors, and manage caching. Centralizing it here means one change updates everything.

---

## `services/scoring_service.py`

**What it is:** The file that applies the user's category preferences to the landmark scores before running any algorithm.

**What it does:**

When the user says "I care more about historical sites than nature," they're assigning weights — for example `{"historical": 1.5, "nature": 0.8}`. A historical landmark with a base score of 8.0 becomes 12.0. A nature landmark with a base score of 7.0 becomes 5.6.

The solver then runs on this modified problem. Because the solver maximizes total score, it naturally prioritizes the categories the user cares about. The algorithm code itself never needs to know about user preferences — they're baked into the scores before the solver even starts.

**What it does NOT do:** It does not modify the cached base problem. It creates a brand new `Problem` object with new `Landmark` objects containing adjusted scores. The original data is never touched.

**Why create new Landmark objects instead of modifying existing ones?** Because your `Landmark` dataclass is `frozen=True` — it's immutable by design. You cannot change `lm.interest_score` in place. You must create a new object. This was a deliberate design decision from Phase 2 and it pays off here — there's no risk of accidentally corrupting the cached base data.

---

## `services/routing_service.py`

**What it is:** The file that talks to map to get real road routes and distances.

**What it does:** Two things.

**Function 1 — `get_route_geometry()`:**
After the solver produces a tour, this function takes the ordered list of stops, sends them to the map public API, and gets back the full road geometry — hundreds of GPS coordinates that trace the exact path along real streets in Algiers. This is what the frontend draws on the map. Not straight lines — actual roads.


map returns a GeoJSON LineString — a list of coordinate pairs. Your backend puts this directly into the API response. The frontend passes it directly to Leaflet. Done.

**Function 2 — `get_leg_distances()`:**
Calls map again (or reuses the same call with a small modification) to get the real road distance for each individual leg — hotel to stop 1, stop 1 to stop 2, etc. These are in kilometers and go into the itinerary panel shown to the user. These are real road distances, not the Haversine approximation used internally by the algorithms.

**What happens if map is unreachable?** Both functions have a `try/except` block. If the map server doesn't respond within 8 seconds (timeout), the functions return a straight-line fallback. The algorithm still ran correctly — you just lose the pretty road-following visualization and get straight lines instead. The demo doesn't crash.

**Why is this in services and not in routes?** Because both `/api/solve` and `/api/solve/all` need road geometry. If it were in the route file, you'd duplicate the code. Putting it in a service means both routes call the same function.

---

## `services/solver_service.py`

**What it is:** The core of the backend. The file that actually runs algorithms and formats results.

**What it does:** Three things.

**The Registry:**
```python
SOLVER_REGISTRY = {
    "grasp":   GraspSolver,
    "greedy":  GreedySolver,
    "sa":      SimulatedAnnealingSolver,
    "genetic": GeneticSolver,
}
```
This dictionary maps a string name (what the frontend sends) to the actual Python class. When the user selects "grasp", the frontend sends `"algorithm": "grasp"`. The backend does `SOLVER_REGISTRY["grasp"]` to get `GraspSolver`. No if-else chain anywhere.

**`run_solver()`:**
Takes an algorithm name, a Problem, and parameters. Creates the right solver, times its execution, runs it, gets a Tour back, then calls `_format_result()` to turn that Tour into a JSON-ready dictionary. Returns the dictionary.

**`run_all_solvers()`:**
Calls `run_solver()` for every algorithm in the registry in a loop. Collects all results, sorts by total score descending, assigns ranks (1st, 2nd, 3rd...), and returns the ranked list plus the full detail of the best tour.

**`_format_result()`:**
This is the most detailed function. It takes a Tour and converts it into everything the frontend needs:
- Each stop with its arrival time, visit start, departure time, waiting time
- Travel time and distance from the previous stop
- The road geometry from map
- Total score, total duration, total distance
- Execution time in milliseconds

This function calls `routing_service` to enrich the tour with real road data before returning.

---

## `schemas/request_schemas.py`

**What it is:** The file that validates incoming API requests before they reach service code.

**What it does:** When the frontend sends a POST request to `/api/solve`, it sends JSON. But what if it sends `"time_budget": "hello"` instead of a number? What if it forgets to include `"algorithm"`? What if `"tour_day"` is `"funday"` which doesn't exist?

Without validation, these errors would crash somewhere deep inside your solver code and return a confusing 500 Internal Server Error. With validation, they return a clear 400 Bad Request with a message like "time_budget must be an integer between 60 and 720."

This file defines what a valid request looks like and provides a function to check incoming data against those rules before any solver runs. Think of it as the security guard at the door — bad requests get turned away immediately with a clear explanation, before they cause damage inside.

---

## How a Single Request Flows Through All These Files

To make everything concrete, here is what happens when a user clicks "Run GRASP" in the frontend:

```
1. Frontend sends: POST /api/solve
   Body: {"algorithm":"grasp","time_budget":720,"tour_day":"saturday",...}

2. main.py receives it, sees /api/solve, hands it to routes/solve.py

3. routes/solve.py:
   - Calls schemas/request_schemas.py → validates the JSON → OK
   - Calls services/problem_loader.py → gets the cached Problem
   - Calls services/scoring_service.py → adjusts scores for category weights
   - Calls services/solver_service.run_solver("grasp", adjusted_problem, params)

4. services/solver_service.py:
   - Looks up GraspSolver in SOLVER_REGISTRY
   - Creates GraspSolver(problem, iterations=50, alpha=0.3)
   - Starts timer
   - Calls solver.solve() → gets Tour object back
   - Stops timer
   - Calls _format_result(tour)

5. _format_result() inside solver_service.py:
   - Reads tour.simulation_cache() for schedule details
   - Calls services/routing_service.get_route_geometry() → calls map
   - Calls services/routing_service.get_leg_distances() → calls map
   - Builds the complete result dictionary

6. routes/solve.py:
   - Receives the result dictionary
   - Returns it as JSON with status 200

7. Frontend receives the JSON:
   - Draws road_geometry on Leaflet map
   - Renders stops in the itinerary panel
   - Displays total_score, total_duration
```

Every file has exactly one job. No file reaches into another file's responsibility. That's the entire architecture.