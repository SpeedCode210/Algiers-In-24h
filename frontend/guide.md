## Frontend Integration Guide

Give this section directly to your frontend team.

---

### Base URL

```
http://localhost:5000/api
```

In production or on the jury demo machine, replace `localhost:5000` with your server's address.

---

### Page Load — call these three on startup

```js
// 1. Load all hotels — populate the hotel selector
const { hotels } = await fetch('/api/hotels').then(r => r.json());
// hotels: [{ id, name, latitude, longitude }, ...]

// 2. Load all landmarks — place pins on the map
const { landmarks } = await fetch('/api/landmarks').then(r => r.json());
// landmarks: [{ id, name, latitude, longitude, interest_score,
//               visit_duration_minutes, category, schedule }, ...]

// 3. Load categories — build the weight preference panel
const { categories } = await fetch('/api/categories').then(r => r.json());
// categories: [{ id, label, default_weight }, ...]
```

Store `hotels`, `landmarks`, and `categories` in your app state. They do not change during a session.

---

### Run one algorithm — `POST /api/solve`

```js
const response = await fetch('/api/solve', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    // Required
    algorithm:   'grasp',         // see algorithm table below
    hotel_id:    'hotel_aurassi', // id from /api/hotels
    time_budget: 720,             // integer minutes (60 to 1440)
    tour_day:    'saturday',      // lowercase day name

    // Optional
    start_time: '09:00',          // default is 09:00
    category_weights: {           // multipliers per category (0.0 to 1.0)
      historical:   1.5,
      attraction:   1.0,
      shopping:     0.3,
    },
    algorithm_params: {           // solver-specific tuning
      iterations: 50,
      alpha: 0.3,
    },
  }),
});

const result = await response.json();
```

**What you get back:**

```json
{
  "algorithm": "grasp",
  "algorithm_label": "GRASP",
  "total_score": 42.5,
  "total_duration_minutes": 387,
  "total_distance_km": 18.4,
  "num_landmarks": 6,
  "is_valid": true,
  "execution_time_ms": 312,

  "hotel": {
    "id": "hotel_aurassi",
    "name": "Hotel El-Aurassi",
    "latitude": 36.7692,
    "longitude": 3.0564
  },

  "stops": [
    {
      "order": 1,
      "id": "casbah",
      "name": "Casbah of Algiers",
      "latitude": 36.7845,
      "longitude": 3.0589,
      "category": "historical",
      "interest_score": 8.9,
      "visit_duration_minutes": 120,
      "arrival_time": "09:12",
      "visit_start_time": "09:12",
      "departure_time": "11:12",
      "waiting_minutes": 0,
      "travel_from_prev_minutes": 12,
      "distance_from_prev_km": 3.2
    }
  ],

  "road_geometry": {
    "type": "LineString",
    "coordinates": [
      [3.0564, 36.7692],
      [3.0571, 36.7699],
      "... hundreds of points following real streets ...",
      [3.0564, 36.7692]
    ]
  }
}
```

**Draw the route on Mapbox:**
```js
// road_geometry is already valid GeoJSON — pass directly
map.addSource('route', {
  type: 'geojson',
  data: { type: 'Feature', geometry: result.road_geometry },
});
map.addLayer({
  id: 'route',
  type: 'line',
  source: 'route',
  paint: { 'line-color': '#e63946', 'line-width': 3 },
});
```

---

### Compare all algorithms — `POST /api/solve/all`

```js
const response = await fetch('/api/solve/all', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    hotel_id:    'hotel_aurassi',
    time_budget: 720,
    tour_day:    'saturday',
    category_weights: { historical: 1.5 },
  }),
});

const { rankings, best_tour, all_results } = await response.json();
```

**Rankings array** — use this for the comparison/leaderboard panel:
```json
[
  { "rank": 1, "algorithm": "grasp",  "algorithm_label": "GRASP",              "total_score": 42.5, "num_landmarks": 6, "total_duration_minutes": 387, "execution_time_ms": 312  },
  { "rank": 2, "algorithm": "tabu",   "algorithm_label": "Tabu Search",         "total_score": 41.0, "num_landmarks": 6, "total_duration_minutes": 401, "execution_time_ms": 890  },
  { "rank": 3, "algorithm": "sa",     "algorithm_label": "Simulated Annealing", "total_score": 39.5, "num_landmarks": 5, "total_duration_minutes": 350, "execution_time_ms": 654  },
  { "rank": 4, "algorithm": "greedy", "algorithm_label": "Greedy",              "total_score": 35.0, "num_landmarks": 5, "total_duration_minutes": 320, "execution_time_ms": 4    }
]
```

`best_tour` has the same shape as a single `/api/solve` response.
`all_results` is a dict keyed by algorithm ID — use it to switch between full results on the map without re-calling the API.

---

### Algorithm IDs

| `algorithm` value | Label |
|---|---|
| `greedy` | Greedy (Score Priority) |
| `greedy_ratio` | Greedy (Score/Time Ratio) |
| `greedy_nearest` | Greedy (Nearest Neighbor) |
| `greedy_random` | SGreedy (Random) |
| `sa` | Simulated Annealing |
| `grasp` | GRASP |
| `tabu` | Tabu Search |
| `ga` | Genetic Algorithm |
| `ga_tailored` | Tailored Genetic Algorithm |
| `cplex` | CPLEX (Exact) |

---

### Error handling

Every error returns JSON with an `error` field:

```js
const response = await fetch('/api/solve', { method: 'POST', ... });

if (!response.ok) {
  const { error, field } = await response.json();
  // field is present on 400 errors to tell you which request field failed
  console.error(`Error on field "${field}": ${error}`);
  return;
}

const result = await response.json();
```

| Status | Meaning |
|---|---|
| `400` | Bad request — check `field` in the response to know which input is wrong |
| `404` | `hotel_id` not found — user selected a hotel not in the dataset |
| `503` | Backend service not ready — a teammate's service file is missing |
| `500` | Unexpected server error — check the Flask terminal for the traceback |

---

### Day names accepted

`sunday` `monday` `tuesday` `wednesday` `thursday` `friday` `saturday`

---

### Valid category weight keys

`historical` `religious` `attraction` `tradition_art` `shopping`

Weight range: `0.0` to `1.0`. Omitting a category leaves its weight at `1.0`.