"""routing_service.py
Fetches real road geometry and leg distances from Mapbox Directions API.

Two public functions are consumed by solve.py:

    get_route_geometry(ordered_stops, hotel)
        → GeoJSON LineString that traces the full road path of the tour.
          Used by the Leaflet map on the frontend.

    get_leg_distances(ordered_stops, hotel)
        → list[float] of real road distances in km, one per stop.
          Used by the itinerary panel (hotel→stop1, stop1→stop2, …).

If Mapbox is unreachable (timeout, connection error, invalid key),
both functions raise an exception so that solve.py can catch it and fall
back to the straight-line geometry it already builds internally.
"""

from __future__ import annotations

import requests


MAPBOX_ACCESS_TOKEN: str = "pk.eyJ1IjoibW9oYW1lZGVnaGJvdWRqIiwiYSI6ImNtb24ybnZsaTBrOXMycHF5MGc3d2dmcXAifQ.0QWaEMordbNjiOwJD7ZqMA"

MAPBOX_BASE_URL: str = "https://api.mapbox.com/directions/v5/mapbox/driving"

REQUEST_TIMEOUT: int = 8


def _build_coordinate_string(ordered_stops, hotel) -> str:
    """Build the semicolon-separated lon,lat string that Mapbox expects.

    Mapbox's Directions API requires coordinates in the form:
        lon1,lat1;lon2,lat2;…;lonN,latN

    The tour always starts and ends at the hotel, so the full sequence is:
        hotel → stop1 → stop2 → … → stopN → hotel

    Args:
        ordered_stops : List of Landmark objects in visit order (hotel excluded).
        hotel         : The hotel Landmark (start/end point).

    Returns:
        A semicolon-separated coordinate string ready for the Mapbox URL.
    """
    waypoints = [hotel] + list(ordered_stops) + [hotel]
    return ";".join(f"{lm.longitude},{lm.latitude}" for lm in waypoints)


def _call_mapbox_directions(coordinate_string: str) -> dict:
    """Call the Mapbox Directions API and return the parsed JSON.

    Args:
        coordinate_string : Semicolon-separated "lon,lat" pairs.

    Returns:
        Parsed JSON dict from Mapbox.

    Raises:
        requests.exceptions.RequestException : Network / timeout error.
        ValueError                           : Mapbox returned an error
                                               or an unexpected response body.
    """
    url = (
        f"{MAPBOX_BASE_URL}/{coordinate_string}"
        f"?geometries=geojson&overview=full&steps=false"
        f"&access_token={MAPBOX_ACCESS_TOKEN}"
    )

    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()  # raises HTTPError for 4xx/5xx

    data = response.json()

    if data.get("code") != "Ok":
        raise ValueError(
            f"Mapbox returned an error code: {data.get('code')} — "
            f"message: {data.get('message', 'no message')}"
        )

    return data


def get_route_geometry(ordered_stops: list, hotel) -> dict:
    """Fetch the full road geometry for a tour as a GeoJSON LineString.

    Calls Mapbox's Directions API with the complete sequence of waypoints
    (hotel → stops → hotel) and extracts the overview geometry.

    The returned GeoJSON LineString is placed directly into the API response
    and passed to the map on the frontend, which draws it as a polyline
    following real roads in Algiers.

    Args:
        ordered_stops : Landmark objects in visit order (hotel excluded).
                        May be empty — in that case a single-point "route"
                        (hotel → hotel) is returned.
        hotel         : The hotel Landmark used as start and end point.

    Returns:
        A GeoJSON LineString dict, e.g.::

            {
                "type": "LineString",
                "coordinates": [
                    [3.0589, 36.7845],   # [longitude, latitude]
                    [3.0601, 36.7852],
                    ...
                ]
            }

    Raises:
        requests.exceptions.RequestException : Mapbox is unreachable or timed out.
        ValueError                           : Mapbox returned an unexpected response.
    """
    if not ordered_stops:
        return {
            "type": "LineString",
            "coordinates": [
                [hotel.longitude, hotel.latitude],
                [hotel.longitude, hotel.latitude],
            ],
        }

    coordinate_string = _build_coordinate_string(ordered_stops, hotel)
    data = _call_mapbox_directions(coordinate_string)

    route = data["routes"][0]
    geometry = route["geometry"]

    return geometry


def get_leg_distances(ordered_stops: list, hotel) -> list[float]:
    """Fetch the real road distance (km) for each leg of the tour.

    A "leg" is one segment of the journey:
        leg 0 → hotel   to stop 1
        leg 1 → stop 1  to stop 2
        …
        leg N → stop N  to hotel   (return leg, not included in the output)

    The output list has exactly len(ordered_stops) elements — one distance
    per stop, representing the road distance from the previous location to
    that stop. The final return leg (last stop → hotel) is excluded because
    the itinerary panel only shows distances to each stop, not the return.

    Args:
        ordered_stops : Landmark objects in visit order (hotel excluded).
        hotel         : The hotel Landmark used as start and end point.

    Returns:
        List of float distances in kilometres, one per stop.
        Example for a 3-stop tour: [1.4, 2.7, 0.9]

    Raises:
        requests.exceptions.RequestException : Mapbox is unreachable or timed out.
        ValueError                           : Mapbox returned an unexpected response.
    """
    if not ordered_stops:
        return []

    coordinate_string = _build_coordinate_string(ordered_stops, hotel)
    data = _call_mapbox_directions(coordinate_string)

    route = data["routes"][0]


    legs: list[dict] = route["legs"]

    distances_km: list[float] = [
        round(leg["distance"] / 1000.0, 3)
        for leg in legs[: len(ordered_stops)]  # here we exclude last (return) leg
    ]

    return distances_km