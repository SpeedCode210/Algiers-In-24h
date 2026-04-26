import math
from typing import Tuple

def distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:

    R = 6371.0  #earth radius
    lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
    lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))
AVG_SPEED = 25.0 #approxmation

def travel_time_minutes(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:

    distance_km = distance(point1, point2)
    return (distance_km / AVG_SPEED) * 60