"""
Synthetic paratransit scenario using LogiEx's Request and Vehicle classes.
All times are in seconds since midnight (e.g. 28800 = 8:00 AM).
Locations are node IDs into the travel_time_matrix.
"""

import numpy as np
from requests import Request
from objects import Vehicle

# 8 locations (nodes): node 0 = depot
TRAVEL_TIME_MATRIX = np.array([
    [  0, 300, 600, 900, 480, 720, 360, 540],
    [300,   0, 420, 780, 300, 600, 240, 480],
    [600, 420,   0, 480, 600, 360, 540, 300],
    [900, 780, 480,   0, 720, 300, 660, 420],
    [480, 300, 600, 720,   0, 540, 180, 360],
    [720, 600, 360, 300, 540,   0, 480, 240],
    [360, 240, 540, 660, 180, 480,   0, 300],
    [540, 480, 300, 420, 360, 240, 300,   0],
])

# 5 requests ordered by request_time (arrival time)
# r_t (request_time) < p_t (pickup_time) < d_t (dropoff_time deadline)
REQUESTS = [
    Request(id_num=0, p_t=28800, d_t=30600, p_l=1, d_l=5, r_t=28500, n_p=1),  # 8:00 pickup -> by 8:30
    Request(id_num=1, p_t=28800, d_t=31200, p_l=2, d_l=6, r_t=28560, n_p=1),  # 8:00 pickup -> by 8:40
    Request(id_num=2, p_t=29400, d_t=32400, p_l=4, d_l=3, r_t=29100, n_p=2),  # 8:10 pickup -> by 9:00
    Request(id_num=3, p_t=29700, d_t=32100, p_l=7, d_l=5, r_t=29400, n_p=1),  # 8:15 pickup -> by 8:55
    Request(id_num=4, p_t=30000, d_t=33000, p_l=1, d_l=3, r_t=29700, n_p=1),  # 8:20 pickup -> by 9:10
]


def make_vehicles(num=2, capacity=3, seed=0):
    """Create vehicles starting at depot (node 0)."""
    np.random.seed(seed)
    vehicles = []
    for i in range(num):
        v = Vehicle(TRAVEL_TIME_MATRIX, num=i, seats=capacity)
        v.current_location = 0  # fix all vehicles to start at depot
        v.travel_history = [(-1, 0)]
        vehicles.append(v)
    return vehicles
