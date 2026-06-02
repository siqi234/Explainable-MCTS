"""
Custom Gymnasium environment for paratransit vehicle assignment.

MDP framing:
  State   : current assignment progress (which requests assigned to which vehicles)
  Action  : vehicle index to assign the current pending request to
             (action = num_vehicles means "reject" / skip)
  Terminal: all requests have been processed
  Reward  : route quality score computed at each assignment
             (time window penalty + detour penalty; capacity violations are illegal actions)
"""

import gymnasium as gym
import numpy as np
import copy
from gymnasium import spaces
from data import TRAVEL_TIME_MATRIX, VEHICLES, REQUESTS


class VehicleState:
    def __init__(self, vehicle_def):
        self.id = vehicle_def["id"]
        self.capacity = vehicle_def["capacity"]
        self.start_location = vehicle_def["start_location"]
        self.current_location = vehicle_def["start_location"]
        self.occupancy = 0
        # route: list of (req_id, location, action) where action is 'pickup' or 'dropoff'
        self.route = []

    def can_accept(self, request):
        return self.occupancy + request["num_passengers"] <= self.capacity

    def assign(self, request, travel_matrix):
        """Insert request into route. Returns estimated pickup time."""
        self.occupancy += request["num_passengers"]
        # Simple append — pickup then dropoff at end of current route
        self.route.append((request["id"], request["pickup_location"], "pickup"))
        self.route.append((request["id"], request["dropoff_location"], "dropoff"))

    def estimated_pickup_time(self, request, travel_matrix):
        """Estimate arrival time at pickup location given current route."""
        t = 0
        loc = self.current_location
        for (_, stop_loc, action) in self.route:
            t += travel_matrix[loc, stop_loc]
            loc = stop_loc
            if action == "pickup" and stop_loc == request["pickup_location"]:
                return t
        # Not in route yet — estimate from current end of route
        t += travel_matrix[loc, request["pickup_location"]]
        return t

    def total_route_time(self, travel_matrix):
        t = 0
        loc = self.current_location
        for (_, stop_loc, _) in self.route:
            t += travel_matrix[loc, stop_loc]
            loc = stop_loc
        return t

    def clone(self):
        v = VehicleState.__new__(VehicleState)
        v.id = self.id
        v.capacity = self.capacity
        v.start_location = self.start_location
        v.current_location = self.current_location
        v.occupancy = self.occupancy
        v.route = list(self.route)
        return v


class ParatransitEnv(gym.Env):
    """
    At each step, the environment presents the next unassigned request.
    The agent picks a vehicle (0..N-1) to assign it to, or N to reject.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, requests=None, vehicles=None, travel_matrix=None, allow_reject=True):
        super().__init__()

        self._requests_def = requests or REQUESTS
        self._vehicles_def = vehicles or VEHICLES
        self._travel_matrix = np.array(travel_matrix or TRAVEL_TIME_MATRIX)
        self._allow_reject = allow_reject

        self.num_vehicles = len(self._vehicles_def)
        # actions: 0..num_vehicles-1 = assign to that vehicle; num_vehicles = reject
        n_actions = self.num_vehicles + (1 if allow_reject else 0)
        self.action_space = spaces.Discrete(n_actions)

        # Observation: flat vector — for MCTS we manage state ourselves,
        # but gymnasium needs something declared
        obs_size = len(self._requests_def) * (self.num_vehicles + 1)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------------
    # Core gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.vehicles = [VehicleState(v) for v in self._vehicles_def]
        self.requests = list(self._requests_def)   # original defs, ordered by pickup_time
        self.current_request_idx = 0
        self.assignment = {}   # {req_id: vehicle_id or 'rejected'}
        return self._get_obs(), {}

    def step(self, action):
        req = self.requests[self.current_request_idx]
        reward = 0.0
        info = {"req_id": req["id"], "action": action}

        if action < self.num_vehicles:
            vehicle = self.vehicles[action]

            if not vehicle.can_accept(req):
                # Illegal — penalize and don't assign
                reward = -100.0
                info["result"] = "capacity_violation"
            else:
                vehicle.assign(req, self._travel_matrix)
                self.assignment[req["id"]] = action
                reward = self._assignment_reward(req, vehicle)
                info["result"] = "assigned"
        else:
            # Reject
            self.assignment[req["id"]] = "rejected"
            reward = -50.0
            info["result"] = "rejected"

        self.current_request_idx += 1
        done = self.current_request_idx >= len(self.requests)

        return self._get_obs(), reward, done, False, info

    def render(self):
        print(f"\n--- Paratransit State ---")
        print(f"Processed {self.current_request_idx}/{len(self.requests)} requests")
        for v in self.vehicles:
            print(f"  Vehicle {v.id}: occupancy={v.occupancy}/{v.capacity}, route={v.route}")
        print(f"  Assignment so far: {self.assignment}")

    # ------------------------------------------------------------------
    # State save/restore (needed for MCTS tree search)
    # ------------------------------------------------------------------

    def get_state(self):
        """Returns a snapshot of the full environment state."""
        return {
            "vehicles": [copy.deepcopy(v) for v in self.vehicles],
            "current_request_idx": self.current_request_idx,
            "assignment": dict(self.assignment),
        }

    def set_state(self, state):
        """Restores environment to a previously saved snapshot."""
        self.vehicles = [copy.deepcopy(v) for v in state["vehicles"]]
        self.current_request_idx = state["current_request_idx"]
        self.assignment = dict(state["assignment"])

    def valid_actions(self):
        """Returns list of actions that are not capacity violations."""
        if self.current_request_idx >= len(self.requests):
            return []
        req = self.requests[self.current_request_idx]
        valid = [i for i, v in enumerate(self.vehicles) if v.can_accept(req)]
        if self._allow_reject:
            valid.append(self.num_vehicles)
        return valid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        """Simple one-hot encoding of assignment decisions so far."""
        obs = np.zeros(len(self.requests) * (self.num_vehicles + 1), dtype=np.float32)
        for req_id, veh in self.assignment.items():
            req_idx = next(i for i, r in enumerate(self.requests) if r["id"] == req_id)
            veh_idx = veh if veh != "rejected" else self.num_vehicles
            obs[req_idx * (self.num_vehicles + 1) + veh_idx] = 1.0
        return obs

    def _assignment_reward(self, request, vehicle):
        """
        Reward for assigning request to vehicle.
        Penalizes: late pickup (vs pickup_time window) and long detours.
        """
        # Estimate when vehicle will reach pickup location
        eta = vehicle.estimated_pickup_time(request, self._travel_matrix)

        # Lateness penalty: how much past the pickup window
        lateness = max(0, eta - request["pickup_time"])
        lateness_penalty = lateness / 60.0   # convert seconds to minutes

        # Detour penalty: extra route time added by this assignment
        detour = self._travel_matrix[vehicle.current_location, request["pickup_location"]]
        detour_penalty = detour / 60.0

        return -(lateness_penalty + 0.5 * detour_penalty)
