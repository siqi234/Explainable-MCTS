"""
Adapted from LogiEx: backend/algo/MCTS/routeplanner.py
Only change: replaces CSV chain / request_generator_real with a static
synthetic request list. Everything else (make_move, find_children, reward,
move_vehicle, etc.) is identical to LogiEx.
"""

import copy
import random
from mcts import Node
from objects import Vehicle
from requests import Request


class RoutePlanner(Node):
    """
    Formulate route planning as a MCTS problem.
    State: current request, assigned requests, vehicle fleet, assignment map, time.

    Replaces LogiEx's `chains` + `request_generator_real` with `requests_list`
    — a plain Python list of Request objects ordered by request_time.
    """

    def __init__(self, requests_list, travel_time_matrix, r_t, R, V, theta,
                 t=0, terminal=False, verbose=True):
        self.requests_list = requests_list      # replaces LogiEx's chains
        self.travel_time_matrix = travel_time_matrix
        self.r_t = r_t          # current request being assigned
        self.R = R              # list of already-assigned requests
        self.V = V              # list of Vehicle objects
        self.theta = theta      # {req_id: vehicle_id}
        self.t = t              # current simulation time
        self.terminal = terminal
        self.verbose = verbose
        self.flag = False
        self.violations = []

    # -- LogiEx interface (unchanged) ---------------------------------------

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def is_terminal(self):
        return self.terminal

    def append_request(self, request):
        """Advance time to request arrival and return new state with r_t set."""
        self.move_vehicle(request.request_time)
        return RoutePlanner(
            self.requests_list, self.travel_time_matrix,
            request, self.R, self.V, self.theta,
            request.request_time, self.terminal, self.verbose
        )

    def find_children(self, check_violations=True):
        """Try assigning current request to each vehicle. Returns (valid, invalid)."""
        if self.terminal:
            return (set(), []) if check_violations else set()

        possible_plan = set()
        invalid_plan = []

        for v in self.V:
            curr_v = copy.deepcopy(v)
            curr_r = copy.deepcopy(self.r_t)
            if curr_v.assign_to_vehicle(curr_r, self.t, self.travel_time_matrix):
                next_state = self.make_move(curr_r, curr_v)
                possible_plan.add(next_state)
            else:
                invalid_plan.append((curr_r, curr_v, "vehicle_full"))

        return (possible_plan, invalid_plan) if check_violations else possible_plan

    def find_random_child(self):
        if self.terminal:
            return None
        try:
            return random.sample([*self.find_children(check_violations=False)], 1)[0]
        except ValueError:
            raise RuntimeError("No valid assignment — increase vehicle capacity.")

    def reward(self, break_down=False, a=1.0, b=1.0):
        """
        Identical to LogiEx's reward function.
        Scores fulfilled trips + timing accuracy vs promised pickup/dropoff windows.
        """
        trip_fulfill = 0
        timing = 0.0

        for i in range(len(self.R)):
            if self.R[i].current_status.name == "in_transit":
                trip_fulfill += 1
                timing += (self.R[i].pickup_time - self.R[i].actual_pickup_time) / 10000
                if self.R[i].actual_pickup_time < self.R[i].pickup_time:
                    timing += (self.R[i].actual_pickup_time + 900 - self.R[i].pickup_time) / 10000

            elif self.R[i].current_status.name == "dropped_off":
                trip_fulfill += 1
                timing += (self.R[i].pickup_time - self.R[i].actual_pickup_time) / 10000
                timing += (self.R[i].dropoff_time - self.R[i].actual_dropoff_time) / 10000
                if self.R[i].actual_pickup_time < self.R[i].pickup_time:
                    timing += (self.R[i].actual_pickup_time + 900 - self.R[i].pickup_time) / 10000
                if self.R[i].actual_dropoff_time < self.R[i].dropoff_time:
                    timing += (self.R[i].actual_dropoff_time + 900 - self.R[i].dropoff_time) / 10000

        if not break_down:
            return a * (trip_fulfill / max(len(self.R), 1)) + b * timing
        else:
            return (a * (trip_fulfill / max(len(self.R), 1)), b * timing)

    def move_vehicle(self, next_t):
        """Advance all vehicles forward in time to next_t."""
        for vh in self.V:
            vh.evolve_time(self.t, next_t, self.R, self.travel_time_matrix)

    def make_move(self, r, v):
        """
        Returns a new RoutePlanner state after assigning request r to vehicle v.
        Replaces LogiEx's request_generator_real with direct list lookup.
        """
        next_id = self.r_t.id + 1
        terminal = next_id >= len(self.requests_list)

        if not terminal:
            next_rt = copy.deepcopy(self.requests_list[next_id])
            next_t = next_rt.request_time
        else:
            # No more requests — use a dummy terminal request at far future time
            next_rt = None
            next_t = self.t + 1

        next_r = copy.deepcopy(self.R)
        next_v = [copy.deepcopy(vi) for vi in self.V]

        # Replace the assigned vehicle with the updated copy
        for i in range(len(next_v)):
            if next_v[i].id == v.id:
                next_v[i] = v

        # Evolve all vehicles forward to next request time
        for vh in next_v:
            vh.evolve_time(self.t, next_t, next_r, self.travel_time_matrix)

        next_r.append(r)
        new_theta = copy.deepcopy(self.theta)
        new_theta[r.id] = v.id

        return RoutePlanner(
            self.requests_list, self.travel_time_matrix,
            next_rt, next_r, next_v, new_theta,
            next_t, terminal, self.verbose
        )

    def get_vehicle_status(self):
        return "".join(v.get_occupancy() for v in self.V)

    def to_dict(self, mcts, current_depth=0, max_depth=5):
        """Serialize this node and its subtree to a dict for JSON export.
        Children are keyed by vehicle_id (the action taken at this node).
        """
        n = mcts.N[self]
        q = mcts.Q[self]
        bd = mcts.breakdown_Q[self]
        data = {
            "type": "RoutePlannerNode",
            "state": self.r_t.id if self.r_t is not None else -1,
            "theta": {str(k): v for k, v in self.theta.items()},
            "visits": n,
            "value": round(q, 6),
            "breakdown_Q": {"trip_fulfillment": round(bd[0], 6), "timing": round(bd[1], 6)},
        }
        if current_depth < max_depth:
            children = mcts.children.get(self, set())
            keyed = {}
            for child in children:
                # Key = vehicle assigned to current request (the action taken)
                if self.r_t is not None and self.r_t.id in child.theta:
                    key = str(child.theta[self.r_t.id])
                else:
                    key = str(id(child))
                keyed[key] = child.to_dict(mcts, current_depth + 1, max_depth)
            data["children"] = keyed
        else:
            data["children"] = "Max depth reached"
        return data

    def __str__(self):
        result = f"Time: {self.t}. Assigning: {self.r_t}. Assigned so far: "
        result += "; ".join(str(r) for r in self.R) if self.R else "None."
        result += "\nVehicles:\n" + "".join(str(v) for v in self.V)
        return result
