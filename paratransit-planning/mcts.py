"""
MCTS — exact copy of LogiEx: backend/algo/MCTS/mcts.py
CTL verification skipped (replaced by LLM explanation layer).
"""

import math
import random
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict


# ---------------------------------------------------------------------------
# Abstract Node — exact LogiEx interface
# ---------------------------------------------------------------------------

class Node(ABC):
    @abstractmethod
    def find_children(self):
        return set()

    @abstractmethod
    def find_random_child(self):
        return None

    @abstractmethod
    def is_terminal(self):
        return True

    @abstractmethod
    def reward(self):
        return 0

    @abstractmethod
    def __hash__(self):
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        return True


# ---------------------------------------------------------------------------
# MCTS — exact LogiEx implementation
# ---------------------------------------------------------------------------

class MCTS:
    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.move = None
        self.children = dict()
        self.exploration_weight = exploration_weight
        self.breakdown_Q = defaultdict(lambda: [0, 0])
        self.invalid_actions = dict()

    def choose(self, node, get_children=False):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")
            return self.Q[n] / self.N[n]

        if get_children:
            return max(self.children[node], key=score), self.children[node]
        return max(self.children[node], key=score)

    def run_mcts(self, node, defined_request=None, defined_vehicle=None):
        "Make the tree one layer better for one iteration."
        path = self._select(node, defined_request, defined_vehicle)
        leaf = path[-1]
        self._expand(leaf)
        dc_r = self._rollout(leaf)
        self._backpropagate(path, dc_r[0] + dc_r[1], dc_r)

    def _select(self, node, defined_request=None, defined_vehicle=None):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            for i in range(len(path)):
                try:
                    if path[i].theta[defined_request] != defined_vehicle:
                        return path
                except:
                    pass
            node = self._uct_select(node)

    def _criteria_select(self, node, defined_request, defined_vehicle):
        assert all(n in self.children for n in self.children[node])
        for nd in self.children[node]:
            try:
                if nd.theta[defined_request] == defined_vehicle:
                    return node
            except:
                pass
        return None

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return
        self.children[node], invalid_children = node.find_children()
        if len(invalid_children) > 0:
            self.invalid_actions[node] = invalid_children
        for i, nd in enumerate(self.children[node]):
            pass    # CTL skipped — LLM layer handles explanation

    def _rollout(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        while True:
            if node.is_terminal():
                reward = node.reward(break_down=True)
                return reward
            rand_child = node.find_random_child()
            node = rand_child

    def _backpropagate(self, path, reward, dc_r):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            self.breakdown_Q[node][0] += dc_r[0]
            self.breakdown_Q[node][1] += dc_r[1]

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        assert all(n in self.children for n in self.children[node])
        log_N_vertex = math.log(self.N[node])

        def uct(n):
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)



# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data import REQUESTS, TRAVEL_TIME_MATRIX, make_vehicles
    from route_planner import RoutePlanner
    import copy

    requests_list = [copy.deepcopy(r) for r in REQUESTS]
    vehicles = make_vehicles(num=2, capacity=3)

    # Build initial state — same pattern as LogiEx's assign_passenger()
    route_plan = RoutePlanner(
        requests_list=requests_list,
        travel_time_matrix=TRAVEL_TIME_MATRIX,
        r_t=None, R=[], V=vehicles, theta=dict(),
        t=requests_list[0].request_time, terminal=False
    )
    route_plan = route_plan.append_request(copy.deepcopy(requests_list[0]))

    tree = MCTS()

    os.makedirs("paratransit-planning/mcts_trees", exist_ok=True)
    TREE_DIR = "paratransit-planning/mcts_trees"

    print("Starting MCTS Paratransit Planner (LogiEx)...")
    for _ in range(50):
        tree.run_mcts(route_plan)

    step = 0
    while not route_plan.is_terminal():
        # Save BEFORE choosing — so all sibling branches are visible in the tree
        tree_data = route_plan.to_dict(tree, current_depth=0, max_depth=4)
        with open(f"{TREE_DIR}/mcts_tree_step_{step}.json", "w") as f:
            json.dump(tree_data, f, indent=4, ensure_ascii=False)
        print(f"Saved MCTS tree for step {step} to mcts_trees/mcts_tree_step_{step}.json")

        route_plan, all_children = tree.choose(route_plan, get_children=True)
        print(f"Assigned request {route_plan.R[-1].id} -> "
              f"Vehicle {route_plan.R[-1].assigned_vehicle} | "
              f"theta={route_plan.theta}")
        step += 1

        if not route_plan.is_terminal():
            for _ in range(50):
                tree.run_mcts(route_plan)

    print("\nFinal assignment:", route_plan.theta)
    print("Vehicle status:")
    for v in route_plan.V:
        print(" ", v)
    print("Request outcomes:")
    for r in route_plan.R:
        print(" ", r)
