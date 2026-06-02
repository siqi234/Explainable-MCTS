"""
Direct copy from LogiEx: backend/algo/MCTS/objects.py
Vehicle class and request status management.
"""
from enum import Enum
import numpy as np
import copy
from math import ceil


status = Enum('status', 'waiting assigned in_transit dropped_off')


def update_requests(requests, id, status, t, v_id=-1):
    "Updates the request status by id."
    for i in range(len(requests)):
        if requests[i].id == id:
            requests[i].current_status = status
            if status == status.assigned:
                requests[i].assigned_vehicle = v_id
                requests[i].actual_assigned_time = t
            elif status == status.in_transit:
                requests[i].actual_pickup_time = t
            elif status == status.dropped_off:
                requests[i].actual_dropoff_time = t


class Vehicle:
    def __init__(self, travel_time_matrix, num=0, seats=2):
        self.id = num
        self.capacity = seats
        self.occupancy = 0
        self.passengers = []
        self.passengers_assigned = []
        self.route = []
        self.next_time = None
        self.current_location = np.random.choice(len(travel_time_matrix))
        self.travel_history = [(-1, self.current_location)]
        self.travel_plan = []

    def get_occupancy(self) -> str:
        repr_string = "Car {}: ".format(self.id)
        repr_string += "occupancy=" + str(self.occupancy)
        repr_string += "; capacity=" + str(self.capacity) + ". "
        return repr_string

    def __str__(self) -> str:
        repr_string = "Car {}: ".format(self.id)
        repr_string += "Onboard Passengers: "
        if len(self.passengers) > 0:
            repr_string += " ".join(str(p.id) for p in self.passengers)
            repr_string += ". "
        else:
            repr_string += "None. "
        repr_string += "Assigned Passengers: "
        if len(self.passengers_assigned) > 0:
            repr_string += " ".join(str(p.id) for p in self.passengers_assigned)
            repr_string += ". "
        else:
            repr_string += "None. "
        repr_string += "Fulfilled Passengers and Locations: "
        if len(self.travel_history) > 1:
            repr_string += "--> ".join(str(ti) for ti in self.travel_history[1:])
            repr_string += ". "
        else:
            repr_string += "None. "
        return repr_string

    def evolve_time(self, cur_t, ev_t, next_r, travel_time_matrix):
        for ti in range(cur_t, ev_t + 1):
            self.check_time(ti, next_r, travel_time_matrix)

    def check_capacity(self, r, i_p, i_d):
        new_route = copy.deepcopy(self.route)
        new_route.insert(i_p + 1, (r.id, r.pickup_location))
        new_route.insert(i_d + 1, (r.id, r.dropoff_location))

        new_plan = copy.deepcopy(self.travel_plan)
        new_plan.insert(i_p + 1, (r.id, 1 * r.num_passengers))
        new_plan.insert(i_d + 1, (r.id, -1 * r.num_passengers))

        capacity_counter = len(self.passengers)
        for pl in new_plan:
            capacity_counter += pl[1]
            if capacity_counter > self.capacity:
                return False
        return True

    def assign_to_vehicle(self, r, t, travel_time_matrix):
        if self.occupancy >= self.capacity:
            return False

        if len(self.route) == 0:
            self.route.append((r.id, r.pickup_location))
            self.route.append((r.id, r.dropoff_location))
            self.travel_plan.append((r.id, 1 * r.num_passengers))
            self.travel_plan.append((r.id, -1 * r.num_passengers))
            self.next_time = int(t + travel_time_matrix[self.current_location, r.pickup_location])
        else:
            min_dist_p = 1e10
            closest_index_p = None
            for i in range(len(self.route)):
                cell = self.route[i][1]
                dist_temp = travel_time_matrix[cell, r.pickup_location]
                if dist_temp < min_dist_p:
                    closest_index_p = i
                    min_dist_p = dist_temp

            min_dist_d = 1e10
            closest_index_d = None
            for i in range(closest_index_p + 1, len(self.route)):
                cell = self.route[i][1]
                dist_temp = travel_time_matrix[cell, r.dropoff_location]
                if dist_temp < min_dist_d:
                    closest_index_d = i
                    min_dist_d = dist_temp

            if closest_index_d is None and closest_index_p == len(self.route) - 1:
                closest_index_d = closest_index_p + 1

            if self.check_capacity(r, closest_index_p, closest_index_d) == False:
                return False
            else:
                self.route.insert(closest_index_p + 1, (r.id, r.pickup_location))
                self.route.insert(closest_index_d + 1, (r.id, r.dropoff_location))
                self.travel_plan.insert(closest_index_p + 1, (r.id, 1 * r.num_passengers))
                self.travel_plan.insert(closest_index_d + 1, (r.id, -1 * r.num_passengers))
                self.next_time = int(t + travel_time_matrix[self.current_location, self.route[0][1]])

        self.passengers_assigned.append(r)
        update_requests(self.passengers_assigned, r.id, status.assigned, t, self.id)
        r.current_status = status.assigned
        r.actual_assigned_time = t
        r.assigned_vehicle = self.id
        return True

    def pick_up_passenger(self, p_id):
        i = 0
        while i < len(self.passengers_assigned):
            if self.passengers_assigned[i].id == p_id:
                self.occupancy += self.passengers_assigned[i].num_passengers
                self.passengers_assigned[i].current_status = status.in_transit
                self.passengers.append(self.passengers_assigned[i])
                self.passengers_assigned.pop(i)
            else:
                i += 1

    def dropoff_passenger(self, p_id):
        i = 0
        while i < len(self.passengers):
            if self.passengers[i].id == p_id:
                self.occupancy -= self.passengers[i].num_passengers
                self.passengers[i].current_status = status.dropped_off
                self.passengers.pop(i)
            else:
                i += 1

    def check_time(self, t, next_r, travel_time_matrix):
        if self.next_time == t:
            if len(self.route) > 0:
                self.current_location = self.route[0][1]

            i = 0
            while i < len(self.route):
                if self.route[i][1] == self.current_location:
                    if self.travel_plan[i][1] >= 1:
                        update_requests(self.passengers_assigned, self.travel_plan[i][0], status.in_transit, t)
                        update_requests(next_r, self.travel_plan[i][0], status.in_transit, t)
                        self.pick_up_passenger(self.travel_plan[i][0])
                    elif self.travel_plan[i][1] <= -1:
                        update_requests(self.passengers, self.travel_plan[i][0], status.dropped_off, t)
                        update_requests(next_r, self.travel_plan[i][0], status.dropped_off, t)
                        self.dropoff_passenger(self.travel_plan[i][0])

                    removed_loc = self.route.pop(i)
                    self.travel_plan.pop(i)
                    self.travel_history.append(removed_loc)
                else:
                    break

            if len(self.route) > 0:
                self.next_time = int(t + ceil(travel_time_matrix[self.current_location, self.route[0][1]]))
