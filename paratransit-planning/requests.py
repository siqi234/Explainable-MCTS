"""
Direct copy from LogiEx: backend/algo/MCTS/requests.py
Request class + request_generator_real for loading from CARTA CSV chains.
When no CSV data is available, data.py provides synthetic requests instead.
"""
from enum import Enum

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


status = Enum('status', 'waiting assigned in_transit dropped_off')


def seconds_to_hour(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:02}:{minutes:02}"


class Request:
    def __init__(self, id_num=None, p_t=0, d_t=0, p_l=0, d_l=0, r_t=0, n_p=1):
        self.id = id_num
        self.pickup_time = p_t
        self.dropoff_time = d_t
        self.pickup_location = p_l
        self.dropoff_location = d_l
        self.request_time = r_t
        self.actual_pickup_time = -1
        self.actual_dropoff_time = -1
        self.actual_assigned_time = -1
        self.current_status = status.waiting
        self.assigned_vehicle = -1
        self.num_passengers = n_p

    def get_info(self, key):
        if key == 'pickup_time':
            return self.pickup_time
        elif key == 'dropoff_time':
            return self.dropoff_time
        elif key == 'pickup_location':
            return self.pickup_location
        elif key == 'dropoff_location':
            return self.dropoff_location
        elif key == 'request_time':
            return self.request_time
        elif key == 'actual_pickup_time':
            return self.actual_pickup_time
        elif key == 'actual_dropoff_time':
            return self.actual_dropoff_time
        elif key == 'actual_assigned_time':
            return self.actual_assigned_time
        elif key == 'current_status':
            return self.current_status
        else:
            raise ValueError("ERROR: Request.get_info got wrong key.")

    def get_request_time(self):
        return "Pick-Up Time: {}, Drop-Off Time: {}".format(
            seconds_to_hour(self.pickup_time), seconds_to_hour(self.dropoff_time))

    def __str__(self) -> str:
        return "Pick-Up Time: {}, Drop-Off Time: {}, Status: {}".format(
            seconds_to_hour(self.pickup_time), seconds_to_hour(self.dropoff_time),
            self.current_status.name)


def request_generator_real(chains, id_num, chain_id=9, relative_t=-1, spec_req_id=None):
    """
    Direct copy from LogiEx: generates a Request from CARTA CSV chain data.
    Requires pandas and a loaded chains DataFrame.
    Switch from synthetic data to real data by passing train_chains DataFrame here.
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError("pandas is required for request_generator_real. Install it or use synthetic data.")

    if relative_t < 0:
        stop = False
        while not stop:
            selected_val = chains.loc[(chains['chain_id'] == chain_id) & (chains['chain_order'] == id_num)]
            try:
                p_t = int(selected_val['pickup_time_since_midnight'])
                stop = True
            except:
                id_num += 1
    else:
        selected_val = chains.loc[(chains['chain_id'] == chain_id) & (chains['chain_order'] == id_num)]
        while int(selected_val['request_arrive_time']) < relative_t:
            id_num += 1
            selected_val = chains.loc[(chains['chain_id'] == chain_id) & (chains['chain_order'] == id_num)]

    p_t = int(selected_val['pickup_time_since_midnight'])
    d_t = int(selected_val['dropoff_time_since_midnight'])
    p_l = int(selected_val['pickup_node_id'])
    d_l = int(selected_val['dropoff_node_id'])
    r_t = int(selected_val['request_arrive_time'])

    if spec_req_id:
        return Request(id_num=spec_req_id, p_t=p_t, d_t=d_t, p_l=p_l, d_l=d_l, r_t=r_t)
    return Request(id_num=id_num, p_t=p_t, d_t=d_t, p_l=p_l, d_l=d_l, r_t=r_t)
