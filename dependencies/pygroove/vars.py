import pinocchio as pin
import numpy as np
import time
import yaml
import os 


class RelaxedIKVars:
    def __init__(self) -> None:
        self.robot = pin.Model()
        self.init_state = np.array
        self.xopt = np.array
        self.prev_state = np.array
        self.prev_state2 = np.array
        self.prev_state3 = np.array
        self.goal_positions = np.array
        self.goal_quats     = np.array
        self.tolerances     = np.array
        self.init_ee_positions = np.array
        self.init_ee_quats     = np.array

    def from_local_settings(self, path_to_file : str):
        """_summary_

        Args:
            path_to_file (str): _description_
        """
        