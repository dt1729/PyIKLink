import pinocchio as pin
import numpy as np
import time
import yaml
import os 
import copy
import json

from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.models.panda import load_models, add_self_collisions


class RelaxedIKVars:
    def __init__(self) -> None:
        self.robot             = pin.Model()
        self.init_state        = []
        self.xopt              = []
        self.prev_state        = []
        self.prev_state2       = []
        self.prev_state3       = []
        self.goal_positions    = []
        self.goal_quats        = []
        self.tolerances        = []
        self.ee_joint_names    = []
        self.init_ee_positions = []
        self.init_ee_rotation  = []

    def from_local_settings(self, path_to_file : str):
        """_summary_

        Args:
            path_to_file (str): _description_
        """
        with open(path_to_file, 'r') as f:
            settings = json.load(f)

        self.base_links_arr = settings["base_links"]
        self.ee_links_arr   = settings["ee_links"]

        # Currently code is being written assuming 1 kinematic chain

        # ill defined in vars.rs
        self.joint_ordering = None
        self.robot, collision_model, _ = load_models()

        # UPDATE THE SETTINGS SANITY CHECK IN A PYTHONIC WAY:
        # if settings["starting_config"].is_badvalue() {
        #     println!("No starting config provided, using all zeros");
        #     for i in 0..robot.num_dofs {
        #         starting_config.push(0.0);
        #     }
        # } else {
        #     let starting_config_arr = settings["starting_config"].as_vec().unwrap();
        #     for i in 0..starting_config_arr.len() {
        #         starting_config.push(starting_config_arr[i].as_f64().unwrap());
        #     }
        # }

        self.pose = self.robot.jointPlacements.tolist()[-1:-2]
        self.init_ee_positions.append(self.pose.translation)
        self.init_ee_rotation.append(self.pose.rotation)

    def update(self, xopt : list):
        """_summary_

        Args:
            xopt (list): _description_
        """
        self.prev_state3 = copy.deepcopy(self.prev_state2)
        self.prev_state2 = copy.deepcopy(self.prev_state)
        self.prev_state = copy.deepcopy(self.xopt)
        self.xopt = copy.deepcopy(xopt)

    def reset(self, init_state: list):
        """_summary_

        Args:
            init_state (list): _description_
        """

        self.prev_state3 = copy.deepcopy(init_state)
        self.prev_state2 = copy.deepcopy(init_state)
        self.prev_state  = copy.deepcopy(init_state)
        self.xopt        = copy.deepcopy(init_state)
        self.init_state  = copy.deepcopy(init_state)

        self.init_ee_positions = []
        self.init_ee_rotation  = []

        self.pose = self.robot.jointPlacements.tolist()[-1]
        self.init_ee_positions.append(self.pose.translation)
        self.init_ee_rotation.append(self.pose.rotation)

        self.goal_positions = copy.deepcopy(self.init_ee_positions)
        self.goal_quats     = copy.deepcopy(self.init_ee_rotation)

        for _ in range(self.tolerances, self.goal_positions):
            self.tolerances.append(
                            np.array(
                                [0., 0., 0., 0., 0., 0.]
                                    )
                                  )
