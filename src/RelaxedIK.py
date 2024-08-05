from pygroove.vars import RelaxedIKVars
from pygroove.groove import OptimisationEngineOpen
from pygroove.objective_master import ObjectiveMaster
from ABC import abc

import os
import pinocchio as pin
import numpy as np


class Opt():
    def __init__(self) -> None:
        self.data = 0.0
        self.length = 0
    
class RelaxedIK():
    def __init__(self) -> None:
        self.vars          = RelaxedIKVars
        self.om_relaxedik  = ObjectiveMaster
        self.om_standardik = ObjectiveMaster
        self.groove        = OptimisationEngineOpen
        
    def load_settings(self, path_to_setting : str):
        """Loads settings into initialised variables

        Args:
            path_to_setting (str): global path to settings file
        """
        print(f"RelaxedIK is using below setting file {path_to_setting}")

        self.vars           = RelaxedIKVars.from_local_settings(path_to_setting)
        self.om_relaxedik   = ObjectiveMaster.relaxed_ik(vars.robot.chain_indices)
        self.om_standardik  = ObjectiveMaster.standard_ik(vars.robot.chain_indices)
        self.groove         = OptimisationEngineOpen(vars.robot.nv)
    