from pygroove.vars import RelaxedIKVars
from pygroove.groove import OptimisationEngineOpen
from pygroove.objective_master import ObjectiveMaster
from ABC import abc

import os
import pinocchio as pin
import numpy as np
import copy

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
        self.groove         = OptimisationEngineOpen() #TODO: Add number of dofs

    def solve(self, constraint_velocity : bool):
        out_x = copy.deepcopy(self.vars.xopt)
        
        if constraint_velocity: 
            self.groove.optimize(out_x, self.vars, self.om_relaxedik, 100)
        else:
            self.groove.optimize(out_x, self.vars, self.om_standardik, 1000)
            
        if None in out_x:
            return self.vars.xopt
        
        self.vars.update(out_x)
        
        return out_x.tolist() #TODO: Check what array type optimizer gives out if np.array then convert to list                 