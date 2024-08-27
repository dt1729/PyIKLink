from pygroove.gradient import forward_diff, central_finite_diff, central_finite_diff_2
from pygroove.vars import RelaxedIKVars
from pygroove.objective_master import ObjectiveMaster

import alpaqa
import casadi 

import numpy as np

class OptimisationEngineOpen():
    def __init__(self, usize : int) -> None:
        self.dim = usize
    
    def optimize(self, x : list, v : RelaxedIKVars, om : ObjectiveMaster, max_iter : int):
        