from vars import RelaxedIKVars
from groove import objective as obj

class ObjectiveMaster:
    def __init__(self) -> None:
        self.objectives = []
        self.num_chains = int
        self.weight_priors = []
        self.lite = bool
        self.finite_diff_grad = bool

    def standard_ik(self, chain_indices : list):
        self.objectives = []
        self.weight_priors = []
        self.num_chains = len(chain_indices)
        for i in range(self.num_chains):
            self.objectives.append(obj.MatchEEPosGoals(i))
            self.weight_priors.append(10.0)
            self.objectives.append(obj.MatchEEQuatGoals(i))
            self.weight_priors.append(1.0)
        return self
    
    def relaxed_ik(self, chain_indices : list):
        self.objectives = []
        self.weight_priors = []
        self.num_chains = len(chain_indices)
        self.num_dofs   = 0
        for i in range(self.num_chains):
            self.objectives.append(obj.MatchEEPosiDoF(i, 0))
            self.weight_priors.append(50.0)
            self.objectives.append(obj.MatchEEPosiDoF(i, 1))
            self.weight_priors.append(50.0)
            self.objectives.append(obj.MatchEEPosiDoF(i, 2))
            self.weight_priors.append(50.0)
            self.objectives.append(obj.MatchEERotaDoF(i, 0))
            self.weight_priors.append(1.0)
            self.objectives.append(obj.MatchEERotaDoF(i, 1))
            self.weight_priors.append(1.0)
            self.objectives.append(obj.MatchEERotaDoF(i, 2))
            self.weight_priors.append(1.0)
        
        self.num_dofs = 
        