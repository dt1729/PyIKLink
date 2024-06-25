import numpy as np
import copy
import vars
import time
import pinocchio as pin

class objective_trait:
    """_summary_
    """
    def call(self, x : list, v : vars.RelaxedIKVars, frames : list):
        """_summary_

        Args:
            x (list): _description_
            v (vars.RelaxedIKVars): _description_
            frames (list): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    def call_lite(self, x: list, v : vars.RelaxedIKVars, ee_poses : list): # EE poses is list of pinocchio.pinocchio_pywrap.SE3
        """_summary_

        Args:
            x (list): _description_
            v (vars.RelaxedIKVars): _description_
            ee_poses (list): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    def gradient(self, x : list, v : vars.RelaxedIKVars, frames : list) -> tuple:
        """_summary_

        Args:
            x (list): _description_
            v (vars.RelaxedIKVars): _description_
            frames (list): _description_

        Returns:
            tuple: _description_
        """
        grad = []
        f_0 = self.call(x, v, frames)
        temp = v.robot.frame.tolist()
        
        for i in range(len(x)):
            x_h = copy.deepcopy(x)
            x_h[i] += 0.000000001
            frames_h = [i.placement for i in temp]
            f_h      = self.call(copy.deepcopy(x_h), v, frames_h)
            grad.append((f_h - f_0)/0.000000001)

        return f_0, grad

    def gradient_lite(self, x: list, v : vars.RelaxedIKVars, ee_poses : list) -> tuple:
        """_summary_

        Args:
            x (list): _description_
            v (vars.RelaxedIKVars): _description_
            ee_poses (list): _description_

        Returns:
            tuple: _description_
        """
        grad = []
        f_0 = self.call_lite(x, v, ee_poses)
        temp = v.robot.frame.tolist()
        temp_names = v.robot.names.tolist()
        for i in range(len(x)):
            x_h         = copy.deepcopy(x)
            x_h[i]      += 0.000000001
            ee_poses_h  = [temp[temp_names.index(i)] for i in v.ee_joint_names]
            f_h         = self.call_lite(copy.deepcopy(x_h), v, ee_poses_h)
            grad.append((f_h - f_0)/0.000000001)

        return f_0, grad

        

    

class MatchEEPosGoals:
    def __init__(self) -> None:
        self.arm_idx = None
        
    def objective_trait(self) -> None:
        def call(self, x : list, v : vars.RelaxedIKVars, frames : list) -> None:
            # TODO: add assert for arm IDX
            x_val     = np.linalg.norm(frames[self.arm_idx][0][-1] - v.goal_positions[self.arm_idx])
            # groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
            quadratic_loss(x_val, 0., 2)
        
        def call_lite(self, x: list, v : vars.RelaxedIKVars, ee_poses : list): # EE poses is list of pin.poses
            x_val = np.linalg.norm(ee_poses[self.arm_idx][0] - v.goal_positions[self.arm_idx])
            groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
        
class MatchEEQuatGoals:
    def __init__(self) -> None:
        self.arm_idx = None

    def objective_trait(self) -> None:
        def call(self, x : list, v : vars.RelaxedIKVars, frames : list) -> None:
            tmp = pin.quaternion()

