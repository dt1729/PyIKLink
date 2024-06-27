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




class MatchEEPosGoals(objective_trait):
    def __init__(self) -> None:
        self.arm_idx = None 
    def call(self, x : list, v : vars.RelaxedIKVars, frames : list) -> None:
        # TODO: add assert for arm IDX
        x_val     = np.linalg.norm(frames[self.arm_idx][0][-1] - v.goal_positions[self.arm_idx])
        # groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
        return quadratic_loss(x_val, 0., 2)

    def call_lite(self, x: list, v : vars.RelaxedIKVars, ee_poses : list): # EE poses is list of pin.poses
        x_val = np.linalg.norm(ee_poses[self.arm_idx][0] - v.goal_positions[self.arm_idx])
        return groove_loss(x_val, 0., 2, 0.1, 10.0, 2)


class MatchEEQuatGoals(objective_trait):
    def __init__(self) -> None:
        self.arm_idx = None

    def call(self, x : list, v : vars.RelaxedIKVars, frames : list) -> None:
        ee_quat2 = []
        ee_quat2.append(pin.SE3ToXYZQUAT(frames[self.arm_idx][-1])[-1])
        ee_quat2.append(pin.SE3ToXYZQUAT(frames[self.arm_idx][-1])[3:5]) #This is unit quaternion

        disp = self.angle_between_quaternion(v.goal_quats[self.arm_idx], frames[self.arm_idx][-1])
        disp2 = self.angle_between_quaternion(v.goal_quats[self.arm_idx], ee_quat2)

        x_val = min(disp, disp2)
        # groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
        return quadratic_loss(x_val, 0., 2)

    def call_lite(self, x : list, v : vars.RelaxedIKVars, ee_poses : list) -> None:
        ee_quat2 = []
        ee_quat2.append(pin.SE3ToXYZQUAT(ee_poses[self.arm_idx])[-1])
        ee_quat2.append(pin.SE3ToXYZQUAT(ee_poses[self.arm_idx])[3:5]) #This is unit quaternion

        disp = self.angle_between_quaternion(v.goal_quats[self.arm_idx], ee_poses[self.arm_idx])
        disp2 = self.angle_between_quaternion(v.goal_quats[self.arm_idx], ee_quat2)

        x_val = min(disp, disp2)
        return groove_loss(x_val, 0., 2, 0.1, 10.0, 2)

class MinimizeJerk(objective_trait):
    def call(self, x: list, v: vars.RelaxedIKVars, frames: list):
        x_val = 0.0
        for i in enumerate(x):
            v1 = i[1] - v.xopt[i[0]]
            v2 = v.xopt[i[0]] - v.prev_state[i[0]]
            v3 = v.prev_state[i[0]] - v.prev_state2[i[0]]
            a1 = v1 - v2
            a2 = v2 - v3
            x_val += (a1 - a2)**2

        x_val = x_val**0.5
        return groove_loss(x_val, 0.0, 2, 0.1 , 10.0, 2)

    def call_lite(self, x: list, v: vars.RelaxedIKVars, ee_poses: list):
        x_val = 0.0
        for i in enumerate(x):
            v1 = i[1] - v.xopt[i[0]]
            v2 = v.xopt[i[0]] - v.prev_state[i[0]]
            v3 = v.prev_state[i[0]] - v.prev_state2[i[0]]
            a1 = v1 - v2
            a2 = v2 - v3
            x_val += (a1 - a2)**2

        x_val = x_val**0.5
        return groove_loss(x_val, 0.0, 2, 0.1 , 10.0, 2)

class MinimizeAcceleration(objective_trait):
    def call(self, x: list, v: vars.RelaxedIKVars, frames: list):
        x_val = 0.0

        for i in enumerate(x):
            v1 = i[1] - v.xopt[i[0]]
            v2 = v.xopt[i[0]] - v.prev_state[i[0]]
            x_val = (v1 - v2)**2

        x_val = x_val**0.5
        return groove_loss(x_val, 0.0, 2, 0.1 , 10.0, 2)

    def call_lite(self, x: list, v: vars.RelaxedIKVars, ee_poses: list):
        x_val = 0.0

        for i in enumerate(x):
            v1 = i[1] - v.xopt[i[0]]
            v2 = v.xopt[i[0]] - v.prev_state[i[0]]
            x_val = (v1 - v2)**2

        x_val = x_val**0.5
        return groove_loss(x_val, 0.0, 2, 0.1 , 10.0, 2)
    
class MinimizeVelocity(objective_trait):
    def call(self, x: list, v: vars.RelaxedIKVars, frames: list):
        x_val = 0.0 
        for i in enumerate(x):
            x_val += (i[1] - v.xopt[i[0]])**2
        x_val = x_val**0.5
        return groove_loss(x_val, 0.0, 2, 0.1 , 10.0, 2)

    def call_lite(self, x: list, v: vars.RelaxedIKVars, ee_poses: list):
        x_val = 0.0 
        for i in enumerate(x):
            x_val += (i[1] - v.xopt[i[0]])**2
        x_val = x_val**0.5
        return groove_loss(x_val, 0.0, 2, 0.1 , 10.0, 2)

class EachJointLimits(objective_trait):
    def __init__(self) -> None:
        super().__init__()
        self.joint_idx = None
    
    def call(self, x: list, v: vars.RelaxedIKVars, frames: list):
        #TODO: Check the values for continuous joint in pinocchio
        if v.robot.lowerPositionLimit[self.joint_idx] == None and v.robot.upperPositionLimit[self.joint_idx] == None:
            return -1

        l = v.robot.lowerPositionLimit[self.joint_idx]
        u = v.robot.upperPositionLimit[self.joint_idx]
        
        return swamp_loss(x[self.joint_idx], l, u, 10.0, 10.0, 20)

    def call_lite(self, x: list, v: vars.RelaxedIKVars, ee_poses: list):
        return 0.0

