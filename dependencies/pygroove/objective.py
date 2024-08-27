import numpy as np
import copy
import math
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
    def __init__(self, i : int) -> None:
        self.arm_idx = i 
    def call(self, x : list, v : vars.RelaxedIKVars, frames : list) -> None:
        # TODO: add assert for arm IDX
        x_val     = np.linalg.norm(frames[-1].placement.translation - v.goal_positions[self.arm_idx])
        # groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
        return quadratic_loss(x_val, 0., 2)

    def call_lite(self, x: list, v : vars.RelaxedIKVars, ee_poses : list): # EE poses is list of pin.poses
        x_val = np.linalg.norm(ee_poses[0] - v.goal_positions[self.arm_idx])
        return groove_loss(x_val, 0., 2, 0.1, 10.0, 2)


class MatchEEQuatGoals(objective_trait):
    def __init__(self) -> None:
        self.arm_idx = None

    def call(self, x : list, v : vars.RelaxedIKVars, frames : list) -> None:
        ee_quat = []
        ee_quat.append(pin.SE3ToXYZQUAT(frames[-1])[-1])
        ee_quat.append(pin.SE3ToXYZQUAT(frames[-1])[3:5]) #This is end effector unit quaternion
        ee_quat2 = -1 * ee_quat
        
        disp = self.angle_between_quaternion(v.goal_quats[self.arm_idx], ee_quat)
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

class MaximizeManipulability(objective_trait):
    def __init__(self) -> None:
        super().__init__()

    def call(self, x: list, v: vars.RelaxedIKVars, frames: list):
        jac   = pin.computeJointJacobians(v.robot,v.robot.create_data(), v.robot.q0)
        x_val = np.linalg.det(np.dot(jac, np.transpose(jac)))**0.5
        return groove_loss(x_val, 1.0, 2, 0.5, 0.1, 2)

    def call_lite(self, x: list, v: vars.RelaxedIKVars, ee_poses: list):
        return 0.0

class SelfCollision(objective_trait):
    def __init__(self) -> None:
        self.arm_idx = None
        self.first_link = None
        self.second_link = None

    def call(self, x: list, v: vars.RelaxedIKVars, frames: list):

        for i in enumerate(x):
            if i[1] is None:
                i[1] = 10.0

        geom_model = pin.buildGeomFromUrdf(v.robot,
                                           v.urdf_model_path,
                                           v.mesh_dir,
                                           pin.GeometryType.COLLISION
                                           )
        geom_model.addAllCollisionPairs()
        geom_data  = pin.GeometryData(geom_model)

        pin.computeCollisions(v.robot,
                              v.robot.create_data(),
                              geom_model,
                              geom_data,
                              v.robot.q0,
                              False)

        k = [i for i in range(len(geom_model.collisionPairs))
             if geom_model.collisionPairs[i].first == frames[self.arm_idx][0][self.first_link]
             and
             geom_model.collisionPairs[i].first == frames[self.arm_idx][0][self.second_link]
            ]

        dist = np.linalg.norm(pin.SE3ToXYZQUAT(frames[self.arm_idx][k])[0:2] -
                              pin.SE3ToXYZQUAT(frames[self.arm_idx][k])[0:2]) - 0.05

        return swamp_loss(dist, 0.02, 1.5, 60.0, 0.0001, 30)

    def call_lite(self, x: list, v: vars.RelaxedIKVars, ee_poses: list):
        x_val = 1.0
        return groove_loss(x_val, 0., 2, 2.1, 0.0002, 4)

class MatchEERotaDof(objective_trait):
    def __init__(self, arm_idx : int, axis : int) -> None:
        super().__init__()
        self.arm_idx = arm_idx
        self.axis = axis
        
    def call(self, x: list, v: vars.RelaxedIKVars, frames: list):
        last_elem = frames[self.arm_idx][-1]
        ee_quat   = pin.SE3ToXYZQUAT(last_elem)[3:5]
        goal_quat = v.goal_quats[self.arm_idx]
        rotation  = goal_quat.inverse()*ee_quat # TODO: Find the inverse and multiplication operation for quaternion and see if pinocchio uses that.
        
        # TODO: Axis angle representation 
        euler = rotation.euler_angles() 
        scaled_axis = rotation.scaled_axis()
        
        angle = 0.0
        angle += np.abs(scaled_axis[self.arm_idx])

        bound = v.tolerances[self.arm_idx][self.axis + 3]

        if bound < 1e-2:
            return groove_loss(angle, 0., 2, 0.1, 10.0, 2)
        else:
            if bound >= 3.14159260:
                return swamp_loss(angle, -bound, bound, 100.0, 0.1, 20)
            else:
                return swamp_groove_loss(angle, 0.0, -bound, bound, bound*2.0, 1.0, 0.01, 100.0, 20)

    def call_lite(self, x: list, v: vars.RelaxedIKVars, ee_poses: list):
        x_val = np.linalg.norm(ee_poses[self.arm_idx][0]- v.goal_positions[self.arm_idx])
        return groove_loss(x_val, 0., 2, 0.1, 10.0, 2)

class MatchEEPosiDoF(objective_trait):
    def __init__(self, arm_idx : int, axis : int) -> None:
        super().__init__()
        self.arm_idx = arm_idx
        self.axis = axis

    def call(self, x: list, v: vars.RelaxedIKVars, frames: list):
        goal_quat = v.goal_quats[self.arm_idx]
        last_elem_pos = pin.SE3ToXYZQUAT(frames[self.arm_idx][-1])[0:2]
        T_gw_T_wc = np.array([  last_elem_pos[0] - v.goal_positions[self.arm_idx].x,\
                                last_elem_pos[1] - v.goal_positions[self.arm_idx].y,\
                                last_elem_pos[2] - v.goal_positions[self.arm_idx].z])
        T_gc      = goal_quat.inverse() * T_gw_T_wc #TODO: Fix quaternion conversion here
        dist      = T_gc[self.axis]
        bound     = v.tolerances[self.arm_idx][self.axis]

        if bound < 1e-2:
            return groove_loss(angle, 0., 2, 0.1, 10.0, 2)
        else:
            return swamp_groove_loss(dist, 0.0, -bound, bound, bound*2.0, 1.0, 0.01, 100.0, 20)

    def call_lite(self, x: list, v: vars.RelaxedIKVars, ee_poses: list):
        x_val = np.linalg.norm(ee_poses[self.arm_idx][0]- v.goal_positions[self.arm_idx])
        return groove_loss(x_val, 0., 2, 0.1, 10.0, 2)

def quadratic_loss(x_val : float, t : float, g : float):
    return (x_val - t)**g

def groove_loss(x_val: float, t: float, d: int, c: float, f: float, g: int):
    return np.exp(-((-(x_val - t)**d)/(2.0 * c**2))) + f * (x_val - t)**g

def groove_loss_derivative(x_val: float, t: float, d: int, c: float, f: float, g: int):
    return np.exp(-((-(x_val - t)**d)/(2.0 * c**2))) * ((float(d)*(x_val - t))/(2.0 * c**2)) + float(g)*f*(x_val - t)

def swamp_loss(x_val: float, l_bound: float, u_bound: float, f1: float, f2: float, p1: int):
    x = (2.0 * x_val - l_bound - u_bound) / (u_bound - l_bound)
    b = (np.log(-1.0 / (0.05)))**(1.0 / float(p1))
    return (f1 + f2 * x**2) * (1.0 - (np.exp(-(x/b)**(p1)))) - 1.0

def swamp_groove_loss(x_val: float, g: float, l_bound: float, u_bound: float, c: float, f1: float, f2: float, f3: float, p1: int):
    x = (2.0 * x_val - l_bound - u_bound) / (u_bound - l_bound)
    b = np.log(-1.0 / 0.05)**(1.0/p1)  # powf(1.0 / p1 as f64)
    t1 = -f1 * np.exp( (-(x_val - g)**2) / (np.exp(2.0 * (c**2))) ) \
         +f2 * (x_val - g)**2 \
         +f3 * (1.0 - np.exp(-(x/b)**p1))

    return t1


def swamp_groove_loss_derivative(x_val: float, g: float, l_bound: float, u_bound: float, c : float, f1: float, f2: float, f3: float, p1: int):
    if math.fabs(2.0*x_val - l_bound - u_bound) < 1e-8:
        return 0.0

    x =  (2.0 * x_val - l_bound - u_bound) / (u_bound - l_bound)
    b = (np.log(-1.0 / (0.05)))**(1.0 / float(p1))

    return - f1 * np.exp( (-x_val**(2)) / (2.0 * c**(2) ) ) *  ((-2.0 * x_val) /  (2.0 * c**(2))) \
    + 2.0 * f2 * x_val \
    + f3 / (2.0 * x_val - l_bound - u_bound)\
    * ( 2.0 * (x/b)**(p1) * float(p1) * (np.exp(- (x/b)**(p1))) )
