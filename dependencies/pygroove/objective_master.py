from vars import RelaxedIKVars
from groove import objective as obj
import copy
import numpy as np 

class ObjectiveMaster:
    def __init__(self) -> None:

        self.objectives = []
        self.num_chains = int
        self.weight_priors = []
        self.lite = bool
        self.finite_diff_grad = bool

    def standard_ik(self, chain_indices : list):
        self.objectives = [] # TODO: Check with your objective solver to see how it wants objectives and add them 
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

        self.num_dofs = len(chain_indices) + 1
        #self.objectives.append() # TODO: Add objective to minimize velocity Do it according to PanocPy
        self.weight_priors.append(0.01)

    def call(self, x : list, vars : RelaxedIKVars):

        if self.lite:
            return self.__call_lite(x, vars)
        else:
            return self.__call(x, vars)

    def gradient(self, x : list, vars : RelaxedIKVars):

        if self.lite:
            if self.finite_diff_grad:
                return self.__gradient_finite_diff_lite(x , vars)
            else:
                return self.__gradient_lite(x, vars)
        else:
            if self.finite_diff_grad:
                return self.__gradient_finite_diff(x, vars)
            else:
                return self.__gradient(x, vars)

    def gradient_finite_diff(self, x : list, vars : RelaxedIKVars):

        if self.lite:
            return self.__gradient_finite_diff_lite(x, vars)
        else:
            return self.__gradient_finite_diff(x, vars)

    def __call(self, x : list, vars : RelaxedIKVars):

        out = 0.0
        frames = vars.robot.get_frames_immutable(x)

        for i in enumerate(self.objectives):
            out += self.weight_priors[i] * self.objectives[i].call(x, vars, frames)

        return out

    def __call_lite(self, x : list, vars : RelaxedIKVars):

        out = 0.0
        poses = vars.robot.get_ee_pos_and_quat_immutable(x)

        for i in enumerate(self.objectives):
            out += self.weight_priors[i[0]] * self.objectives[i[0]].call(x, vars, poses)

        return out

    def __gradient(self, x : list, vars : RelaxedIKVars):

        grad = [0.0 for _ in range(len(x))]
        obj = 0.0

        finite_diff_list = []
        f_0s             = []

        robot_copy = copy.deepcopy(vars.robot)

        pin.forwardKinematics(robot_copy, robot_copy.createData(), np.array(x))
        frames_0 = robot_copy.frames.tolist()
        del robot_copy

        for i in enumerate(self.objectives):
            if self.objectives[i[0]].gradient_type() == 0:
                (local_obj, local_grad) = self.objectives[i[0]].gradient(x, vars, frames_0)
                f_0s.append(local_obj)
                obj += self.weight_priors[i] * local_obj
                for j in enumerate(local_grad):
                    grad[j] += self.weight_priors[i[0]] * local_grad[j[0]]
            elif self.objectives[i[0]].gradient_type() == 1:
                finite_diff_list.append(i[0])
                local_obj = self.objectives[i[0]].call(x, vars, frames_0)
                obj += self.weight_priors[i[0]]*local_obj
                f_0s.append(local_obj)

            if len(finite_diff_list) > 0:
                for i in enumerate(x):
                    x_h = copy.deepcopy(x)
                    x_h += 0.0000001
                    robot_copy = copy.deepcopy(vars.robot)

                    pin.forwardKinematics(robot_copy, robot_copy.createData(), np.array(x_h))
                    frames_h = robot_copy.frames.tolist()

                    del robot_copy 

                    for j in finite_diff_list:
                        f_h = self.objectives[j].call(x_h, vars, frames_h)
                        grad[i] += self.weight_priors[j] * ((-f_0s[j] + f_h) / 0.0000001)

        return obj, grad

    def __gradient_lite(self, x : list, vars : RelaxedIKVars):

        grad = [0.0 for _ in range(len(x))]
        obj = 0.0

        finite_diff_list = []
        f_0s             = []

        robot_copy = copy.deepcopy(vars.robot)
        pin.forwardKinematics(robot_copy, robot_copy.createData(), np.array(x))
        poses_0 = robot_copy.frames.tolist()[-1:-4]
        del robot_copy

        for i in enumerate(self.objectives):
            if self.objectives[i].gradient_type() == 1:
                local_obj, local_grad = self.objectives[i[0]].gradient_lite(x, vars, poses_0)
                f_0s.append(local_obj)
                obj += self.weight_priors[i[0]] * local_obj
                for j in enumerate(local_grad):
                    grad[j[0]] += self.weight_priors[i[0]] * local_grad[j[0]]
            elif self.objectives[i].gradient_type() == 0:
                finite_diff_list.append(i[0])
                local_obj = self.objectives[i[0]].call_lite(x, vars, poses_0)
                obj += self.weight_priors[i[0]] * local_obj
                f_0s.append(local_obj)

        if len(finite_diff_list) > 0:
            for i in enumerate(x):
                x_h = copy.deepcopy(x)
                x_h[i[0]] += 0.0000001

                robot_copy = copy.deepcopy(vars.robot)
                pin.forwardKinematics(robot_copy, robot_copy.createData(), np.array(x_h))
                poses_h = robot_copy.frames.tolist()[-1:-4]
                del robot_copy

                for j in finite_diff_list:
                    f_h = self.objectives[j].call_lite(x, vars, poses_h)
                    grad[i[0]] += self.weight_priors[j] * ((-f_0s[j] + f_h)/0.0000001)

        return obj, grad

    def __gradient_finite_diff(self, x : list, vars : RelaxedIKVars):
        grad = [0.0 for _ in range(len(x))]
        f_0  = self.call(x, vars)

        for i in enumerate(x):
            x_h = copy.deepcopy(x)
            x_h[i[0]] += 0.000001
            f_h = self.call(x_h, vars)
            grad[i[0]] = (-f_0 + f_h)/0.000001

        return f_0, grad

    def __gradient_finite_diff_lite(self, x : list, vars : RelaxedIKVars):
        grad = [0.0 for _ in range(len(x))]
        f_0  = self.call(x, vars)

        for i in enumerate(x):
            x_h = copy.deepcopy(x)
            x_h[i[0]] += 0.000001
            f_h = self.__call_lite(x_h, vars)
            grad[i[0]] = (-f_0 + f_h)/0.000001

        return f_0, grad
