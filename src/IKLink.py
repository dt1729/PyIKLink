from abc import ABC
import pinocchio as pin
import numpy as np

class Node(ABC):
    """Creates a node for the specific inverse kinematic solution
        given by relaxedIK solver and forms a node for the graph

    Args:
        ABC (_type_): _description_
    """
    def __init__(self) -> None:
        self.ik             = np.array # Array of float64 values
        self.primary_score  = float
        self.secondary_score = float
        self.predecessor    = int

    def new(self, ik: np.array):
        """Creates a new node for the specific inverse kinematic solution
        given by relaxedIK solver and forms a node for the graph

        Args:
            ik (list): IK solution numpy vector

        Returns:
            Node: Returns the instance of a node.
        """
        self.primary_score      = 100000.0
        self.secondary_score    = 100000.0
        self.predecessor        = 0
        self.ik                 = ik
        return self


class IKLink(ABC):
    """Class to implement IKLink algorithm for minimal 
    reconfiguration based Inverse Kinematics solutions.

    Args:
        ABC (abstract base class): _description_
    """
    def __init__(self):
        self.robot      = pin.Model()
        # Trajectory of form list(float,
        #                         np.array[3],
        #                         pin.Quaternion(np.random.rand(4,1)).normalized()
        #                         )
        self.trajectory = []
        self.table      = [[]]
        