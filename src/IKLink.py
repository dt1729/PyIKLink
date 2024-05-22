import pinocchio as pin
import numpy as np
import copy

from abc import ABC
from sklearn.cluster import DBSCAN


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
    def __init__(self, target_frame):
        self.robot          = pin.Model()
        
        # Trajectory of form tuple(float,
        #                           pinocchio.SE3()
        #                         )
        
        self.target_frame   = target_frame
        self.trajectory     = []
        self.table          = [[]]  # list(list(Node))

    def sample_candidates(self, target_transform, init_node = None):
        """Samples candidate ik solutions that solve the inverse kinematic problem for the given robot.

        Args:
            target_transform (pinocchio.SE3): Target pose to be reached
            init_node (pinocchio.SE3): Initial pose of the robot from where it has to reach
        """

        n = len(self.trajectory)
        tmp_ik_table = [[]]

        for i in range(n):
            print("Constructing nodes for point {} / {}", i, n)
            # clustering IK solutions using DBSCAN
            tmp_iks = DBSCAN(tmp_ik_table[i])
            # let clusters = Dbscan::params(2).tolerance(0.01).transform(&tmp_iks).unwrap();
            # Python => clusters = 
            assert(np.shape(clusters) == len(tmp_ik_table[i]))
            # labels = vec![false; clusters.shape()[0]];

            for j in range(np.shape(clusters)[0]):
                # Convert to python code, this checks if cluster idx exists,
                # based on that copies the generated ik table into the Node.

                if labels[clusters[j]] is not None:
                    labels[clusters[j]] = True
                    node = Node().new(copy.deepcopy(tmp_ik_table[i][j]))
                    self.table[i].append(node)
                else:
                    node = Node().new(copy.deepcopy(tmp_ik_table[i][j]))
                    self.table[i].append(node)


            while len(self.table[i]) < 200:
                ik = self.robot.try_to_reach()
                if ik is not None:
                    continue

                node = Node().new(ik)
                self.table[i].append(node)

            if i < n-1:
                for j in range(len(self.table[i])):
                    # self.robot.ik_solver.reset(self.table[i][j].ik.to_vec());
                    found_ik, ik =  self.ik.solve(
                                                    self.target_frame,
                                                    # TODO: Extract end effector frame that will track the position
                                                    self.trajectory[i+1][2],
                                                    init_state=self.trajectory[i+1][2],
                                                    nullspace_components=[]
                                                )
                    if found_ik:
                        break
                    tmp_ik_table[i+1].append(ik)

    def dp(self):
        """Implementation of dynamic programming algorithm from IKLink paper to generate the least configuration path from all
        the existing 
        """
        
        assert(len(self.trajectory) == len(self.table))
        print("Running Dynamic Programming algorithm")

        n = len(self.trajectory)

        for i in self.table[0]:
            i.primary_score = 0.0
            i.secondary_score = 0.0
            i.predecessor = 0

        for x in range(1, n):
            delta_t = self.trajectory[x][0] - self.trajectory[x-1][0]

            min_primary_score_with_config = 100000.0
            min_secondary_score_with_config = 100000.0
            min_idx_with_config = 0

            for y2 in range(len(self.table[x-1])):
                primary_score = self.table[x-1][y2].primary_score + 1.0
                secondary_score = self.table[x-1][y2].secondary_score
                if primary_score < min_primary_score_with_config or (primary_score == min_primary_score_with_config and secondary_score < min_secondary_score_with_config):
                    min_primary_score_with_config = primary_score
                    min_secondary_score_with_config = secondary_score
                    min_idx_with_config = y2
        
        # find best predecessor with no arm reconfiguration
        for y1 in range(len(self.table[x])):
            min_primary_score = min_primary_score_with_config
            min_secondary_score = min_secondary_score_with_config
            predecessor = min_idx_with_config
            
            for y in range(len(self.robot.check_velocity ... )): # TODO: Complete later with robot coming from pinocchio and not custom defined as done in rust
                primary_score = self.table[x-1][y2].primary_score
                secondary_score = self.table[x-1][y2].secondary_score + self.robot.joint_movement(self.table[x][y1].ik, self.table[x-1][y2].ik) # TODO
                if primary_score < min_primary_score:
                    min_primary_score = primary_score
                    min_secondary_score = secondary_score
                    predecessor = y2
                
            self.table[x][y1].primary_score = min_primary_score
            self.table[x][y1].secondary_score = min_secondary_score
            self.table[x][y1].predecessor = predecessor
        
        best_primary_score = 100000.0
        best_secondary_score = 100000.0
        best_idx = 0
        
        for j in range(len(self.table[n-1])):
            primary_score = self.table[n-1][j].primary_score
            secondary_score = self.table[n-1][j].secondary_score
            
            if primary_score < best_primary_score or (primary_score == best_primary_score and secondary_score < best_secondary_score):
                best_primary_score = primary_score
                best_secondary_score = secondary_score
                best_idx = j
            
        assert(best_primary_score < 100000.0, "No valid solution found!")
        print(f"Min Num of Reconfig: {best_primary_score}")
        
        idx = best_idx
        ans_traj = []
        
        for i in range(n, 0, -1):
            ans_traj.append((self.trajectory[i][0], copy.deepcopy(self.table[i][idx].ik)))
            idx = self.table[i][idx].predecessor
        
        ans_traj.reverse()
        return ans_traj

                