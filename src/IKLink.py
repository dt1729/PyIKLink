import pinocchio as pin
import numpy as np


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
    def __init__(self):
        self.robot      = pin.Model()
        # Trajectory of form list(float,
        #                         np.array[3],
        #                         pin.Quaternion(np.random.rand(4,1)).normalized()
        #                         )
        self.trajectory = []
        self.table      = [[]]  # list(list(Node))

    def sample_candidates(self):
        """Samples candidates from the IK algorithm 
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
                
                # match clusters[j] {
                #     Some(cluster_idx) => {
                #         if !labels[cluster_idx] {
                #             labels[cluster_idx] = true;
                #             let node = Node::new(tmp_ik_table[i][j].clone());
                #             self.table[i].push(node);
                #         }
                #     },
                #     None => {
                #         let node = Node::new(tmp_ik_table[i][j].clone());
                #         self.table[i].push(node);
                #     }
                # }
                raise NotImplementedError

            while(len(self.table[i]) < 200):
                ik = self.robot.try_to_reach(self.trajectory[i].x, self.trajectory[i])
                if ik is not None:
                    continue

                node = Node().new(ik)
                self.table[i].append(node)

            if(i < n-1):
                for j in range(len(self.table[i])):
                    # self.robot.ik_solver.reset(self.table[i][j].ik.to_vec());
                    # let (found_ik, ik) = self.robot.try_to_track(self.trajectory[i+1].1, self.trajectory[i+1].2);
                    # if !found_ik {
                    #     break;
                    # }
                    # tmp_ik_table[i+1].push(Array1::from(ik));
                    raise NotImplementedError