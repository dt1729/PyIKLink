import numpy as np
import copy

class forward_diff:
    """forward difference calculator.
    """
    def __init__(self) -> None:
        self.dim = int
        self.f   = function
        self.h   = 0.00001
        self.out_grad = np.zeros(self.dim)
        self.__x_h = np.zeros(self.dim)

    def compute_gradient(self, x : np.array):
        """Computes gradient
        """
        val_0 = self.f(x)
        for (i, _) in range(x):
            self.reset(x)
            self.__x_h[i] += self.h
            self.out_grad[i] = (-1*val_0 + self.f(self.__x_h)) / self.h

    def compute_and_return_grad(self, x : np.array) -> np.array:
        """Computes and returns the gradient using forward difference method

        Args:
            x (np.array): Input position where the gradient is to be computed

        Returns:
            np.array: gradient vector
        """

        self.compute_gradient(x)
        return copy.copy(self.out_grad)

    def reset(self, x: np.array):
        """_summary_

        Args:
            x (np.array): _description_
        """
        self.__x_h = x

