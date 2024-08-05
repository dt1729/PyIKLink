#!python3

import casadi as cs
from alpaqa import minimize

# Build the problem (CasADi code, independent of alpaqa)
    # Make symbolic decision variables
    # Collect decision variables into one vector
    # Make a parameter symbol

# Objective function f and the constraints function g
    # Define the bounds

# Generate and compile C code for the objective and constraints using alpaqa
    #    Objective function f(x)
    #    Box constraints x ∊ C
    #    General ALM constraints g(x) ∊ D
    #    Parameter with default value (can be changed later)

# You can change the bounds and parameters after loading the problem

# Build a solver with the default parameters

# Build a solver with custom parameters

# Build a solver with alternative fast directions

# Compute a solution
    # Set initial guesses at arbitrary values
    # decision variables
    # Lagrange multipliers for g(x)
    # Solve the problem
