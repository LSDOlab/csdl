"""Define the LinearBlockJac class."""
from csdl.solvers.block_linear_solver import BlockLinearSolver


class LinearBlockJac(BlockLinearSolver):
    """
    Linear block Jacobi solver.
    """

    SOLVER = 'LN: LNBJ'
