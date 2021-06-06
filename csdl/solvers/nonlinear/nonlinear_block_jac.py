"""Define the NonlinearBlockJac class."""
from csdl.solvers.nonlinear_solver import NonlinearSolver


class NonlinearBlockJac(NonlinearSolver):
    """
    Nonlinear block Jacobi solver.
    """

    SOLVER = 'NL: NLBJ'
