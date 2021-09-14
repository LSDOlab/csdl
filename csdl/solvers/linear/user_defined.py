"""Define the LinearUserDefined class."""

from csdl.solvers.linear_solver import LinearSolver


class LinearUserDefined(LinearSolver):
    """
    LinearUserDefined solver.

    This is a solver that wraps a user-written linear solve function.

    Attributes
    ----------
    solve_function : function
        Custom function containing the solve_linear function. The default is None, which means
        the name defaults to "solve_linear".
    """

    SOLVER = 'LN: USER'

    def __init__(self, solve_function=None, **kwargs):
        """
        Initialize all attributes.

        **Parameters**

        solve_function : function
            Custom function containing the solve_linear function. The default is None, which means
            the name defaults to "solve_linear".
        **kwargs : dict
            Options dictionary.
        """
        super().__init__(**kwargs)

        self.solve_function = solve_function
