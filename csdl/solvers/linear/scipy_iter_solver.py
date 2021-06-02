"""Define the scipy iterative solver class."""

from scipy.sparse.linalg import gmres

from csdl.solvers.linear_solver import LinearSolver

_SOLVER_TYPES = {
    # 'bicg': bicg,
    # 'bicgstab': bicgstab,
    # 'cg': cg,
    # 'cgs': cgs,
    'gmres': gmres,
}


class ScipyKrylov(LinearSolver):
    """
    The Krylov iterative solvers in scipy.sparse.linalg.

    Attributes
    ----------
    precon : Solver
        Preconditioner for linear solve. Default is None for no preconditioner.
    """

    SOLVER = 'LN: SCIPY'

    def __init__(self, **kwargs):
        """
        Declare the solver option.

        Parameters
        ----------
        **kwargs : {}
            dictionary of options set by the instantiating class/script.
        """
        super().__init__(**kwargs)

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('solver',
                             default='gmres',
                             values=tuple(_SOLVER_TYPES.keys()),
                             desc='function handle for actual solver')

        self.options.declare(
            'restart',
            default=20,
            types=int,
            desc='Number of iterations between restarts. Larger values increase '
            'iteration cost, but may be necessary for convergence. This '
            'option applies only to gmres.')

        # changing the default maxiter from the base class
        self.options['maxiter'] = 1000
        self.options['atol'] = 1.0e-12
