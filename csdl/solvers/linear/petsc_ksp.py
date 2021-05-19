"""LinearSolver that uses PetSC KSP to solve for a system's derivatives."""

from csdl.solvers.linear_solver import LinearSolver

KSP_TYPES = [
    "richardson", "chebyshev", "cg", "groppcg", "pipecg", "pipecgrr", "cgne",
    "nash", "stcg", "gltr", "fcg", "pipefcg", "gmres", "pipefgmres", "fgmres",
    "lgmres", "dgmres", "pgmres", "tcqmr", "bcgs", "ibcgs", "fbcgs", "fbcgsr",
    "bcgsl", "cgs", "tfqmr", "cr", "pipecr", "lsqr", "preonly", "qcg", "bicg",
    "minres", "symmlq", "lcd", "python", "gcr", "pipegcr", "tsirm", "cgls"
]


class PETScKrylov(LinearSolver):
    """
    LinearSolver that uses PetSC KSP to solve for a system's derivatives.
    """

    SOLVER = 'LN: PETScKrylov'

    def __init__(self, **kwargs):
        """
        Declare the solver options.

        Parameters
        ----------
        **kwargs : dict
            dictionary of options set by the instantiating class/script.
        """
        super().__init__(**kwargs)

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('ksp_type',
                             default='fgmres',
                             values=KSP_TYPES,
                             desc="KSP algorithm to use. Default is 'fgmres'.")

        self.options.declare(
            'restart',
            default=1000,
            types=int,
            desc='Number of iterations between restarts. Larger values increase '
            'iteration cost, but may be necessary for convergence')

        self.options.declare('precon_side',
                             default='right',
                             values=['left', 'right'],
                             desc='Preconditioner side, default is right.')

        # changing the default maxiter from the base class
        self.options['maxiter'] = 100
