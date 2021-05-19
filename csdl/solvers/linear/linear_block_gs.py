"""Define the LinearBlockGS class."""

from csdl.solvers.block_linear_solver import BlockLinearSolver


class LinearBlockGS(BlockLinearSolver):
    """
    Linear block Gauss-Seidel solver.

    Attributes
    ----------
    _delta_d_n_1 : dict of ndarray
        Cached change in the d_output vectors for the previous iteration. Only used if the
        aitken acceleration option is turned on. The dictionary is keyed by linear vector name.
    _theta_n_1 : dict of float
        Cached relaxation factor from previous iteration. Only used if the aitken acceleration
        option is turned on. The dictionary is keyed by linear vector name.
    """

    SOLVER = 'LN: LNBGS'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super().__init__(**kwargs)

        self._theta_n_1 = {}
        self._delta_d_n_1 = {}

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('use_aitken',
                             types=bool,
                             default=False,
                             desc='set to True to use Aitken relaxation')
        self.options.declare('aitken_min_factor',
                             default=0.1,
                             desc='lower limit for Aitken relaxation factor')
        self.options.declare('aitken_max_factor',
                             default=1.5,
                             desc='upper limit for Aitken relaxation factor')
        self.options.declare('aitken_initial_factor',
                             default=1.0,
                             desc='initial value for Aitken relaxation factor')
