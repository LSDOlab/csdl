"""Define the base NonlinearSolver class."""

from collections import OrderedDict

from csdl.solvers.solver import Solver


class NonlinearSolver(Solver):
    """
    Base class for nonlinear solvers.

    Attributes
    ----------
    _err_cache : dict
        Dictionary holding input and output vectors at start of iteration, if requested.
    """
    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        **Parameters**

        **kwargs : dict
            options dictionary.
        """
        super().__init__(**kwargs)
        self._err_cache = OrderedDict()

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare(
            'debug_print',
            types=bool,
            default=False,
            desc='If true, the values of input and output variables at '
            'the start of iteration are printed and written to a file '
            'after a failure to converge.')
        self.options.declare(
            'stall_limit',
            default=0,
            desc=
            'Number of iterations after which, if the residual norms are '
            'identical within the stall_tol, then terminate as if max '
            'iterations were reached. Default is 0, which disables this '
            'feature.')
        self.options.declare(
            'stall_tol',
            default=1e-12,
            desc=
            'When stall checking is enabled, the threshold below which the '
            'residual norm is considered unchanged.')
