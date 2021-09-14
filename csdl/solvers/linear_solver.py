import os

from csdl.solvers.solver import Solver


class LinearSolver(Solver):
    """
    Base class for linear solvers.

    Attributes
    ----------
    _rel_systems : set of str
        Names of systems relevant to the current solve.
    _assembled_jac : AssembledJacobian or None
        If not None, the AssembledJacobian instance used by this solver.
    """
    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        **Parameters**

        **kwargs : dict
            options dictionary.
        """
        self._rel_systems = None
        self._assembled_jac = None
        super().__init__(**kwargs)

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare(
            'assemble_jac',
            default=False,
            types=bool,
            desc='Activates use of assembled jacobian by this solver.')

        self.supports.declare('assembled_jac', types=bool, default=True)
