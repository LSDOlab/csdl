import numpy as np

from csdl.solvers.linear_solver import LinearSolver


class BlockLinearSolver(LinearSolver):
    """
    A base class for LinearBlockGS and LinearBlockJac.
    """
    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        self.supports['assembled_jac'] = False
