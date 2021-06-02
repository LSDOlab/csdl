"""LinearSolver that uses linalg.solve or LU factor/solve."""

from csdl.solvers.linear_solver import LinearSolver


class DirectSolver(LinearSolver):
    """
    LinearSolver that uses linalg.solve or LU factor/solve.
    """

    SOLVER = 'LN: Direct'

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare(
            'err_on_singular',
            types=bool,
            default=True,
            desc="Raise an error if LU decomposition is singular.")

        # this solver does not iterate
        self.options.undeclare("maxiter")
        self.options.undeclare("err_on_non_converge")

        self.options.undeclare("atol")
        self.options.undeclare("rtol")

        # Use an assembled jacobian by default.
        self.options['assemble_jac'] = True
