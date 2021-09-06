from csdl.core.custom_operation import CustomOperation


class ImplicitOperation(CustomOperation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_solver = None
        self.nonlinear_solver = None

    def evaluate_residuals(
        self,
        inputs,
        outputs,
        residuals,
    ):
        """
        User defined method to evaluate residuals
        """
        pass

    def compute_derivatives(self, inputs, outputs, derivatives):
        """
        User defined method to evaluate derivatives of residuals wrt
        inputs and outputs
        """
        pass

    def solve_residual_equations(self, inputs, outputs):
        """
        User defined method to solve residual equations, converging
        residuals to compute outputs.
        """
        pass

    def apply_inverse_jacobian(
        self,
        d_outputs,
        d_residuals,
        mode,
    ):
        """
        Optional. Solve linear system. Invoked when solving coupled
        linear system; i.e. when solving Newton system to update
        implicit state variables, and when computing total derivatives
        """
        pass

    def compute_jacvec_product(
        self,
        inputs,
        outputs,
        d_inputs,
        d_outputs,
        d_residuals,
        mode,
    ):
        """
        Optional.
        """
        pass
