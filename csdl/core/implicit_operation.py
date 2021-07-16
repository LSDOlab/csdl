from csdl.core.custom_operation import CustomOperation


class ImplicitOperation(CustomOperation):
    def evaluate_residuals(
        self,
        inputs,
        outputs,
        residuals,
        discrete_inputs=None,
        discrete_outputs=None,
    ):
        """
        User defined method to evaluate residuals
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
        pass
