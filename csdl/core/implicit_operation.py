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
        pass

    def solve_residual_equations(self, inputs, outputs):
        pass

    def apply_inverse_jacobian(
        self,
        inputs,
        outputs,
        d_inputs,
        d_outputs,
        d_residuals,
        mode,
    ):
        pass
