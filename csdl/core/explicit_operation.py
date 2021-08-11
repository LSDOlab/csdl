from csdl.core.custom_operation import CustomOperation


class ExplicitOperation(CustomOperation):
    def compute(self, inputs, outputs):
        """
        Define outputs as an explicit function of the inputs
        """
        pass

    def compute_derivatives(self, inputs, derivatives):
        """
        User defined method to compute partial derivatives for this
        operation
        """
        pass

    def compute_jacvec_product(
        self,
        inputs,
        d_inputs,
        d_outputs,
        mode,
    ):
        pass
