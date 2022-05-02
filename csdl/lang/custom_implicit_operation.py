from csdl.lang.custom_operation import CustomOperation


class CustomImplicitOperation(CustomOperation):
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

        *Example*

        .. code-block:: python

            def evaluate_residuals(self, inputs, outputs, residuals):
                x = outputs['x']
                a = inputs['a']
                b = inputs['b']
                c = inputs['c']
                residuals['x'] = a * x**2 + b * x + c
        """
        pass

    def compute_derivatives(self, inputs, outputs, derivatives):
        """
        [Optional] User defined method to evaluate exact derivatives of
        residuals wrt inputs and outputs

        *Example*

        .. code-block:: python

            def compute_derivatives(self, inputs, outputs, derivatives):
                a = inputs['a']
                b = inputs['b']
                x = outputs['x']

                derivatives['x', 'a'] = x**2
                derivatives['x', 'b'] = x
                derivatives['x', 'c'] = 1.0
                derivatives['x', 'x'] = 2 * a * x + b

                # only necessary if implementing `apply_inverse_jacobian`
                self.inv_jac = 1.0 / (2 * a * x + b)
        """
        pass

    def solve_residual_equations(self, inputs, outputs):
        """
        [Optional] User defined method to solve residual equations,
        computing the outputs given the inputs. Define this method to
        implement a custom solver. Assigning a nonlinear solver will
        cause `evaluate_residual_equations` to run instead.

        *Example*

        .. code-block:: python

            def solve_residual_equations(self, inputs, outputs):
                a = inputs['a']
                b = inputs['b']
                c = inputs['c']
                outputs['x'] = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        """
        pass

    def apply_inverse_jacobian(
        self,
        d_outputs,
        d_residuals,
        mode,
    ):
        """
        [Optional] Solve linear system. Invoked when solving coupled
        linear system; i.e. when solving Newton system to update
        implicit state variables, and when computing total derivatives

        *Example*

        .. code-block:: python

            # using self.inv_jac defined in `compute_derivatives` example
            def apply_inverse_jacobian( self, d_outputs, d_residuals, mode)
                if mode == 'fwd':
                    d_outputs['x'] = self.inv_jac * d_residuals['x']
                elif mode == 'rev':
                    d_residuals['x'] = self.inv_jac * d_outputs['x']
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
        [Optional] Implement partial derivatives by computing a
        matrix-vector product.

        *Example*

        .. code-block:: python

            def compute_jacvec_product(
                    self,
                    inputs,
                    outputs,
                    d_inputs,
                    d_outputs,
                    d_residuals,
                    mode,
                ):
                    a = inputs['a']
                    b = inputs['b']
                    c = inputs['c']
                    x = outputs['x']
                    if mode == 'fwd':
                        if 'x' in d_residuals:
                            if 'x' in d_outputs:
                                d_residuals['x'] += (2 * a * x + b) * d_outputs['x']
                            if 'a' in d_inputs:
                                d_residuals['x'] += x ** 2 * d_inputs['a']
                            if 'b' in d_inputs:
                                d_residuals['x'] += x * d_inputs['b']
                            if 'c' in d_inputs:
                                d_residuals['x'] += d_inputs['c']
                    elif mode == 'rev':
                        if 'x' in d_residuals:
                            if 'x' in d_outputs:
                                d_outputs['x'] += (2 * a * x + b) * d_residuals['x']
                            if 'a' in d_inputs:
                                d_inputs['a'] += x ** 2 * d_residuals['x']
                            if 'b' in d_inputs:
                                d_inputs['b'] += x * d_residuals['x']
                            if 'c' in d_inputs:
                                d_inputs['c'] += d_residuals['x']
        """
        pass
