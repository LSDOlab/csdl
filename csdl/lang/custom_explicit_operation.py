from csdl.lang.custom_operation import CustomOperation


class CustomExplicitOperation(CustomOperation):
    def compute(self, inputs, outputs):
        """
        Define outputs as an explicit function of the inputs

        **Example**

        ```py
        def compute(self, inputs, outputs):
            outputs['L'] = 1/2 * inputs['Cl'] * inputs['rho'] * inputs['V']**2 * inputs['S']
            outputs['D'] = 1/2 * inputs['Cd'] * inputs['rho'] * inputs['V']**2 * inputs['S']
        ```
        """
        pass

    def compute_derivatives(self, inputs, derivatives):
        """
        User defined method to compute partial derivatives for this
        operation

        **Example**

        ```py
        def compute(self, inputs, outputs):
            outputs['L'] = 1/2 * inputs['Cl'] * inputs['rho'] * inputs['V']**2 * inputs['S']
            outputs['D'] = 1/2 * inputs['Cd'] * inputs['rho'] * inputs['V']**2 * inputs['S']

        def compute_derivatives(self, inputs, derivatives):
            derivatives['L', 'Cl'] = 1/2 * inputs['rho'] * inputs['V']**2 * inputs['S']
            derivatives['L', 'rho'] = 1/2 * inputs['Cl'] * inputs['V']**2 * inputs['S']
            derivatives['L', 'V'] = inputs['Cl'] * inputs['rho'] * inputs['V'] * inputs['S']
            derivatives['L', 'S'] = 1/2 * inputs['Cl'] * inputs['rho'] * inputs['V']**2

            derivatives['D', 'Cd'] = 1/2 * inputs['rho'] * inputs['V']**2 * inputs['S']
            derivatives['D', 'rho'] = 1/2 * inputs['Cd'] * inputs['V']**2 * inputs['S']
            derivatives['D', 'V'] = inputs['Cd'] * inputs['rho'] * inputs['V'] * inputs['S']
            derivatives['D', 'S'] = 1/2 * inputs['Cd'] * inputs['rho'] * inputs['V']**2
        ```
        """
        pass

    def compute_jacvec_product(
        self,
        inputs,
        d_inputs,
        d_outputs,
        mode,
    ):
        """
        [Optional] Implement partial derivatives by computing a
        matrix-vector product

        *Example*

        ```py
        def compute(self, inputs, outputs):
            outputs['area'] = inputs['length'] * inputs['width']

        def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
            if mode == 'fwd':
                if 'area' in d_outputs:
                    if 'length' in d_inputs:
                        d_outputs['area'] += inputs['width'] * d_inputs['length']
                    if 'width' in d_inputs:
                        d_outputs['area'] += inputs['length'] * d_inputs['width']
            elif mode == 'rev':
                if 'area' in d_outputs:
                    if 'length' in d_inputs:
                        d_inputs['length'] += inputs['width'] * d_outputs['area']
                    if 'width' in d_inputs:
                        d_inputs['width'] += inputs['length'] * d_outputs['area']
        ```
        """
        pass
