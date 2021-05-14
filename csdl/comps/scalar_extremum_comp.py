import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class ScalarExtremumComp(ExplicitComponent):
    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)

        self.options.declare('shape', types=tuple)

        self.options.declare('lower_flag', types=bool, default=False)
        self.options.declare('rho', 50.0, desc="Aggregation Factor.")
        self.options.declare('val', types=np.ndarray)

    def setup(self):
        """
        Declare inputs, outputs, and derivatives for the KS component.
        """
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']
        val = self.options['val']

        # Inputs
        self.add_input(in_name, shape=shape, val=val)

        # Outputs
        self.add_output(out_name, shape=(1, ))

        self.declare_partials(of=out_name, wrt=in_name)

    def compute(self, inputs, outputs):
        """
        Compute the output of the KS function.
        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']
        lower_flag = self.options['lower_flag']
        rho = self.options['rho']

        if lower_flag:
            g_max = np.max(-1.0 * inputs[in_name])
            g_diff = -1.0 * inputs[in_name] - g_max
        else:
            g_max = np.max(inputs[in_name])
            g_diff = inputs[in_name] - g_max

        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents)
        result = g_max + 1.0 / rho * np.log(summation)

        if lower_flag:
            outputs[out_name] = -result
        else:
            outputs[out_name] = result

        dsum_dg = rho * exponents
        dKS_dsum = 1.0 / (rho * summation * np.ones(shape))
        dKS_dg = dKS_dsum * dsum_dg

        self.dKS_dg = dKS_dg

        # if lower_flag:
        #     self.dKS_dg = -self.dKS_dg

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.
        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        partials[out_name, in_name] = self.dKS_dg.flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    shape = (2, 3, 4, 5, 6)
    # shape = (1,)

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', val=np.arange(np.prod(shape)).reshape(shape))
    prob.model.add_subsystem('ivc', comp, promotes=['*'])

    comp = ScalarExtremumComp(
        in_name='x',
        out_name='y',
        shape=shape,
        lower_flag=False,
        rho=100.,
    )
    prob.model.add_subsystem('ScalarExtremumComp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)

    # print(prob['x'], 'x')
    # print(prob['y'], 'y')
