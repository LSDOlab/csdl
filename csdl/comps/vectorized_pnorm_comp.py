import numpy as np
from openmdao.api import ExplicitComponent


class VectorizedPnormComp(ExplicitComponent):
    """
    This is a component that computes the p-norm of a vectorized tensor.
    This is exclusively for p-norms that are greater than 0 and even.
    The output is a scalar.

    Options
    -------
    in_name: str
        Name of the input

    out_name: str
        Name of the output

    shape: tuple[int]
        Shape of the input

    pnorm_type: int
        An even integer denoting the p-norm

    """
    def initialize(self):
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)
        self.options.declare('shape', types=tuple)
        self.options.declare('pnorm_type', types=int, default=2)
        self.options.declare('val', types=np.ndarray)

    def setup(self):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']
        pnorm_type = self.options['pnorm_type']
        val = self.options['val']

        self.add_input(in_name, shape=shape, val=val)
        self.add_output(out_name)

        self.declare_partials(out_name, in_name)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        pnorm_type = self.options['pnorm_type']

        self.outputs = outputs[out_name] = np.linalg.norm(
            inputs[in_name].flatten(), ord=pnorm_type)

    def compute_partials(self, inputs, partials):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        pnorm_type = self.options['pnorm_type']

        partials[out_name, in_name] = self.outputs**(
            1 - pnorm_type) * inputs[in_name]**(pnorm_type - 1)


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp, Group
    n = 2
    m = 3
    p = 10
    shape = (n, m, p)
    val = np.random.rand(n, m, p)
    indeps = IndepVarComp()
    indeps.add_output(
        'x',
        val=val,
        shape=shape,
    )
    prob = Problem()
    prob.model = Group()
    prob.model.add_subsystem(
        'indeps',
        indeps,
        promotes=['*'],
    )
    prob.model.add_subsystem(
        'vectorized_pnorm',
        VectorizedPnormComp(in_name='x',
                            out_name='y',
                            shape=shape,
                            pnorm_type=2),
        promotes=['*'],
    )
    prob.setup()
    prob.check_partials(compact_print=True)
    prob.run_model()
