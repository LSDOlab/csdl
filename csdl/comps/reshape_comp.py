import numpy as np
from openmdao.api import ExplicitComponent


class ReshapeComp(ExplicitComponent):
    '''
    This component reshapes the input to a new shape that is specified by the user.
    The input must be a numpy array.

    Options
    -------
    in_name: str
        Name of the input

    out_name: str
        Name of the output

    shape: tuple[int]
        Shape of the input

    new_shape: tuple[int]
        The desired shape of the output

    '''
    def initialize(self):
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)
        self.options.declare('shape', types=tuple)
        self.options.declare('new_shape', types=tuple)
        self.options.declare('val', types=np.ndarray)

    def setup(self):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']
        new_shape = self.options['new_shape']
        val = self.options['val']

        self.add_input(in_name, shape=shape, val=val)

        self.add_output(out_name, shape=new_shape)

        self.size = np.prod(shape)

        r = np.arange(self.size)

        val = np.ones((self.size, ))

        self.declare_partials(out_name, in_name, rows=r, cols=r, val=val)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']
        new_shape = self.options['new_shape']

        outputs[out_name] = inputs[in_name].reshape(new_shape)


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp, Group
    n = 2
    m = 3
    p = 4
    k = 5
    shape = (n, m, p, k)

    val = np.random.rand(n, m, p, k)
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
        'reshape_comp',
        ReshapeComp(in_name='x',
                    out_name='y',
                    shape=shape,
                    new_shape=(p * m, n * k)),
        promotes=['*'],
    )
    prob.setup()
    prob.check_partials(compact_print=True)
    prob.run_model()
