import numpy as np
from openmdao.api import ExplicitComponent


class VectorizedAxisWisePnormComp(ExplicitComponent):
    """
    This is a component that computes the axis-wise p-norm of a tensor.
    This is exclusively for p-norms that are greater than 0 and even.
    The output is a tensor.

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

    axis: tuple[int]
        Represents the axis along which the p-norm is computed

    out_shape: tuple[int]
        Shape of the output after the p-norm has been taken around the axis
    """
    def initialize(self):
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)
        self.options.declare('shape', types=tuple)
        self.options.declare('pnorm_type', types=int, default=2)
        self.options.declare('axis', types=tuple)
        self.options.declare('out_shape', default=None, types=tuple)
        self.options.declare('val', types=np.ndarray)

    def setup(self):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']
        pnorm_type = self.options['pnorm_type']
        axis = self.options['axis']
        out_shape = self.options['out_shape']
        val = self.options['val']

        self.add_input(in_name, shape=shape, val=val)

        # Computation of the einsum string that will be used in partials
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        rank = len(shape)
        input_subscripts = alphabet[:rank]
        output_subscripts = np.delete(list(input_subscripts), axis)
        output_subscripts = ''.join(output_subscripts)

        self.operation = '{},{}->{}'.format(
            output_subscripts,
            input_subscripts,
            input_subscripts,
        )

        # Computation of Output shape if the shape is not provided
        if out_shape == None:
            output_shape = np.delete(shape, axis)
            self.output_shape = tuple(output_shape)
        else:
            self.output_shape = out_shape

        self.add_output(out_name, shape=self.output_shape)

        # Defining the rows and columns of the sparse partial matrix
        input_size = np.prod(shape)
        cols = np.arange(input_size)
        rows = np.unravel_index(np.arange(input_size), shape=shape)
        rows = np.delete(np.array(rows), axis, axis=0)
        rows = np.ravel_multi_index(rows, dims=self.output_shape)

        self.declare_partials(out_name, in_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        pnorm_type = self.options['pnorm_type']
        axis = self.options['axis']

        self.outputs = outputs[out_name] = np.sum(inputs[in_name]**pnorm_type,
                                                  axis=axis)**(1 / pnorm_type)

    def compute_partials(self, inputs, partials):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        pnorm_type = self.options['pnorm_type']
        axis = self.options['axis']

        partials[out_name, in_name] = np.einsum(
            self.operation, self.outputs**(1 - pnorm_type),
            inputs[in_name]**(pnorm_type - 1)).flatten()


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp, Group
    n = 2
    m = 3
    p = 4
    k = 5
    shape = (n, m, p, k)
    axis = (0, 2)

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
        'vectorized_pnorm',
        VectorizedAxisWisePnormComp(in_name='x',
                                    out_name='y',
                                    axis=axis,
                                    shape=shape,
                                    pnorm_type=8),
        promotes=['*'],
    )
    prob.setup()
    prob.check_partials(compact_print=True)
    prob.run_model()
