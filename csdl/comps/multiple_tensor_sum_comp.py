import numpy as np
from copy import deepcopy
from openmdao.api import ExplicitComponent


class MultipleTensorSumComp(ExplicitComponent):
    """
    This component can output multiple variants of summation of elements of arrays:
    Option 1: If the input contains multiple tensors all of which have the same dimension, without any axis provided, the component sums all the tensors elementwise. Output is a tensor with the same dimension as the input.
    Option 2: If the input contains multiple tensors all of which have the same dimension, with axis/axes specified, the component sums all the elements of each matrix along the specified axis/axes, and then sums all the resulting vectors elementwise. Output is a tensor whose shape correspond to the shape of the input tensors with the size/s of the specified axis/axes removed.

    Options
    -------
    in_names: list
        Input component names that represent input arrays.
    shape: tuple
        Shape of the input arrays.
    axis: tuple
        Axis/axes along which the sum is computed when the inputs are matrices.
    out_name: str
        Output component name that represents the summed output tensor.
    """
    def initialize(self):
        self.options.declare('in_names',
                             default=None,
                             types=list,
                             allow_none=True)
        self.options.declare('out_name', types=str)
        self.options.declare('shape', types=tuple)
        self.options.declare('axes', default=None, types=tuple)
        self.options.declare('out_shape', default=None, types=tuple)
        self.options.declare('vals', types=list)

    def setup(self):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        shape = self.options['shape']
        out_shape = self.options['out_shape']
        axes = self.options['axes']
        vals = self.options['vals']

        # Computation of Output shape if the shape is not provided
        if out_shape == None and axes != None:
            output_shape = np.delete(shape, axes)
            self.output_shape = tuple(output_shape)

        else:
            self.output_shape = out_shape

        self.num_inputs = len(in_names)
        input_size = np.prod(shape)
        val = np.ones(input_size)

        # axes not specified => elementwise sum
        if axes == None:
            self.add_output(out_name, shape=shape)
            for in_name, in_val in zip(in_names, vals):
                self.add_input(in_name, shape=shape, val=in_val)
                rows = cols = np.arange(input_size)
                self.declare_partials(out_name,
                                      in_name,
                                      rows=rows,
                                      cols=cols,
                                      val=val)

        # axes specified => axiswise sum
        else:
            self.add_output(out_name, shape=self.output_shape)
            cols = np.arange(input_size)

            rows = np.unravel_index(np.arange(input_size), shape=shape)
            rows = np.delete(np.array(rows), axes, axis=0)
            rows = np.ravel_multi_index(rows, dims=self.output_shape)

            for in_name, in_val in zip(in_names, vals):
                self.add_input(in_name, shape=shape, val=in_val)
                self.declare_partials(out_name,
                                      in_name,
                                      rows=rows,
                                      cols=cols,
                                      val=val)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        axes = self.options['axes']

        # axes == None does the elementwise sum of the tensors
        if axes == None:
            outputs[out_name] = inputs[in_names[0]]
            for i in range(1, self.num_inputs):
                outputs[out_name] += inputs[in_names[i]]

        # axes != None takes the sum along specified axes
        else:
            outputs[out_name] = np.sum(inputs[in_names[0]], axis=axes)
            for i in range(1, self.num_inputs):
                outputs[out_name] += np.sum(inputs[in_names[i]], axis=axes)


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp, Group
    n = 20
    m = 1
    val1 = np.arange(n * m).reshape(n, m) + 1
    val2 = np.random.rand(n, m)

    indeps = IndepVarComp()
    indeps.add_output(
        'x',
        val=val1,
        shape=(n, m),
    )

    indeps.add_output(
        'y',
        val=val2,
        shape=(n, m),
    )
    prob = Problem()
    prob.model = Group()
    prob.model.add_subsystem(
        'indeps',
        indeps,
        promotes=['*'],
    )
    prob.model.add_subsystem(
        'sum',
        MultipleTensorSumComp(in_names=['x', 'y'],
                              out_name='f',
                              shape=(n, m),
                              axes=(0, )),
        promotes=['*'],
    )
    prob.setup()
    prob.check_partials(compact_print=True)
    prob.run_model()
