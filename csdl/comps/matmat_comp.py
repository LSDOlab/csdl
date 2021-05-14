import numpy as np
from openmdao.api import ExplicitComponent


class MatMatComp(ExplicitComponent):
    '''
    This is a component that computes the matrix multiplication between two matrices using @

    Options
    -------
    in_name: list[str]
        Name of the input

    out_name: str
        Name of the output

    in_shapes: list[tuples]
        A list of tuples that represents the size of each input

    out_shape: tuple
        A tuple that is the size of the output after the matrix multiplication
    '''
    def initialize(self):
        self.options.declare('in_names', types=list)
        self.options.declare('out_name', types=str)
        self.options.declare('in_shapes', types=list)
        self.options.declare('in_vals', types=list)

    def setup(self):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        in_shapes = self.options['in_shapes']
        in_vals = self.options['in_vals']

        self.add_input(in_names[0], shape=in_shapes[0], val=in_vals[0])
        self.add_input(in_names[1], shape=in_shapes[1], val=in_vals[1])

        if in_shapes[1][1] != 1:
            output_shape = (in_shapes[0][0], in_shapes[1][1])

        else:
            output_shape = (in_shapes[0][0], )

        self.add_output(
            out_name,
            shape=output_shape,
        )

        output_size = (np.prod(output_shape))
        input1_size = np.prod(in_shapes[0])
        input2_size = np.prod(in_shapes[1])

        r0 = np.repeat(np.arange(output_size), in_shapes[0][1])
        c0 = np.tile(
            np.arange(input1_size).reshape(in_shapes[0]),
            in_shapes[1][1]).flatten()

        self.declare_partials(out_name, in_names[0], rows=r0, cols=c0)

        r1 = np.repeat(np.arange(output_size), in_shapes[0][1])
        c1 = np.tile(
            np.transpose(np.arange(input2_size).reshape(
                in_shapes[1])).flatten(), in_shapes[0][0])

        self.declare_partials(out_name, in_names[1], rows=r1, cols=c1)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']

        outputs[out_name] = inputs[in_names[0]] @ inputs[in_names[1]]

    def compute_partials(self, inputs, partials):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        in_shapes = self.options['in_shapes']

        partials[out_name, in_names[0]] = np.tile(
            np.transpose(inputs[in_names[1]]).flatten(), in_shapes[0][0])

        partials[out_name, in_names[1]] = np.tile(inputs[in_names[0]],
                                                  in_shapes[1][1]).flatten()


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp, Group
    m = 3
    n = 2
    p = 4

    shape1 = (m, n)
    shape2 = (n, p)

    mat1 = np.arange(m * n).reshape(shape1)
    mat2 = np.arange(n * p).reshape(shape2)

    indeps = IndepVarComp()
    indeps.add_output(
        'mat1',
        val=mat1,
        shape=shape1,
    )

    indeps.add_output(
        'mat2',
        val=mat2,
        shape=shape2,
    )

    prob = Problem()
    prob.model = Group()
    prob.model.add_subsystem(
        'indeps',
        indeps,
        promotes=['*'],
    )
    prob.model.add_subsystem('matmat',
                             MatMatComp(
                                 in_names=['mat1', 'mat2'],
                                 out_name='y',
                                 in_shapes=[shape1, shape2],
                             ),
                             promotes=['*'])

    prob.setup()
    prob.check_partials(compact_print=True)
    prob.run_model()
