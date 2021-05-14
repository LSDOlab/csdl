import numpy as np
from openmdao.api import ExplicitComponent


class MatVecComp(ExplicitComponent):
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

        self.add_output(
            out_name,
            shape=in_shapes[0][0],
        )

        # Dense
        self.declare_partials(out_name, in_names[1])

        c = np.arange(np.prod(in_shapes[0]))

        r = np.repeat(np.arange(in_shapes[0][0]), in_shapes[0][1])

        # Sparse
        self.declare_partials(out_name, in_names[0], cols=c, rows=r)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']

        outputs[out_name] = inputs[in_names[0]] @ inputs[in_names[1]]

    def compute_partials(self, inputs, partials):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        in_shapes = self.options['in_shapes']

        partials[out_name, in_names[0]] = np.tile(inputs[in_names[1]],
                                                  in_shapes[0][0])

        partials[out_name, in_names[1]] = inputs[in_names[0]]


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp, Group
    m = 3
    n = 5

    shape1 = (m, n)
    shape2 = (n, )

    mat1 = np.arange(m * n).reshape(shape1)
    vec1 = np.arange(n).reshape(shape2)

    indeps = IndepVarComp()
    indeps.add_output(
        'mat1',
        val=mat1,
        shape=shape1,
    )

    indeps.add_output(
        'vec1',
        val=vec1,
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
                             MatVecComp(in_names=['mat1', 'vec1'],
                                        out_name='y',
                                        in_shapes=[shape1, shape2]),
                             promotes=['*'])

    prob.setup()
    prob.check_partials(compact_print=True)
    prob.run_model()

    print(prob['mat1'])
    print(prob['vec1'])
    print(prob['y'])
