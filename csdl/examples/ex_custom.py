from csdl import Model, GraphRepresentation
import csdl


class ExampleQuadraticEquationImplicitScalar(Model):
    """
    :param var: x
    """

    def define(self):
        from csdl.examples.operations.solve_quadratic import SolveQuadratic

        # These values overwrite the values within the CustomOperation
        a = self.declare_variable('a', val=1.)
        b = self.declare_variable('b', val=-4.)
        c = self.declare_variable('c', val=3.)

        # Solve quadratic equation using a CustomImplicitOperation
        x = csdl.custom(a, b, c, op=SolveQuadratic())
        self.register_output('x', x)


class ExampleQuadraticEquationImplicitArray(Model):
    """
    :param var: x
    """

    def define(self):
        from csdl.examples.operations.solve_quadratic import SolveQuadratic

        # These values overwrite the values within the CustomOperation
        a = self.declare_variable('a', val=[1, -1])
        b = self.declare_variable('b', val=[-4, 4])
        c = self.declare_variable('c', val=[3, -3])

        # Solve quadratic equation using a CustomImplicitOperation
        x = csdl.custom(a, b, c, op=SolveQuadratic())
        self.register_output('x', x)
