from csdl import Model
import csdl


class ExampleOTGroupWithinOTGroup(Model):
    """
    :param var: x1
    :param var: x2
    :param var: y1

    """
    def define(self):
        # Create independent variable
        x1 = self.create_input('x1', val=40)

        # Create subsystem that depends on previously created
        # independent variable
        m = csdl.Model()

        # Declaring and creating variables within the csdl subgroup
        a = m.declare_variable('x1')
        b = m.create_input('x2', val=12)
        m.register_output('prod', a * b)
        self.add(m, name='m', promotes=['*'])

        # Declaring input that will receive its value from the csdl subgroup
        x2 = self.declare_variable('x2')

        # Simple addition
        y1 = x2 + x1
        self.register_output('y1', y1)
