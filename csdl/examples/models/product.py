from csdl import Model


class Product(Model):

    def define(self):
        a = self.declare_variable('a', val=2)
        b = self.create_input('b', val=12)
        self.register_output('prod', a * b)
