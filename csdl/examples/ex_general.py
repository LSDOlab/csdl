from csdl import Model


class ExampleUnusedInputs(Model):
    def define(self):
        # These inputs are unused; will not be supplied to compiler
        # back end
        a = self.declare_variable('a', val=10)
        b = self.declare_variable('b', val=5)
        c = self.declare_variable('c', val=2)


class ErrorRegisterInput(Model):
    def define(self):
        a = self.declare_variable('a', val=10)
        # This will raise a TypeError
        self.register_output('a', a)
