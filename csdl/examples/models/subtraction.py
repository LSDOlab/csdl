from csdl import Model


class SubtractionFunction(Model):

    def define(self):

        # inputs
        f = self.declare_variable('f')
        c = self.declare_variable('c')

        d = f - c

        # outputs
        self.register_output('d', d)


class SubtractionVectorFunction(Model):

    def define(self):

        # inputs
        f = self.declare_variable('f', shape=(3,))
        c = self.declare_variable('c', shape=(3,))

        d = f - c

        # outputs
        self.register_output('d', d)
