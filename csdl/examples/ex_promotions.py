import csdl
from csdl import Model, GraphRepresentation

# TESTS:
# - Different shapes throw error:
# --- ErrorPromotionShapeMismatch
# --- ErrorWrongShapeTwoModels
# - Cycles throw error
# --- ErrorCycle
# --- ErrorCycleTwoModels
# - Same name inputs
# --- ErrorInputs
# - Same name outputs
# --- ErrorOutputs
# - Same name input and output
# --- ErrorInputOutput
# - Test if two variables form a connection when promoted to same
# --- ExampleManualPromotion
# - Try to promote multiple variables with some of them invalid
# --- ErrorPartialPromotion
# - Check value of variable using same names
# --- ExampleAbsoluteRelativeName
# - Other edge cases:
# --- ExampleSameIOUnpromoted
# --- ErrorTwoModelsPromoted
# --- ExampleTwoModelsUnpromoted
# --- ExampleUnconnectedVars
# --- ExampleStackedModels

# --- ExampleComplex

class ExampleManualPromotion(Model):
    # Promoting a variable to a model will automatically connect them
    # Return sim['f'] = 6

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        # a = self.create_input('a', val=3.0)
        a = self.declare_variable('a', val=3.0)

        self.add(
            AdditionFunction(),
            promotes=['a', 'f'],
            name='addition',
        )

        f = self.declare_variable('f')
        self.register_output('b', f + a)


class ExampleAbsoluteRelativeName(Model):
    # We can access absolute and relative names for variables interchangibly
    # Return sim['model.f'] = 4 or sim['f'] = 4

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        a = self.create_input('a', val=3.0)

        self.add(
            AdditionFunction(),
            promotes=[
                'a', 'b', 'f'
            ],  # We can access f as sim['f']  or sim['model.f']
            name='model')


class ErrorInputs(Model):
    # Can't promote a variable to a model containing inputs with the same name
    # Return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        f = self.create_input('f', val=3.0)

        # Promoting b will raise an error as it is already an input
        self.add(AdditionFunction(),
                 promotes=['a', 'b', 'f'],
                 name='model')

        self.register_output('y', f + 1)


class ErrorOutputs(Model):
    # Can't promote a variable to a model containing outputs with the same name
    # Return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        b = self.create_input('b', val=3.0)
        self.register_output('f', b + 1)

        # Promoting f will raise an error because it is already an output
        self.add(AdditionFunction(),
                 promotes=['a', 'b', 'f'],
                 name='model')


class ErrorInputOutput(Model):
    # Can't promote a model containing inputs that are outputs in another model
    # Return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction
        from csdl.examples.models.subtraction import SubtractionFunction

        a = self.create_input('a', val=3.0)
        d = self.create_input('a', val=3.0)

        self.add(SubtractionFunction(),
                 promotes=['d', 'c', 'f'],
                 name='model')

        # Promoting output model2.a will raise an error because it is already an input of model
        self.add(AdditionFunction(),
                 promotes=['a', 'b', 'f'],
                 name='model2')


class ErrorPartialPromotion(Model):
    # Trying to promote multiple variables with some of them invalid will result in an error
    # Return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        a = self.create_input('a', val=3.0)
        self.register_output('f', a + 1.0)

        # We can promote a and b but trying to
        # promote f will result in an error as it is already an output
        # in the parent model
        self.add(AdditionFunction(),
                 promotes=['a', 'b', 'f'],
                 name='model')
class ErrorCycleTwoModels(Model):
    # Promotions cannot be made if connections cause cycles
    # return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction
        from csdl.examples.models.subtraction import SubtractionFunction

        self.add(AdditionFunction(),
                 promotes=['b', 'a', 'f'],
                 name='model')

        self.add(SubtractionFunction(),
                 promotes=['c', 'd', 'f'],
                 name='model2')

        d = self.declare_variable('d')
        self.register_output(
            'b', d + 1.0)  # b is an input to 'model', creating a cycle


class ExampleStackedModels(Model):
    # Autopromotions should work for models within models
    # return sim['b'] = 11

    def define(self):

        a = self.create_input('a', val=3.0)

        m = Model()
        am = m.create_input('am', val=2.0)
        mm = Model()
        amm = mm.declare_variable(
            'a', val=1.0
        )  # 'model1.model2.a' should automatically promote and connect to 'a'
        mm.register_output('bmm', amm * 2.0)
        m.add(mm, name='model2')
        bmm = m.declare_variable('bmm')
        m.register_output('bm', bmm + am)
        self.add(m, name='model1')

        bm = self.declare_variable('bm')
        self.register_output('b', bm + a)

class ExampleComplex(Model):
    # Use a complex hierarchical model
    # return sim['x3_out'] = 1.0
    # return sim['x4_out'] = 10.0
    # return sim['hierarchical.ModelB.x2'] = 6.141
    # return sim['hierarchical.ModelB.ModelC.x1'] = 5.0
    # return sim['hierarchical.ModelB.ModelC.x2'] = 9.0
    # return sim['hierarchical.ModelF.x3_out'] = 0.01

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.hierarchical import Hierarchical

        self.add(
            Hierarchical(),
            name='hierarchical',
        )

        # f = self.declare_variable('f')
        # self.register_output('b', f + a)




class ErrorPromotionShapeMismatch(Model):
    # Promotions should not be made if two variables have different shapes
    # return sim['f'] = 4

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        a = self.create_input('a', shape=(2, 1))

        # a and f in parent model have different shapes from a and f in
        # submodel
        self.add(AdditionFunction())

        f = self.declare_variable('f', shape=(2, 1))
        self.register_output('y', f + a)


class ErrorWrongShapeTwoModels(Model):
    # Promotions should not be made if two variables with different shapes
    # return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction
        from csdl.examples.models.subtraction import SubtractionVectorFunction

        a = self.create_input('a', val=3.0)

        input_dict = {}
        input_dict['name'] = 'a'
        # add two models where the second has the wrong shape
        self.add(AdditionFunction(),
                 name='model',
                 promotes=['a', 'b', 'f'])

        self.add(
            SubtractionVectorFunction(),
            name='model2',
            promotes=[
                'f', 'c', 'd'
            ])  # promoting f will throw an error for shape mismatch

        d = self.declare_variable('d')
        self.register_output('y', d + a)


class ErrorCycle(Model):
    # Promotions cannot be made if connections cause cycles
    # return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        g = self.create_input('g', val=3.0)
        f = self.declare_variable('f')
        b = self.register_output('b', f + g)

        # Add a model that outputs model.f and takes in g+f
        # Promoting b will create a cycle which throws an error
        self.add(AdditionFunction(),
                 promotes=['a', 'b', 'f'],
                 name='model')

        self.register_output('y', g + b)


# OTHERS:
class ExampleSameIOUnpromoted(Model):
    # Can create two outputs in two models with same names if unpromoted
    # Return sim['f'] = 1, sim['model.f'] = 2

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        a = self.create_input('a')

        self.add(
            AdditionFunction(),
            promotes=[
                'a', 'b'
            ],  # Promoting only a and b allows two outputs: f and model.f
            name='model')

        self.register_output('f', a * 1.0)


class ErrorTwoModelsPromoted(Model):
    # Can't have two models with same variable names if both promoted
    # return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        self.add(AdditionFunction(), promotes=['a', 'f'], name='model')

        self.add(
            AdditionFunction(),
            promotes=[
                'b', 'f'
            ],  # can't promote model2.f as we promoted model.f
            name='model2')


class ExampleTwoModelsUnpromoted(Model):
    # Can have two models with same variable names if only one variable is promoted
    # return sim['f'] = 3, sim['model2.f'] = 2

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        self.create_input('a', val=2.0)
        self.add(AdditionFunction(),
                 promotes=['a', 'b',
                           'f'],
                 name='model')

        self.add(
            AdditionFunction(),
            promotes=[],  # can't promote model2.f as we promoted model.f
            name='model2')


class ExampleUnconnectedVars(Model):
    # Declared variable with same local name should not be connected if unpromoted
    # sim['f'] should be = 2

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        self.add(AdditionFunction(), promotes=[], name='model')

        f = self.declare_variable('f')

        self.register_output('y', f + 1.0)



class ExampleJumpedPromotion(Model):
    # Promote a variable that has not been explicitly declared in that model but in the model above
    # return sim['f'] = 5

    def define(self):

        a = self.create_input('a', val=3.0)

        m = Model()
        am = m.create_input('am', val=2.0)
        m.register_output('bm', am*2.0)

        mm = Model()
        mm.create_input(
            'b', val=2.0
        )
        # promote b in m even though it hasn't been declared in m
        m.add(mm, name='model2', promotes=['b'])
        self.add(m, name='model1', promotes=['b'])

        bb = self.declare_variable('b')
        self.register_output('f', bb + a)


class ErrorPromoteUnpromoted(Model):
    # Try to promote an unpromoted variable
    # return keyerror

    def define(self):

        a = self.create_input('a', val=3.0)

        m = Model()
        am = m.create_input('am', val=2.0)
        m.register_output('bm', am*2.0)

        mm = Model()
        mm.create_input(
            'b', val=2.0
        )
        # Do no promote b
        m.add(mm, name='model2', promotes=[])

        # but promote model2.b
        self.add(m, name='model1', promotes=['model2.b'])

        bb = self.declare_variable('b')
        self.register_output('f', bb + a)


class ErrorPromoteNonexistant(Model):
    # Try to promote a variable that does no exist
    # return keyerror

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        # a = self.create_input('a', val=3.0)
        a = self.declare_variable('a', val=3.0)

        self.add(
            AdditionFunction(),
            promotes=['a', 'f', 'c'],
            name='addition',
        )

        f = self.declare_variable('f')
        self.register_output('b', f + a)


class ErrorPromoteAbsoluteName(Model):
    # Try to promote a variable using absolute name
    # return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        # a = self.create_input('a', val=3.0)
        a = self.declare_variable('a', val=3.0)

        self.add(
            AdditionFunction(),
            promotes=['addition.a', 'f'],
            name='addition',
        )

        f = self.declare_variable('f')
        self.register_output('b', f + a)



class ExampleParallelTargets(Model):
    # Use 1 input for several declared variables of the same instance
    # return sim[f_out] = 15

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        self.create_input('a', val=2.0)
        self.create_input('b', val=3.0)

        self.add(
            AdditionFunction(),
            name='addition1',
            promotes=['a', 'b'],
        )

        self.add(
            AdditionFunction(),
            name='addition2',
            promotes=['a', 'b'],
        )

        self.add(
            AdditionFunction(),
            name='addition3',
            promotes=['a', 'b'],
        )

        f1 = self.declare_variable('addition1.f')
        f2 = self.declare_variable('addition2.f')
        f3 = self.declare_variable('addition3.f')

        self.register_output('f_out', f1 + f2 + f3)


class ExampleSameIOUnpromoted(Model):
    # Can create two outputs in two models with same names if unpromoted
    # Return sim['f'] = 1, sim['model.f'] = 2

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        a = self.create_input('a')

        self.add(
            AdditionFunction(),
            promotes=[
                'a', 'b'
            ],  # Promoting only a and b allows two outputs: f and model.f
            name='model')

        self.register_output('f', a * 1.0)




# rep = GraphRepresentation(ExampleComplex())  # front end
# # rep = GraphRepresentation(AdditionFunction())
# rep.visualize_graph()
# # rep.visualize_unflat()
# # sim = Simulator(rep)  # back end
# # sim.run()
# # print(sim['f'])
# exit()
