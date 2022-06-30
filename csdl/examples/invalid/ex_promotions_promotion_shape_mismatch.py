def example(Simulator):
    from csdl import Model, GraphRepresentation
    
    
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
    
    
    rep = GraphRepresentation(ErrorPromotionShapeMismatch())
    sim = Simulator(rep)
    sim.run()
    