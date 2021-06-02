from csdl.core.node import Node


class Operation(Node):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # TODO: why also custom operation?
        if not hasattr(self, 'nouts'):
            AttributeError(
                "Operation subclass has not defined number of outputs")
        if not hasattr(self, 'nargs'):
            AttributeError(
                "Operation subclass has not defined number of arguments")
        if self.nouts < 1:
            ValueError("Operation classes must declare at least one output")
        if self.nargs is not None and self.nargs < 1:
            ValueError(
                "Operation classes must declare at least one argument"
                "or None if number of arguments is variable", )
        for arg in args:
            self.add_dependency_node(arg)

    def add_dependency_node(self, dependency):
        from csdl.core.variable import Variable
        if not isinstance(dependency, Variable):
            raise TypeError(
                "Dependency of an Operation object must be a Variable object")
        if len(self.dependencies) == self.nargs:
            raise TypeError(
                "{} can have at most {} nonliteral (Variable) object dependencies"
                .format(repr(self), self.nargs), )
        self.dependencies.append(dependency)

        # # Add dependency
        # if dependency not in self.dependencies:
        #     self.dependencies.append(dependency)
        # else:
        #     raise ValueError(dependency.name, 'is duplicate')
