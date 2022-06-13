from typing import Any, Tuple
from csdl.lang.node import Node
try:
    from csdl.lang.variable import Variable
except ImportError:
    pass


class Operation(Node):

    def __init__(
        self,
        *args: 'Variable',
        **kwargs: Any,
    ):
        self.nouts: int | None = None
        self.nargs: int | None = None
        super().__init__(*args, **kwargs)

        # TODO: why also custom operation?
        if self.nouts is None:
            AttributeError(
                "Operation subclass has not defined number of outputs")
        if self.nargs is None:
            AttributeError(
                "Operation subclass has not defined number of arguments"
            )
        if self.nouts is not None and self.nouts < 1:
            ValueError(
                "Operation classes must declare at least one output")
        if self.nargs is not None and self.nargs < 1:
            ValueError(
                "Operation classes must declare at least one argument"
                "or None if number of arguments is variable", )
        for arg in args:
            self.add_dependency_node(arg)

        if self.nargs is not None:
            if len(self.dependencies) > self.nargs:
                raise TypeError(
                    "{} can have at most {} nonliteral (Variable) object dependencies, found {}"
                    .format(repr(self), self.nargs,
                            len(self.dependencies)))
        from csdl.lang.output import Output
        self.outs: Tuple[Output, ...] = ()

    # TODO: this isn't very clean; op should never be None
    def add_dependency_node(self, dependency: 'Variable'):
        from csdl.lang.variable import Variable
        if not isinstance(dependency, Variable):
            raise TypeError(
                "Dependency of an Operation object must be a Variable object"
            )

        self.dependencies.append(dependency)

        # # Add dependency
        # if dependency not in self.dependencies:
        #     self.dependencies.append(dependency)
        # else:
        #     # raise ValueError(dependency.name, 'is duplicate')
        #     print(dependency.name, 'is duplicate')
