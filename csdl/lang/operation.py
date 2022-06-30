from typing import Any, Tuple
from csdl.lang.node import Node
try:
    from csdl.lang.output import Output
except ImportError:
    pass
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
        super().__init__(*args, **kwargs)
        self.nargs: int = len(args)
        for arg in args:
            self.add_dependency_node(arg)
        self.nouts: int = 0

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
