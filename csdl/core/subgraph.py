from typing import Union, List
from csdl.core.custom_operation import CustomOperation
from csdl.core.node import Node


class Subgraph(Node):
    """
    Class for declaring an input variable
    """
    def __init__(
        self,
        name: str,
        submodel,
        *args,
        promotes=None,
        min_procs=1,
        max_procs=None,
        proc_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from csdl.core.model import Model
        from csdl.core.implicit_operation import ImplicitOperation
        if not isinstance(submodel, (Model)):
            TypeError("subsys must be a Model")
        self.name: str = name
        self.promotes = promotes
        self.submodel: Model = submodel

    def add_dependency_node(self, dependency):
        from csdl.core.variable import Variable
        if not isinstance(dependency, (Variable, Subgraph)):
            raise TypeError(
                "Dependency of a Subgraph object must be a Variable or Subraph object"
            )
        self.dependencies.append(dependency)

        # # Add dependency
        # if dependency not in self.dependencies:
        #     self.dependencies.append(dependency)
        # else:
        #     raise ValueError(dependency.name, 'is duplicate')
