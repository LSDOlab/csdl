from typing import Union
from csdl.core.operation import Operation
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
        promotes_inputs=None,
        promotes_outputs=None,
        min_procs=1,
        max_procs=None,
        proc_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from csdl.core.model import Model
        from csdl.core.implicit_model import ImplicitModel
        if not isinstance(submodel, (Model, ImplicitModel, Operation)):
            TypeError("subsys must be a Model or Operation")
        self.name: str = name
        self.promotes = promotes
        self.promotes_inputs = promotes_inputs
        self.promotes_outputs = promotes_outputs
        self.submodel: Union[Model, ImplicitModel, Operation] = submodel

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
