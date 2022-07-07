from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.model import Model
from csdl.lang.input import Input
from typing import List, Tuple

# TODO: this should raise an error
def set_default_values(
    model: Model,
    promotes=[],
) -> Tuple[List[Input], List[DeclaredVariable]]:
    """
    This function only needs to be called on the unflat graph.
    This assumes that promotions and connections are valid.
    """
    variables_promoted_from_children: List[DeclaredVariable] = []
    inputs_promoted_from_children: List[Input] = []
    # gather all variables promoted to this level
    for subgraph in model.subgraphs:
        inputs, variables = set_default_values(
            subgraph.submodel,
            promotes=subgraph.promotes,
        )
        inputs_promoted_from_children.extend(inputs)
        variables_promoted_from_children.extend(variables)

    # set default values for children based on parent values
    for var in model.declared_variables:
        for child_var in variables_promoted_from_children:
            if var.name == child_var.name:
                child_var.val = var.val

    # set default values for inputs created in children
    for var in model.declared_variables:
        for child_var in inputs_promoted_from_children:
            if var.name == child_var.name:
                var.val = child_var.val

    # gather variables promoted to parent
    inputs_promoted_to_parent = []
    variables_promoted_to_parent = []
    if promotes is None:
        return inputs_promoted_to_parent, variables_promoted_to_parent
    else:
        for name in promotes:
            variables_promoted_to_parent.extend(
                list(
                    filter(lambda var: var.name == name,
                           model.variables_promoted_from_children)))

    return inputs_promoted_to_parent, variables_promoted_to_parent
