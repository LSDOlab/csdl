from typing import List
from csdl.core.model import Model
from csdl.core.subgraph import Subgraph


def check_for_siblings_conflicts(self: Model, desired_promotion: str):
    count = 1
    for submodel in self.submodels:
        if desired_promotion in submodel.

def promote_inputs_from_submodels(
    self: Model,
    promotes: List[str],
):
    """
    Check if inputs from submodels can be promoted to this level.
    If a given input cannot be promoted, do nothing.
    Otherwise, update self.promoted_inputs
    """
    for desired_promotion in promotes:
        # do not promote if a submodel has an input with the same name
        if desired_promotion not in self.inputs:
            conflict = check_for_siblings_conflicts(self, desired_promotion)
            # promote only if there are no conflicting names among siblings
            if conflict is False:
                self.promoted_inputs.append(desired_promotion)

def promote_inputs_to_parent(
    subgraph: Subgraph,
):
    submodel: Model = subgraph.submodel
    promotes: List[str] = subgraph.promotes
    # input_names = [var.name for var in submodel.inputs]
    input_names = [var.name for var in submodel.promoted_inputs]

    promoted_inputs = []
    for desired_promotion in promotes:
        if desired_promotion in input_names:
            promoted_inputs.append(desired_promotion)
    return promoted_inputs
