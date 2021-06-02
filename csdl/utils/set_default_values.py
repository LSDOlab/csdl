# TODO: ensure explicit output values are not reassigned
def set_default_values(model, promotes=[], promotes_inputs=[]):
    from csdl.core.model import Model
    variables_promoted_from_children = []
    inputs_promoted_from_children = []
    # gather all variables promoted to this level
    for subgraph in model.subgraphs:
        if isinstance(subgraph.submodel, Model):
            inputs, variables = set_default_values(
                subgraph.submodel,
                promotes=subgraph.promotes,
                promotes_inputs=subgraph.promotes_inputs,
            )
            inputs_promoted_from_children.extend(inputs)
            variables_promoted_from_children.extend(variables)

    # set default values for children based on parent values
    for var in model.inputs + model.variables:
        for child_var in variables_promoted_from_children:
            if var.name == child_var.name:
                child_var.val = var.val

    # set default values for inputs created in children
    for var in model.variables:
        for child_var in inputs_promoted_from_children:
            if var.name == child_var.name:
                var.val = child_var.val

    # gather variables promoted to parent
    inputs_promoted_to_parent = []
    variables_promoted_to_parent = []
    if promotes is None and promotes_inputs is None:
        return inputs_promoted_to_parent, variables_promoted_to_parent
    if promotes == ['*'] or promotes_inputs == ['*']:
        variables_promoted_to_parent = model.variables + variables_promoted_from_children
        inputs_promoted_to_parent = model.inputs + inputs_promoted_from_children
    else:
        for name in promotes_inputs:
            variables_promoted_to_parent.extend(
                list(
                    filter(lambda var: var.name == name,
                           model.variables_promoted_from_children)))
        for name in promotes:
            variables_promoted_to_parent.extend(
                list(
                    filter(lambda var: var.name == name,
                           model.variables_promoted_from_children)))

    return inputs_promoted_to_parent, variables_promoted_to_parent
