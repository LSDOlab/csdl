from typing import List

from csdl.core.variable import Variable


def collect_input_exprs(
    inputs: list,
    root: Variable,
    expr: Variable,
) -> List[Variable]:
    """
    Collect input nodes so that the resulting ``ImplicitModel`` has
    access to inputs outside of itself.
    """
    for dependency in expr.dependencies:
        if dependency.name != root.name:
            if isinstance(dependency, Variable) and len(
                    dependency.dependencies) == 0:
                inputs.append(dependency)
            inputs = collect_input_exprs(inputs, root, dependency)
    return inputs
