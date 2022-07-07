try:
    from csdl.lang.model import Model
except:
    pass
from typing import Dict, Any, Set
from csdl.utils.prepend_namespace import prepend_namespace
from csdl.utils.find_promoted_name import find_promoted_name


def collect_design_variables(
    model: 'Model',
    promoted_to_unpromoted: Dict[str, Set[str]],
    unpromoted_to_promoted: Dict[str, str],
    namespace: str = '',
    design_variables: Dict[str, Dict[str, Any]] = dict(),
) -> Dict[str, Dict[str, Any]]:
    for k, design_variable in model.design_variables.items():
        name = find_promoted_name(
            k,
            model.promoted_to_unpromoted,
            model.unpromoted_to_promoted,
        )
        name = find_promoted_name(
            prepend_namespace(namespace, name),
            promoted_to_unpromoted,
            unpromoted_to_promoted,
        )

        # TODO: make this more helpful for users to find both
        # constraints
        if name in design_variables.keys():
            raise ValueError(f"Redundant design variable {name} declared.")

        design_variables[name] = design_variable

    for s in model.subgraphs:
        design_variables = collect_design_variables(
            s.submodel,
            promoted_to_unpromoted,
            unpromoted_to_promoted,
            prepend_namespace(namespace, s.name),
            design_variables=design_variables,
        )

    return design_variables
