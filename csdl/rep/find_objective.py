try:
    from csdl.lang.model import Model
except:
    pass
from typing import Dict, Any, Set
from csdl.utils.prepend_namespace import prepend_namespace
from csdl.utils.find_promoted_name import find_promoted_name


def find_objective(
        model: 'Model',
        promoted_to_unpromoted: Dict[str, Set[str]],
        unpromoted_to_promoted: Dict[str, str],
        namespace: str = '',
        objective: Dict[str, Any] = dict(),
) -> Dict[str, Any]:
    if len(model.objective) > 0:
        name = find_promoted_name(
            model.objective['name'],
            model.promoted_to_unpromoted,
            model.unpromoted_to_promoted,
        )
        name = find_promoted_name(
            prepend_namespace(namespace, name),
            promoted_to_unpromoted,
            unpromoted_to_promoted,
        )

        if len(objective) > 1:
            raise ValueError(
                f"Cannot add more than one objective. Attempting to add two objectives, {objective['name']} and {name}."
            )

        objective = model.objective
        objective['name'] = name

    for s in model.subgraphs:
        objective = find_objective(
            s.submodel,
            promoted_to_unpromoted,
            unpromoted_to_promoted,
            prepend_namespace(namespace, s.name),
            objective=objective,
        )

    return objective
