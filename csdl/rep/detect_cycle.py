try:
    from csdl.lang.model import Model
except ImportError:
    pass
try:
    from csdl.lang.input import Input
except ImportError:
    pass
try:
    from csdl.lang.output import Output
except ImportError:
    pass

from typing import Dict, List, Union
from csdl.utils.typehints import Shape
from csdl.lang.node import Node
from csdl.lang.subgraph import Subgraph


def detect_cycle(
    start: Subgraph,
    n: Node,
    namespace: Union[str, None] = None,
):
    for prev in start.dependencies:
        if prev is start:
            raise RecursionError(
                "Cycle detected due to model {} of type {}".format(
                    start.name if namespace is None else namespace +
                    '.' + start.name, type(start.submodel)), )
        detect_cycle(start, prev)


