try:
    from csdl import Model
except ImportError:
    pass
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.declared_variable import DeclaredVariable

from csdl.utils.typehints import Shape
from typing import Tuple, Dict, List


def collect_locally_defined_variables(
    model: 'Model',
) -> Tuple[Dict[str, Shape], Dict[str, Shape], Dict[
        str, Input | Output], Dict[str, DeclaredVariable]]:
    io: List[Input | Output] = []
    io.extend(model.inputs)
    io.extend(model.registered_outputs)
    source_shapes: Dict[str, Shape] = {x.name: x.shape for x in io}
    target_shapes: Dict[str, Shape] = {
        x.name: x.shape
        for x in model.declared_variables
    }
    sources: Dict[str, Input | Output] = {x.name: x for x in io}
    targets: Dict[str, DeclaredVariable] = {
        x.name: x
        for x in model.declared_variables
    }
    return source_shapes, target_shapes, sources, targets
