from typing import Callable
from csdl.opt.combine_operations import combine_operations
from csdl.opt.remove_dead_code import remove_dead_code
from csdl.opt.remove_duplicate_nodes import remove_duplicate_nodes
from csdl.opt.run_all_optimizations import run_all_optimizations
from csdl.rep.graph_representation import GraphRepresentation
from typing import Callable

functions: list[Callable[
    [GraphRepresentation], GraphRepresentation]] = [
        lambda x: x,
        remove_dead_code,
        combine_operations,
        lambda x: remove_dead_code(combine_operations(x)),
        # lambda x: combine_operations(remove_dead_code(x)),
        run_all_optimizations,
    ]
