from csdl.rep.graph_representation import GraphRepresentation
from csdl.opt.remove_dead_code import remove_dead_code
from csdl.opt.combine_operations import combine_operations


def run_all_optimizations(
        rep: GraphRepresentation) -> GraphRepresentation:
    """
    Perform implementation-independent optimizations on intermediate
    representation and determine one possible execution order of model
    operations in final executable object
    This is the default function for a compiler back end to call, which
    performs all available optimizations.
    Back end developers can choose which optimizations to perform.
    """
    return combine_operations(remove_dead_code(rep))
