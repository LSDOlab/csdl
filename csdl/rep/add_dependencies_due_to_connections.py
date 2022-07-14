try:
    from csdl.lang.model import Model
except ImportError:
    pass
from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.subgraph import Subgraph
from typing import Set, Dict, Tuple, List, Union


def find_src_node(
    model: 'Model',
    node_name: List[str],
) -> Union[Input, Output, Subgraph]:
    if len(node_name) == 1:
        io: List[Union[Input, Output]] = []
        io.extend(model.inputs)
        io.extend(model.registered_outputs)
        # This list has length <= 1
        possible_node_names = list(
            filter(lambda x: x.name == node_name[0], io))
        if len(possible_node_names) == 0:
            raise KeyError(
                "{} is not a valid source name".format(node_name))
        return possible_node_names[0]
    else:
        return list(
            filter(lambda x: x.name == node_name[0],
                   model.subgraphs))[0]


def find_tgt_node(
    model: 'Model',
    node_name: List[str],
) -> Union[DeclaredVariable, Subgraph]:
    if len(node_name) == 1:
        return list(
            filter(lambda x: x.name == node_name[0],
                   model.declared_variables))[0]
    else:
        return list(
            filter(lambda x: x.name == node_name[0],
                   model.subgraphs))[0]


def add_dependencies_due_to_connections(model: 'Model'):
    """
    Add dependency relationships due to connections so that the compiler
    can automatically reorder models by sorting the "unflat" graph.
    """
    for a, b in model.connections:
        src_path = a.rsplit('.')
        tgt_path = b.rsplit('.')

        # add dependency relationship at the level in the model tree
        # where the connection exists between two branches of the model
        # tree
        stop: int = min(len(src_path), len(tgt_path)) - 1
        for i, (p, q) in enumerate(zip(src_path, tgt_path)):
            if p == q and i < stop:
                # src_path, tgt_path are on the same branch
                # i < stop ensures p, q are a model name
                subgraphs = {x.name: x for x in model.subgraphs}
                add_dependencies_due_to_connections(
                    subgraphs[p].submodel)
            else:
                # src_path, tgt_path are not on the same branch and may
                # represent a variable if e.g. len(src_path[i:]) == 1
                src_node: Union[Input, Output, Subgraph] = find_src_node(
                    model,
                    src_path[i:],
                )
                tgt_node: Union[DeclaredVariable, Subgraph] = find_tgt_node(
                    model,
                    tgt_path[i:],
                )
                if isinstance(src_node, (Input, Output)) and isinstance(
                        tgt_node, DeclaredVariable):
                    # TODO: error or warning?
                    pass
                # add dependency relationship between
                # - source (input/output) and model
                # - source (input/output) and target (declared variable)
                # - model and target (declared variable)
                # - model and model
                if src_node not in tgt_node.dependencies and tgt_node not in src_node.dependents:
                    tgt_node.add_dependency_node(src_node)
                    src_node.add_dependent_node(tgt_node)
                else:
                    # TODO: warn redundant connections
                    pass
