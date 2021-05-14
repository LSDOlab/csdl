from csdl.core.variable import Variable
from csdl.core.output import Output
from csdl.core.operation import Operation


def replace_output_leaf_nodes2(
    root: Output,
    op: Operation,
    leaf: Variable,
):
    for var in op.dependencies:
        if var is root:
            # replace dependency reference with Variable object
            op.remove_dependency_node(var)
            op.add_dependency_node(leaf)
        replace_output_leaf_nodes(root, var, leaf)


def replace_output_leaf_nodes(
    root: Output,
    node: Variable,
    leaf: Variable,
):
    for op in node.dependencies:
        replace_output_leaf_nodes2(root, op, leaf)
