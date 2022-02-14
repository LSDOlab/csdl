from csdl.core.variable import Variable
from csdl.core.output import Output
from csdl.core.subgraph import Subgraph
from csdl.core.input import Input
from csdl.core.standard_operation import StandardOperation
from csdl.operations.combined import combined
from csdl.utils.check_property import check_property

# TODO: derivatives wrt op2.dependencies
# TODO: derivatives wrt variables that are in op1.dependencies AND
# op2.dependencies


def can_combine(op: StandardOperation):
    if not isinstance(op, StandardOperation):
        return False
    if len(op.outs) != 1:
        return False
    cc = True
    cc = check_property(cc, op, 'elementwise', True)
    return cc


def terminal(var, registered_outputs):
    if var in registered_outputs or not isinstance(var, Output):
        return True
    if len(var.dependencies) == 0:
        return True
    return False


# TODO: where to set step size for combining operations?
# TODO: support operations with multiple outputs
def combine_operations(registered_outputs, out: Output):
    # value to return; only used by initial call from outer
    # while loop, not recursive calls
    terminate = True
    combine = False
    # only combine preceding operations for an output
    if isinstance(out, Output):
        combine_op = combined()
        # op2 is the second operation to combine
        ultimate_operation = out.dependencies[0]
        # if op2 satisfies conditions for combining with another
        # operation, try to combine with previous operation
        if can_combine(ultimate_operation):
            ultimate_operation.define_compute_strings()
            for state in ultimate_operation.dependencies:
                # FIXME: erasing terminal nodes, not adding as dependencies
                # if state is not terminal, do not stop; continue only
                # if state is Output, not Subgraph
                if terminal(state, registered_outputs):
                    # add state as dependency of new combined operation
                    if state not in combine_op.dependencies:
                        combine_op.dependencies.append(state)
                elif isinstance(state, Output):
                    # ??
                    terminate = False
                    penultimate_operation = state.dependencies[0]
                    # if op1 satisfies conditions for combining with
                    # another operation, try to combine with current
                    # operation operation
                    if can_combine(penultimate_operation):
                        combine = True
                        # update combined operation string
                        penultimate_operation.define_compute_strings()
                        combine_op.compute_string += penultimate_operation.compute_string + '\n'

                        # update graph
                        for depvar in penultimate_operation.dependencies:
                            try:
                                depvar.dependents.remove(
                                    penultimate_operation)
                            except:
                                pass
                            depvar.dependents.append(combine_op)
                        combine_op.dependencies.extend(
                            penultimate_operation.dependencies)
                        try:
                            state.dependents.remove(ultimate_operation)
                        except:
                            pass
                        state.dependents.append(combine_op)
                    # else:
                    #     combine_op.dependencies.remove(state)
            # append ultimate operation string to combined operation string
            if combine is True:
                ultimate_operation.dependencies = []
                combine_op.compute_string += ultimate_operation.compute_string + '\n'
                out.dependencies = [combine_op]
                combine_op.outs = (out, )
        else:
            # try combining earlier operations
            for state in ultimate_operation.dependencies:
                _ = combine_operations(registered_outputs, state)
    return terminate
