from csdl.core.variable import Variable
from csdl.core.output import Output
from csdl.core.subgraph import Subgraph
from csdl.core.input import Input
from csdl.core.standard_operation import StandardOperation
from csdl.operations.combined import combined

# TODO: derivatives wrt op2.dependencies
# TODO: derivatives wrt variables that are in op1.dependencies AND
# op2.dependencies


def check_property(b, op, prop, truthy):
    try:
        b = b and op.properties[prop] == truthy
    except:
        pass
    return b


def can_combine(op: StandardOperation):
    if not isinstance(op, StandardOperation):
        return False
    if len(op.outs) != 1:
        return False
    cc = True
    cc = check_property(cc, op, 'elementwise', True)
    cc = check_property(cc, op, 'iterative', False)
    return cc


# TODO: where to set step size for combining operations?
# TODO: support operations with multiple outputs
def combine_operations(registered_outputs, out: Output):
    terminate = True
    if isinstance(out, Output):
        combine_op = combined()
        op2 = out.dependencies[0]
        if can_combine(op2):
            combine = True
            if combine is True:
                op2.define_compute_strings()
            for state in op2.dependencies:
                if state in registered_outputs:
                    combine = False
                    break
                if combine is True:
                    # if state is not terminal, do not stop
                    if isinstance(state, Variable) and not isinstance(
                            state, (Output, Subgraph)):
                        if state not in combine_op.dependencies:
                            combine_op.dependencies.append(state)
                    elif isinstance(state, Output):
                        terminate = False
                        op1 = state.dependencies[0]
                        if can_combine(op1):
                            # update combined operation string
                            op1.define_compute_strings()
                            combine_op.compute_string += op1.compute_string + '\n'

                            # update graph
                            for depvar in op1.dependencies:
                                try:
                                    depvar.dependents.remove(op1)
                                except:
                                    pass
                                depvar.dependents.append(combine_op)
                            combine_op.dependencies.extend(op1.dependencies)
                        else:
                            combine_operations(registered_outputs, state)
                    # update graph
                    try:
                        state.dependents.remove(op2)
                    except:
                        pass
                    state.dependents.append(combine_op)
            # update combined operation string
            if combine is True:
                op2.dependencies = []
                combine_op.compute_string += op2.compute_string + '\n'
                out.dependencies = [combine_op]
                combine_op.outs = (out, )
        else:
            for state in op2.dependencies:
                combine_operations(registered_outputs, state)
    return terminate
