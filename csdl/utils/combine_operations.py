from csdl.core.output import Output
from csdl.core.standard_operation import StandardOperation
from csdl.operations.combined import combined

# TODO: derivatives wrt op2.dependencies
# TODO: derivatives wrt variables that are in op1.dependencies AND
# op2.dependencies


def can_combine(op: StandardOperation):
    if not isinstance(op, StandardOperation):
        return False
    if len(op.outs) != 1:
        return False
    cc = True
    if 'elementwise' in op.properties.keys():
        cc = cc and op.properties['elementwise']
    if 'iterative' in op.properties.keys():
        cc = cc and not op.properties['iterative']
    return cc


def combine_operations(registered_outputs, out: Output):
    repeat = False
    combined_op = combined()
    if isinstance(out, Output):
        op2 = out.dependencies[0]
        # TODO: support operations with multiple outputs
        if can_combine(op2):
            op2.define_compute_strings()

            # Do not combine operations if intermediate variables are
            # registered outputs of the model
            combine = True
            for state in op2.dependencies:
                if state in registered_outputs:
                    combine = False
                    break

            if combine is True:
                # add derivatives wrt nonterminals
                for state in op2.dependencies:
                    if len(state.dependencies) != 0:
                        repeat = True
                        op1 = state.dependencies[0]

                        # TODO: support operations with multiple outputs
                        if can_combine(op1):

                            op1.define_compute_strings()

                            # combine operations
                            combined_op.compute_string += op1.compute_string + '\n'

                            # add combined_op as a dependency of output
                            combined_op.outs = [out]
                            combined_op.dependents = [out]
                            out.dependencies = [combined_op]

                            for depvar in op1.dependencies:
                                # add combined_op to graph
                                depvar.dependents.append(combined_op)
                                depvar.dependents.remove(op1)

                            # add depvars as dependencies of combined_op
                            combined_op.dependencies.extend(op1.dependencies)
                        else:
                            for depvar in op1.dependencies:
                                combine_operations(registered_outputs, state)

                combined_op.compute_string += op2.compute_string + '\n'
                print('{}, compute_string (after state loop): {}'.format(
                    combined_op.name, combined_op.compute_string))
        else:
            for state in op2.dependencies:
                combine_operations(registered_outputs, state)
    return repeat
