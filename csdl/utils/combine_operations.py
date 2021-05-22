from csdl.core.output import Output
from csdl.core.standard_operation import StandardOperation
from csdl.operations.combined import combined

# TODO: derivatives wrt op2.dependencies
# TODO: derivatives wrt variables that are in op1.dependencies AND op2.dependencies


def combine_operations(registered_outputs, out: Output):
    combined_op = combined()
    if isinstance(out, Output):
        op2 = out.dependencies[0]
        # TODO: support operations with multiple outputs
        if len(op2.outs) == 1 and isinstance(
                op2,
                StandardOperation):  # and 'elementwise' in op2.properties:
            op2.define_compute_strings()

            # Do not combine operations if intermediate variables are
            # registered outputs of the model
            combine = True
            for state in op2.dependencies:
                if state in registered_outputs:
                    combine = False
                    break

            if combine is True:
                # add derivatives wrt terminals
                for state in op2.dependencies:
                    if len(state.dependencies) == 0:
                        combined_op.compute_derivs[
                            state.name] = op2.compute_derivs[state.name]
                print('{}, compute_derivs (before loops): {}'.format(
                    combined_op.name, repr(combined_op.compute_derivs)))

                # add derivatives wrt nonterminals
                for state in op2.dependencies:
                    if len(state.dependencies) != 0:
                        op1 = state.dependencies[0]

                        # TODO: support operations with multiple outputs
                        if len(op1.outs) == 1 and isinstance(
                                op1, StandardOperation
                        ):  # and 'elementwise' in op1.properties:

                            op1.define_compute_strings()

                            # combine operations
                            combined_op.compute_string += op1.compute_string + '\n'

                            # add combined_op as a dependency of output
                            combined_op.outs = [out]
                            combined_op.dependents = [out]
                            out.dependencies = [combined_op]

                            for depvar in op1.dependencies:
                                # apply chain rule
                                if depvar in op2.dependencies:
                                    # op2 also depends explicitly on depvar
                                    combined_op.compute_derivs[
                                        depvar.
                                        name] += '+(' + op2.compute_derivs[
                                            state.
                                            name] + ')*(' + op1.compute_derivs[
                                                depvar.name] + ')'
                                else:
                                    # only op1 depends explicitly on depvar
                                    combined_op.compute_derivs[
                                        depvar.
                                        name] = '(' + op2.compute_derivs[
                                            state.
                                            name] + ')*(' + op1.compute_derivs[
                                                depvar.name] + ')'
                                print(
                                    '{}, compute_derivs (within depvar loop): {}'
                                    .format(combined_op.name,
                                            repr(combined_op.compute_derivs)))

                                # add combined_op to graph
                                depvar.dependents.append(combined_op)
                                depvar.dependents.remove(op1)
                            print('{}, compute_derivs (after depvar loop): {}'.
                                  format(combined_op.name,
                                         repr(combined_op.compute_derivs)))

                            # add depvars as dependencies of combined_op
                            combined_op.dependencies.extend(op1.dependencies)
                        else:
                            for depvar in op1.dependencies:
                                combine_operations(registered_outputs, depvar)

                print('{}, compute_derivs (after state loop): {}'.format(
                    combined_op.name, repr(combined_op.compute_derivs)))

                combined_op.compute_string += op2.compute_string
                print('{}, compute_string (after state loop): {}'.format(
                    combined_op.name, combined_op.compute_string))
        else:
            for state in op2.dependencies:
                combine_operations(registered_outputs, state)
