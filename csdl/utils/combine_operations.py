from csdl.core.variable import Variable
from csdl.core.output import Output
from csdl.core.subgraph import Subgraph
from csdl.core.input import Input
from csdl.core.standard_operation import StandardOperation
from csdl.operations.combined import combined
from csdl.utils.check_property import check_property
from typing import Dict, List
from copy import copy
from csdl.utils.graph import remove_op_from_dag
from csdl.utils.graph import insert_op_into_dag


def can_combine(a: StandardOperation, b: StandardOperation,
                reg_dict: Dict[str, Output]):
    if not isinstance(a, StandardOperation):
        return False
    if not isinstance(b, StandardOperation):
        return False
    # only allow standard operations with single input and single output
    # to make derivative computation more efficient
    if len(a.dependencies) == 1 or len(a.dependents) != 1:
        return False
    if len(b.dependencies) == 1 or len(b.dependents) != 1:
        return False
    if check_property(a, 'elementwise', True) is False:
        return False
    if check_property(b, 'elementwise', True) is False:
        return False
    for v in b.dependencies:
        if v.name in reg_dict.keys():
            return False
    return True


def combine_operations(a, b, reg_dict) -> bool:
    c = can_combine(a, b, reg_dict)
    if c:
        # define compute strings for original operations;
        # must occur prior to removing from DAG
        a.define_compute_strings()
        b.define_compute_strings()

        # remove operations that we are combining from IR
        dca, dta = remove_op_from_dag(a)
        dcb, dtb = remove_op_from_dag(b)

        # create new combined operation with compute strings from
        # original operations
        combine_op = combined()
        combine_op.compute_string += a.compute_string + '\n'
        combine_op.compute_string += b.compute_string + '\n'

        # dependencies of b that are not dependents of a, plus dependencies
        # of a
        dc = list((set(dcb) - set(dta)).union(set(dca)))
        # dependents of a that are not dependencies of b, plus dependents
        # of b
        dt = list((set(dta) - set(dcb)).union(set(dtb)))
        insert_op_into_dag(combine_op, dc, dt)

    return c


def combine_operation_pass(outputs, reg_dict):
    for vb in outputs:
        terminate = False
        for b in vb.dependencies:  # list of length <= 1
            if isinstance(b, StandardOperation):
                for va in b.dependencies:
                    for a in va.dependencies:  # list of length <= 1
                        if isinstance(a, StandardOperation):
                            terminate = combine_operations(
                                a, b, reg_dict)
                        # if operations have been combined, then v.dependencies
                        # and b.dependencies have been modified, and it makes no
                        # sense to continue loop
                        if terminate is True:
                            break
                    if terminate is True:
                        break
        # continue combining operations earlier in graph
        for b in vb.dependencies:  # list of length <= 1
            # b is either original StandardOperation, or newly inserted
            # CombinedOperation
            combine_operation_pass(b.dependencies, reg_dict)
