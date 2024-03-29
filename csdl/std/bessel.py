import csdl.operations as ops
from csdl.lang.variable import Variable
from csdl.lang.output import Output
from numbers import Number
import numpy as np

def bessel(
        var:Variable,
        kind:int=1,
        order:int=1,
        ):
    '''
    This function computes the Bessel function

    **Parameters**

    var: Variable
        The variable to evaluate the Bessel function

    kind: int
        The kind of Bessel function. The options are 1 (J) or 2 (Y)

    order: int
        Order of the Bessel function
    '''

    # Check if variable
    if not isinstance(var, Variable):
        raise TypeError(var, " is not an Variable object")
    
    # check if arguments are integers
    if kind not in [1,2]:
        raise TypeError(f"Bessel argument \'kind\' (input {var.name}) must be the integer 1 or 2. {kind} given.")
    if not isinstance(order, (Number, np.ndarray)):
        raise TypeError(f"Bessel \'order\' (input {var.name}) must be a integer/float or number array. {order} given.")
    if isinstance(order, np.ndarray):
        if var.shape != order.shape:
            raise TypeError(f"Bessel \'order\' (input {var.name}) must be the same shape as the input variable. order shape {order.shape} given, {var.shape} expected.")
    op = ops.bessel(var, kind=kind, order=order)
    op.outs = (Output(
        None,
        op=op,
        shape=var.shape
    ), )
    return op.outs[0]

# def bessel(var):
#     if not isinstance(var,Variable):
#         raise TypeError(var, "is not a Variable object")
#     op = ops.bessel(var)
#     op.outs = (Output(
#         None,
#         op=op,
#         shape=op.dependencies[0].shape,
#     ),)
#     return op.outs[0]