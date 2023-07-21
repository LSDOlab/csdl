import csdl.operations as ops
from csdl.lang.variable import Variable
from csdl.lang.output import Output 

def hankel(
        var: Variable,
        kind: int=1,
        order: int=1
        ):
    '''
    This function computes the Hankel function

    **Parameters**

    var: Variable
        The variable to evaluate the Hankel function

    kind: int
        The kind of Hankel function. The options are 1 (J) or 2 (Y)

    order: int
        Order of the Hankel function
    '''

    # Check if variable
    if not isinstance(var, Variable):
        raise TypeError(var, " is not an Variable object")
    
    # check if arguments are integers
    if kind not in [1,2]:
        raise TypeError(f"Hankel argument \'kind\' (input {var.name}) must be the integer 1 or 2. {kind} given.")
    if not isinstance(order, int):
        raise TypeError(f"Hankel \'order\' (input {var.name}) must be an integer. {order} given.")
    op = ops.hankel(var, kind=kind, order=order)
    op.outs = (Output(
        None,
        op=op,
        shape=var.shape
    ), )
    return op.outs[0]
