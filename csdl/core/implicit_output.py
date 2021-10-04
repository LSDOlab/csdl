from typing import Tuple
from csdl.core.output import Output


class ImplicitOutput(Output):
    """
    Class for creating an implicit output
    """
    def __init__(
        self,
        name,
        val=1.0,
        shape: Tuple[int] = (1, ),
        units=None,
        res_units=None,
        desc='',
        lower=None,
        upper=None,
        ref=1.0,
        ref0=0.0,
        res_ref=1.0,
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
        *args,
        **kwargs,
    ):
        """
        Initialize implicit output

        **Parameters**

        name: str
            Name of variable to compute implicitly
        shape: Tuple[int]
            Shape of variable to compute implicitly
        val: Number or ndarray
            Initial value of variable to compute implicitly
        """
        super().__init__(
            name,
            val=val,
            shape=shape,
            units=units,
            res_units=res_units,
            desc=desc,
            lower=lower,
            upper=upper,
            ref=ref,
            ref0=ref0,
            res_ref=res_ref,
            tags=tags,
            shape_by_conn=shape_by_conn,
            copy_shape=copy_shape,
            # *args,
            # **kwargs,
        )
