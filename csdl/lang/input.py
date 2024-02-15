from csdl.lang.variable import Variable
from csdl.utils.get_shape_val import get_shape_val


class Input(Variable):
    """
    Container for creating a value that is constant during model
    evaluation; i.e. independent variable, or design variable
    """
    def __init__(
        self,
        name: str,
        val=1.0,
        shape=(1, ),
        units=None,
        desc='',
        *args,
        **kwargs,
    ):
        super().__init__(
            name,
            val=val,
            shape=shape,
            units=units,
            desc=desc,
            *args,
            **kwargs,
        )

        self.shape, self.val = get_shape_val(shape, val)
        # if name == 'mtx':
        #     import numpy as np
        #     print(np.linalg.norm(self.val))
        #     exit()