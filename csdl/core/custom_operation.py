from csdl.core.operation import Operation
from csdl.utils.parameters import Parameters


class CustomOperation(Operation):
    def __init__(self, *args, **kwargs):
        self.nargs = 0
        self.nouts = 1
        super().__init__(*args, **kwargs)
        self.nouts = 0
        self.input_meta = dict()
        self.output_meta = dict()
        self.derivatives_meta = dict()
        self.parameters = Parameters()
        self.initialize()
        self.parameters.update(kwargs)

    def initialize(self):
        pass

    def define(self):
        pass

    def compute_derivatives(self, inputs, derivatives):
        pass

    def add_input(
        self,
        name,
        val=1.0,
        shape=None,
        src_indices=None,
        flat_src_indices=None,
        units=None,
        desc='',
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
    ):
        if name in self.input_meta.keys():
            raise KeyError(name +
                           " was already declared an input of this Operation")
        self.input_meta[name] = dict()
        self.input_meta[name]['val'] = val
        if isinstance(shape, int):
            self.input_meta[name]['shape'] = tuple(shape)
        else:
            self.input_meta[name]['shape'] = shape
        self.input_meta[name]['src_indices'] = src_indices
        self.input_meta[name]['flat_src_indices'] = flat_src_indices
        self.input_meta[name]['units'] = units
        self.input_meta[name]['desc'] = desc
        self.input_meta[name]['tags'] = tags
        self.input_meta[name]['shape_by_conn'] = shape_by_conn
        self.input_meta[name]['copy_shape'] = copy_shape
        self.nargs += 1

    def add_output(
        self,
        name,
        val=1.0,
        shape=None,
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
    ):
        if name in self.input_meta.keys():
            raise KeyError(name +
                           " was already declared an input of this Operation")
        self.output_meta[name] = dict()
        self.output_meta[name]['val'] = val
        if isinstance(shape, int):
            self.output_meta[name]['shape'] = tuple(shape)
        else:
            self.output_meta[name]['shape'] = shape
        self.output_meta[name]['units'] = units
        self.output_meta[name]['res_units'] = res_units
        self.output_meta[name]['desc'] = desc
        self.output_meta[name]['lower'] = lower
        self.output_meta[name]['upper'] = upper
        self.output_meta[name]['ref'] = ref
        self.output_meta[name]['ref0'] = ref0
        self.output_meta[name]['res_ref'] = res_ref
        self.output_meta[name]['tags'] = tags
        self.output_meta[name]['tags'] = tags
        self.output_meta[name]['shape_by_conn'] = shape_by_conn
        self.output_meta[name]['copy_shape'] = copy_shape
        self.nouts += 1

    def declare_derivative(
        self,
        of,
        wrt,
        dependent=True,
        rows=None,
        cols=None,
        val=None,
        method='exact',
        step=None,
        form=None,
        step_calc=None,
    ):
        if (of, wrt) in self.derivatives_meta.keys():
            raise KeyError("Derivative {} wrt {}"
                           " already declared for this Operation".format(
                               of, wrt))
        self.derivatives_meta[of, wrt] = dict()
        self.derivatives_meta[of, wrt]['dependent'] = dependent
        self.derivatives_meta[of, wrt]['rows'] = rows
        self.derivatives_meta[of, wrt]['cols'] = cols
        self.derivatives_meta[of, wrt]['val'] = val
        self.derivatives_meta[of, wrt]['method'] = method
        self.derivatives_meta[of, wrt]['step'] = step
        self.derivatives_meta[of, wrt]['form'] = form
        self.derivatives_meta[of, wrt]['step_calc'] = step_calc
