from csdl.lang.operation import Operation
from csdl.utils.get_shape_val import get_shape_val
from csdl.utils.parameters import Parameters
from collections import OrderedDict


class CustomOperation(Operation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nargs = 0
        self.nouts = 0
        self.input_meta = OrderedDict()
        self.output_meta = OrderedDict()
        self.derivatives_meta = dict()
        self.parameters = Parameters()
        self.initialize()
        self.parameters.update(kwargs)

    def initialize(self):
        """
        User defined method to declare parameter values. Parameters are
        compile time constants (neither inputs nor outputs to the model)
        and cannot be updated at runtime. Parameters are intended to
        make a ``CustomOperation`` subclass definition generic, and therefore
        reusable. The example below shows how a ``CustomOperation`` subclass
        definition uses parameters and how the user can set parameters
        when constructing the example ``CustomOperation`` subclass. Note
        that the user never instantiates nor inherits directly from the
        ``CustomOperation`` base class.

        **Example**

        ```py
        # in this example, we inherit from ExplicitOperation, but
        # the user can also inherit from ImplicitOperation
        class Example(ExplicitOperation):
            def initialize(self):
                self.parameters.declare('in_name', types=str)
                self.parameters.declare('out_name', types=str)

            def define(self):
                # use parameters declared in ``initialize``
                in_name = self.parameters['in_name']
                out_name = self.parameters['out_name']

                self.add_input(in_name)
                self.add_output(out_name)
                self.declare_derivatives(out_name, in_name)

            # define run time behavior by defining other methods...

        # compile using Simulator imported from back end...
        sim = Simulator(
            Example(
                in_name='x',
                out_name='y',
            ),
        )
        ```
        """
        pass

    def define(self):
        """
        User defined method to define custom operation

        **Example**

        .. code-block:: python

            def define(self):
                self.add_input('Cl')
                self.add_input('Cd')
                self.add_input('rho')
                self.add_input('V')
                self.add_input('S')
                self.add_output('L')
                self.add_output('D')

                # declare derivatives of all outputs wrt all inputs
                self.declare_derivatives('*', '*'))
        """
        pass

    def add_input(
        self,
        name,
        val=1.0,
        shape=(1, ),
        src_indices=None,
        flat_src_indices=None,
        units=None,
        desc='',
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
    ):
        """
        Add an input to this operation.

        **Example**

        ```py
        class Example(ExplicitOperation):
            def define(self):
                self.add_input('Cl')
                self.add_input('Cd')
                self.add_input('rho')
                self.add_input('V')
                self.add_input('S')
                self.add_output('L')
                self.add_output('D')

            # ...

        class Example(ImplicitOperation):
            def define(self):
                self.add_input('a', val=1.)
                self.add_input('b', val=-4.)
                self.add_input('c', val=3.)
                self.add_output('x', val=0.)

            # ...
        ```
        """
        if name in self.input_meta.keys():
            raise KeyError(
                name +
                ' was already declared an input of this Operation')
        self.input_meta[name] = dict()
        self.input_meta[name]['shape'], self.input_meta[name][
            'val'] = get_shape_val(shape, val)
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
        shape=(1, ),
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
        distributed=None,
    ):
        """
        Add an output to this operation.

        **Example**


        ```py
        class Example(ExplicitOperation):
            def define(self):
                self.add_input('Cl')
                self.add_input('Cd')
                self.add_input('rho')
                self.add_input('V')
                self.add_input('S')
                self.add_output('L')
                self.add_output('D')

            # ...

        class Example(ImplicitOperation):
            def define(self):
                self.add_input('a', val=1.)
                self.add_input('b', val=-4.)
                self.add_input('c', val=3.)
                self.add_output('x', val=0.)

            # ...
        ```
        """
        if name in self.input_meta.keys():
            raise KeyError(
                name +
                ' was already declared an input of this Operation')
        self.output_meta[name] = dict()
        self.output_meta[name]['shape'], self.output_meta[name][
            'val'] = get_shape_val(shape, val)
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
        self.output_meta[name]['distributed'] = distributed
        self.nouts += 1

    def declare_derivatives(
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
        """
        Declare partial derivatives of each output with respect to each
        input (ExplicitOperation) or each residual associated with an output with
        respect to the input/output (ImplicitOperation).


        ```py
        class Example(ExplicitOperation):
            def define(self):
                self.add_input('Cl')
                self.add_input('Cd')
                self.add_input('rho')
                self.add_input('V')
                self.add_input('S')
                self.add_output('L')
                self.add_output('D')

                # declare derivatives of all outputs wrt all inputs
                self.declare_derivatives('*', '*')

            # ...

        class Example(ImplicitOperation):
            def define(self):
                self.add_input('a', val=1.)
                self.add_input('b', val=-4.)
                self.add_input('c', val=3.)
                self.add_output('x', val=0.)
                # declare derivative of residual associated with x
                # wrt x
                self.declare_derivatives('x', 'x')
                # declare derivative of residual associated with x
                # wrt a, b, c
                self.declare_derivatives('x', ['a','b','c'])

                self.linear_solver = ScipyKrylov()
                self.nonlinear_solver = NewtonSolver(solve_subsystems=False)

            # ...
        ```
        """
        # check argument types
        if not isinstance(of, (str, list)):
            raise TypeError(
                'of must be a string or list; {} given'.format(
                    type(of)))
        if not isinstance(wrt, (str, list)):
            raise TypeError(
                'wrt must be a string or list; {} given'.format(
                    type(wrt)))

        # user-provided lists of variables of wildcards
        of_list = []
        wrt_list = []
        if isinstance(of, str):
            if of == '*':
                of_list = list(self.output_meta.keys())
        elif isinstance(of, list):
            if any(x == '*' for x in of):
                of_list = list(self.output_meta.keys())
            else:
                of_list = of
        if isinstance(wrt, str):
            if wrt == '*':
                wrt_list = list(self.input_meta.keys())
        elif isinstance(wrt, list):
            if any(x == '*' for x in wrt):
                wrt_list = list(self.input_meta.keys())
            else:
                wrt_list = wrt

        # declare each derivative one by one
        if len(of_list) > 0 and len(wrt_list) > 0:
            for a in of_list:
                for b in wrt_list:
                    self.declare_derivatives(
                        a,
                        b,
                        dependent=dependent,
                        rows=rows,
                        cols=cols,
                        val=val,
                        method=method,
                        step=step,
                        form=form,
                        step_calc=step_calc,
                    )
        elif len(of_list) > 0:
            for a in of_list:
                self.declare_derivatives(
                    a,
                    wrt=wrt,
                    dependent=dependent,
                    rows=rows,
                    cols=cols,
                    val=val,
                    method=method,
                    step=step,
                    form=form,
                    step_calc=step_calc,
                )
        elif len(wrt_list) > 0:
            for b in wrt_list:
                self.declare_derivatives(
                    of,
                    b,
                    dependent=dependent,
                    rows=rows,
                    cols=cols,
                    val=val,
                    method=method,
                    step=step,
                    form=form,
                    step_calc=step_calc,
                )
        else:
            if (of, wrt) in self.derivatives_meta.keys():
                raise KeyError(
                    'Derivative {} wrt {} already declared'.format(
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
