try:
    from csdl.lang.model import Model
except ImportError:
    pass
from csdl.lang.concatenation import Concatenation
from warnings import warn


def define_recursively(model: 'Model') -> 'Model':
    """
    Construct intermediate representation for each model in the
    hierarchy from the bottom up
    """
    for s in model.subgraphs:
        m = s.submodel
        _ = define_recursively(m)
        if m.defined is False:
            m.define()
            m.defined = True

    if model.defined is False:
        model.define()
        model.defined = True

        # check for empty model
        if model.registered_outputs == [] and model.subgraphs == []:
            if model.inputs == []:
                raise ValueError(
                    "This model doesn't do anything. Either register outputs, or create a model hierarchy."
                )
            else:
                warn(
                    "This model only creates inputs for the top level model"
                )

        # Check if all design variables are inputs
        input_names = [inp.name for inp in model.inputs]
        for name in model.design_variables.keys():
            if name not in input_names:
                raise KeyError(
                    "{} is not the CSDL name of an input to the model".
                    format(name))
        del input_names

        # Check that all outputs are defined
        # for output in self.sorted_nodes:
        for output in model.registered_outputs:
            if isinstance(output, Concatenation):
                if output.defined is False:
                    raise ValueError("Output not defined for {}".format(
                        repr(output)))

    return model
