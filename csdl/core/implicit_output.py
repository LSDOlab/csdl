from typing import Dict, Tuple
from csdl.core.variable import Variable
from csdl.core.output import Output
from csdl.utils.replace_output_leaf_nodes import replace_output_leaf_nodes


def replace_input_leaf_nodes(
    node: Variable,
    leaves: Dict[str, Variable],
):
    for dependency in node.dependencies:
        if isinstance(dependency,
                      Variable) and not isinstance(dependency, Output):
            if len(dependency.dependencies) > 0:
                node.remove_dependency_node(dependency)
                if dependency._id in leaves.keys():
                    node.add_dependency_node(leaves[dependency._id])
                else:
                    leaf = Variable(dependency.name,
                                    shape=dependency.shape,
                                    val=dependency.val)
                    leaf._id = dependency._id
                    node.add_dependency_node(leaf)
                    leaves[dependency._id] = leaf
        replace_input_leaf_nodes(dependency, leaves)


class ImplicitOutput(Output):
    """
    Class for creating an implicit output
    """
    def __init__(
        self,
        implicit_model,
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

        Parameters
        ----------
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

        self.implicit_model = implicit_model
        self.defined = False

    def define_residual(
        self,
        residual: Output,
    ):
        """
        Define the residual that must equal zero for this output to be
        computed

        Parameters
        ----------
        residual: Output
            Residual expression
        """
        if isinstance(residual, ImplicitOutput):
            raise TypeError(
                "The residual {} is an ImplicitOutput object, but residuals must not be ImplicitOutput objects"
                .format(repr(residual)))

        if residual is self:
            raise ValueError(
                "Variable for residual of {} cannot be self".format(
                    repr(residual)))

        if self.defined is True:
            raise ValueError(
                "Variable for residual of {} is already defined".format(
                    repr(self)))

        if residual.name in self.implicit_model.res_out_map.keys():
            raise ValueError("Residual {} already used for output {}".format(
                residual.name,
                self.implicit_model.res_out_map[residual.name],
            ))

        if self.shape != residual.shape:
            raise ValueError(
                "Shapes for implicit output {} and residual {} do not match".
                format(self.name, residual.name))
        # set flag so that this expression is a residual and not an
        # output of an ImplicitModel
        residual.is_residual = True

        # Replace leaf nodes of residual Variable object that
        # correspond to this ImplicitOutput node with Variable objects;
        replace_output_leaf_nodes(
            self,
            residual,
            Variable(self.name, shape=self.shape, val=self.val),
        )

        # Replace None with residual; now this output is defined
        self.implicit_model.out_res_map[self.name] = residual

        # Map residual to self
        self.implicit_model.res_out_map[residual.name] = self

        if residual not in self.implicit_model._model.registered_outputs:
            self.implicit_model.register_output(residual.name, residual)

    def define_residual_bracketed(
        self,
        residual: Output,
        x1=0.,
        x2=1.,
    ):
        """
        Define the residual that must equal zero for this output to be
        computed

        Parameters
        ----------
        residual: Variable
            Residual expression
        """
        if residual is self:
            raise ValueError("Variable for residual of " + self.name +
                             " cannot be self")
        if self.defined == True:
            raise ValueError("Variable for residual of " + self.name +
                             " is already defined")

        if self.shape != residual.shape:
            raise ValueError(
                "Shapes for implicit output {} and residual {} do not match".
                format(self.name, residual.name))

        # set flag so that this expression is a residual
        residual.is_residual = True

        # Replace leaf nodes of residual Variable object that
        # correspond to this ImplicitOutput node with DocInput objects;
        replace_output_leaf_nodes(
            self,
            residual,
            Variable(self.name, shape=self.shape, val=self.val),
        )

        # register expression that computes residual
        self.implicit_model.register_output(
            residual.name,
            residual,
        )

        # Replace None with residual; now this output is defined
        self.implicit_model.out_res_map[self.name] = residual

        # Map residual to self
        self.implicit_model.res_out_map[residual.name] = self

        # Store bracket values
        self.implicit_model.brackets_map[0][self.name] = x1
        self.implicit_model.brackets_map[1][self.name] = x2
