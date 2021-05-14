# CONTRIBUTING

## Configuration Management

The `omtools` repository is hosted on GitHub and uses Git as its Source
Control Manager.

It is recommended to create a username on GitHub and
[fork](https://guides.github.com/activities/forking/) the repository.

Before committing code, be sure to write tests and make sure all tests
pass.

Please begin all commit messages with a verb in the present tense, e.g.
"update docs", not "updated docs".

After working on a feature, push your changes to your fork, and then
issue a
[pull request](https://docs.github.com/en/free-pro-team@latest/desktop/contributing-and-collaborating-using-github-desktop/creating-an-issue-or-pull-request#creating-a-pull-request)
for `omtools`.

### Recommended Git Workflow

1. Fork repository on GitHub
2. Clone repository

```sh
git clone https://gihub.com/your_username/omtools.git
cd omtools
# You don't have to rename your remote repository or name it
# anything in particular, but this helps to avoid confusion
git remote rename origin fork
# You don't have to name the upstream repository anything in
# particular, but we will name it upstream in this guide
git remote add upstream https://https://gihub.com/lsdolab/omtools.git
```

3. Make a branch for a new feature and make changes

```sh
# on branch master
# pull changes from collaborators
git pull upstream master
# push changes from collaborators to your fork
git push fork master

# create and checkout a new branch for a feature you will be
# adding (exclude -b if the branch exists)
git checkout -b feature-name
# now on branch feature-name, free to make and commit changes
```

4. Make changes, write commits...
5. Push changes to YOUR repostory (`fork`)

```sh
# optional: push branch to YOUR repository
git push fork feature-name
git checkout master
# pull changes from collaborators just in case
git pull upstream master
# merge changes into master (RECOMMENDED: use --squash option to merge
# as single commit)
git merge feature-name
# using a tool like VSCode, open the project and review merge
# conflicts if any.
# Then edit the commit message describing the overall feature.

# push changes to YOUR fork's master branch
git push fork master
```

6. Issue [Pull
   Request](https://docs.github.com/en/free-pro-team@latest/desktop/contributing-and-collaborating-using-github-desktop/creating-an-issue-or-pull-request#creating-a-pull-request)
   on GitHub.
7. Review/Approve Pull Request if you are an `omtools` Maintainer.

## Contribute to Docs

`omtools` uses [Sphinx](https://www.sphinx-doc.org/en/master/) to
generate documentation automatically.
Sphinx uses `.rst` files to generate documentation.

**Please review [Writing Tests](#writing-tests) and [Writing
Examples](writing-examples) before contibuting to the docs.**

The directive `jupyter-execute` is available for embedding a code
module and output into documentation. See the documentation for
[jupyter-sphinx](https://jupyter-sphinx.readthedocs.io/en/latest/) for
details on how to use the directive.

The directive `code-include` is available for embedding code for a class
or function, as opposed to the entire module. See the documentation for
[sphinx-code-include](https://sphinx-code-include.readthedocs.io/en/latest/index.html)
for more details on how to use the directive.

It is recommended to write example code (see the `examples/` directory)
each time a feature is added.
Example scripts use the `ex_` prefix by convention.
Use existing files in `docs/_src_docs/examples/` as a guide for writing
`.rst` files.

To generate the docs, run `make html` in the `docs/` directory.

## Writing Examples

All example class definitions are located in the `examples/` directory.
Each example script by convention uses the `ex_` prefix.

`omtools` provides `utils/generate_exaple_scripts.py` for generating
example run files from `ot.Model` or `ot.ImplicitComponent` class
definitions in the `examples/` directory.
The resulting run scripts are written to `examples/documented/` to make
a distinction between files defining example classes to be used in
documentation, and example classes to be used only for testing (e.g.
classes that raise errors).

When defining example classes, give the example class an upper camel
case name that starts with `Example` if you intend to include it in
the docs, and `Error` if you plan to test that Python raises an error
appropriately.

## Writing Tests

`omtools` uses [pytest](https://docs.pytest.org/en/latest/) to run
tests.
Tests for `Expression` subclasses are located in `tests/` and tests for
stock `Component` subclasses are located in `comps/tests/`.

All tests run an example script from `examples/`.
In order for `pytest` to collect the tests, each test suite must be
written in a file with the `test_` prefix.
Each test within a test suite is defined as a function with the `test_`
prefix as well.

> NOTE: Not all example classes must show up in a run file that is
> inluded in the docs, but all (generated)example scripts should have at
> least one test script.

A test suite with a single test looks as follows.

```py
from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest
# Do not import an example script at the start of the file for a
# test suite

def test_example_with_valid_output():
    import omtools.examples.valid.ex_name_of_example as example

    # Test values
    np.testing.assert_approx_equal(example.prob['var.abs.name'], desired_val)
    np.testing.assert_almost_equal(example.prob['var.abs.name'], desired_val)

    # Test partials
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)

def test_example_that_raises_error():
    with pytest.raises(TypeError):
        import omtools.examples.invalid.ex_name_of_example
```

> NOTE: The example script defines a `Problem`, and runs the model upon
> import. There is no `if __name__ == "__main__"` clause in the example
> script.

Tests (functions with the `test_` prefix) are designed so that importing
a generated example script runs the example.
There should be one test per example, so each test should import an
example.
Do not import an example script in a test suite (file with `test_`
prefix).

To test values, use
[numpy.assert_approx_equal](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_approx_equal.html)
or
[numpy.assert_almost_equal](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_almost_equal.html).

To test partial derivatives, use
`openmdao.utils.assert_utils.assert_check_partials` and
`openmdao.utils.assert_utils.assert_no_approx_partials`.

## Defining Stock Components

Stock `Component` subclasses are located in the `comps/` directory.

Stock components must provide an API that can define inputs and outputs,
including their names.

To get a feel for what's required, take a look at the `Expression`
subclasses and `Component` subclasses in `std/` and `comps/`,
respectively.

## Defining Expression Subclasses

All expressions inherit from the `Expression` class.
`Expression` subclasses are stored in the `std/` directory of `omtools`.

The `Expression` class is stored as a node on a Directed Acyclic Graph
(DAG), which `omtools` uses to determine which `Component` objects to
construct in `openmdao`, and in which order.

The two objectives when defining an `Expression` subclass are

- Establish dependence on other Expression objects
- Extract options for the corresponding stock `Component` to construct
  after ther user-defined `omtools.Model.setup` runs
- Define function that constructs

Defining an `Expression` subclass is done as folows:

```py
from omtools.core.expressin import Expression

# use snake case to make the Expression look like a function to the user
class snake_case_expression(Expression):
    def initialize(self, expr, *other_args, **kwargs):
        # First, perform error checking
        if isinstance(expr, Expression) == False
            raise TypeError(expr, " is not an Expression object")

        # Second, establish dependence of this Expression on other
        # Expression objects
        self.add_dependency_node(expr)

        # First and second steps for variable number of Expression
        # objects
        for arg in other_args:
            if isinstance(arg, Expression) == False
                raise TypeError(arg, " is not an Expression object")
            # don't worry about an arg used multiple times
            self.add_dependency_node(arg)

        # Third, extract options for the Component subclass that will
        # be constructed from this Expression

        # Options can come from all of the Expression objects
        # (expr, other_args) as well as named arguments (kwargs)

        # ...

        # Fourth, define the function that Group calls after user defined
        # Group.setup method is called. This function requires a name
        # argument
        self.build = lambda: CorrespondingComponent(
            # CorrespondingComponent options defined in
            # CorrespondingComponent.initialize
        )
```

For more details, take a look at the Expression subclasses already
defined in `std/`.
