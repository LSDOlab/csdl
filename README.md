# CSDL

This package contains the Computational System Design Language (CSDL),
an embedded domain specific language with Python as the host language
that automates the computation of partial derivatives for use in
gradient-based optimization.

This package provides the front end for the compiler for CSDL.
A back end for the compiler is required to run a model.
See [Installation](#installation) for instructions on how to install the
front end and back end for the compiler, as well as a list of back ends
available for CSDL.

## Installation

Clone this repository and install using `pip`.
Then install a backend (see list at the end of this section).

Note that this package requires an implementation of the CSDL compiler
backend to run.
Installing `csdl` alone will not be enough to run code written using
CSDL.

Packages that implement the compiler backend should depend on CSDL.
It is highly recommended to create a different environment for each
backend, as they may require different versions of `csdl`.

Below is a list of backend implementations available.

- [csdl_om](https://github.com/lsdolab/csdl_om), which compiles CSDL
  code to an [OpenMDAO](https://openmdao.org) problem.  No additional knowledge
  of the OpenMDAO API is required.

## Testing

CSDL provides tests that are independent of the backend implementation.
That is, tests are implementation-agnostic.

To run all tests for CSDL, navigate to the directory containing the
`csdl` package and run

```sh
pytest -s --backend <backend>
```

where `<backend>` is the name of the backend implementation, without
angle brackets. Quotes are not required.

For example, if using `csdl_om` as the backend, run

```sh
pytest -s --backend csdl_om
```

If all tests pass for one backend implementation, but not another, that
is an indication that the backend implementation that resulted in test
failures has a bug.

If the bug is an indication of lack of test coverage in the frontend,
new tests should be added to CSDL to ensure compliance among all backend
implementations.

Note that this command does not run any test specific to the backend
implementation.
