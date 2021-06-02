def pytest_addoption(parser):
    parser.addoption("--backend", action="store")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.backend
    if option_value is None:
        raise ValueError("Please provide a backend")
    if 'backend' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("backend", [option_value])
