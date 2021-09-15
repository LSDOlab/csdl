from docstring_parser import parse
import pathlib
import re
import os
import inspect
import importlib

import csdl

lang_pkg = csdl

# choose package that implements CSDL
import csdl_om

# set paths
lang_example_class_definition_directory = inspect.getfile(
    lang_pkg)[:-len('__init__.py')] + 'examples/'
lang_test_exceptions_directory = \
     lang_example_class_definition_directory + 'invalid/'
lang_test_computations_directory = \
    lang_example_class_definition_directory + 'valid/'


def camel_to_snake(name):
    return re.sub(
        '([a-z0-9])([A-Z])',
        r'\1_\2',
        re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name),
    ).lower()


def write_run_phase(example_script_string, obj, options):
    # create simulator from model
    example_script_string += 'sim = Simulator(' + obj.__name__ + '('

    if len(options) > 0:
        example_script_string += '\n'
    for opt in options:
        example_script_string += '    ' + opt + ',\n'
    example_script_string += '))\n'

    # run simulation
    example_script_string += 'sim.run()\n'
    return example_script_string


def get_example_file_name(obj, py_file_path):
    obj_name_snake_case = camel_to_snake(obj.__name__)
    prefix = ''
    example_filename = None
    if obj_name_snake_case[:len('error_')] == 'error_':
        prefix = 'error_'
        example_filename = py_file_path.rsplit(
            '/', 1)[-1][:-len('.py')] + '_' + obj_name_snake_case[
                len(prefix):] + '.py'
    if obj_name_snake_case[:len('example_')] == 'example_':
        prefix = 'example_'
        example_filename = py_file_path.rsplit(
            '/', 1)[-1][:-len('.py')] + '_' + obj_name_snake_case[
                len(prefix):] + '.py'
    return example_filename, prefix


def export_examples(
    pkg_with_example_class_definitions,
    output_directory_example_exceptions,
    output_directory_example_computations,
):
    # Python 3.9: use removesuffix
    example_class_definition_module = pkg_with_example_class_definitions.__name__ + '.examples'
    example_class_definition_directory = inspect.getfile(
        pkg_with_example_class_definitions
    )[:-len('__init__.py')] + 'examples/'
    for examples_file in os.listdir(example_class_definition_directory):
        suffix = '.py'
        if examples_file[-len(suffix):] == suffix:
            example_classes_file_path = (
                example_class_definition_directory + examples_file)

            # gather imports
            import_statements = []
            with open(example_classes_file_path, 'r') as f:
                import_statements = []
                for line in f:
                    line_text = line.lstrip()
                    if re.match('import', line_text) or re.match(
                            'from', line_text):
                        import_statements.append(line_text)

            # Python 3.9: use removesuffix
            lang_examples_module = importlib.import_module(
                example_class_definition_module + '.' +
                examples_file[:-len(suffix)])
            members = inspect.getmembers(lang_examples_module)
            for obj in dict(members).values():
                if inspect.isclass(obj):
                    print(
                        'Generating example script for class {} from file {}'
                        .format(obj.__name__, examples_file))
                    example_run_file_name, prefix = \
                        get_example_file_name(
                            obj, example_classes_file_path,
                        )

                    if example_run_file_name is not None:
                        # collect params
                        docstring = parse(obj.__doc__)
                        var_names = []
                        options = []
                        for param in docstring.params:
                            if param.arg_name == 'var':
                                var_names.append(param.description)
                            if param.arg_name == 'option':
                                options.append(param.description)

                        example_run_file_path = None
                        if prefix == 'error_':
                            example_run_file_path = \
                                output_directory_example_exceptions \
                                + example_run_file_name
                        elif prefix == 'example_':
                            example_run_file_path = \
                                output_directory_example_computations \
                                + example_run_file_name
                        print('generating code for file',
                              example_run_file_path)

                        example_script_string = ''
                        for stmt in import_statements:
                            example_script_string += stmt
                        example_script_string += '\n\n'

                        # write example class
                        source = re.sub('.*:param.*:.*\n', '',
                                        inspect.getsource(obj))
                        source = re.sub('\n.*"""\n.*"""', '', source)
                        example_script_string += source
                        example_script_string += '\n\n'

                        example_script_string = write_run_phase(
                            example_script_string, obj, options)

                        # output values
                        if len(var_names) > 0:
                            example_script_string += '\n'
                        for var in var_names:
                            example_script_string += 'print(\'' + var + '\', sim[\'' + var + '\'].shape)\n'
                            example_script_string += 'print(sim[\'' + var + '\'])'
                            example_script_string += '\n'

                        example_script_lines = [
                            'def example(Simulator):'
                        ] + [('    ' + line) for line in
                             example_script_string.split('\n')]
                        if prefix == 'example_':
                            example_script_lines += ['    return sim']
                        with open(example_run_file_path, 'w') as f:
                            print('writing to file',
                                  example_run_file_path)
                            f.write('\n'.join(example_script_lines))
                            f.close()


# export_examples(
#     lang_pkg,
#     lang_test_exceptions_directory,
#     lang_test_computations_directory,
# )

# generate run scripts from examples in CSDL package using this
# implementation of CSDL
print('START: implementation-agnostic examples')
pathlib.Path(lang_test_exceptions_directory).mkdir(parents=True,
                                                   exist_ok=True)
pathlib.Path(lang_test_computations_directory).mkdir(parents=True,
                                                     exist_ok=True)
export_examples(
    lang_pkg,
    lang_test_exceptions_directory,
    lang_test_computations_directory,
)
print('END: implementation-agnostic examples')
