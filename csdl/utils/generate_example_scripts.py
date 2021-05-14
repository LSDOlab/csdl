from docstring_parser import parse
import re, os, inspect, importlib

# choose package
import csdl
pkg = csdl

# set paths
examples_module_path = 'csdl.examples'
examples_dir = 'examples/'
test_examples_subdir = examples_dir + 'invalid/'
doc_examples_subdir = examples_dir + 'valid/'


def camel_to_snake(name):
    return re.sub(
        '([a-z0-9])([A-Z])',
        r'\1_\2',
        re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name),
    ).lower()


def write_setup(f, obj, options):

    # setup and run problem
    superclasses = [sc.__name__ for sc in inspect.getmro(obj)]
    f.write('prob = Problem()\n')
    if 'Group' in superclasses:
        f.write('prob.model = ' + obj.__name__ + '(')
        if len(options) > 0:
            f.write('\n')
        for opt in options:
            f.write('    ' + opt + ',\n')
        f.write(')\n')
    elif 'Component' in superclasses:
        f.write('prob.model = Model()\nprob.model.add(')
        if len(options) > 0:
            f.write('\n    ')
        f.write('\'example\', ')
        if len(options) > 0:
            f.write('\n    ')
        f.write(obj.__name__ + '(')
        if len(options) > 0:
            f.write('\n')
        for opt in options:
            f.write('    ' + opt + ',\n')
        f.write('))\n')
    else:
        raise TypeError('Example class', obj.__name__,
                        'is not a Component or Group')
    f.write('prob.setup(force_alloc_complex=True)\nprob.run_model()')
    f.write('\n')


def get_example_file_name(obj, py_file_path):
    obj_name_snake_case = camel_to_snake(obj.__name__)
    prefix = ''
    example_filename = ''
    generate_example_script = False
    if obj_name_snake_case[:len('error_')] == 'error_':
        prefix = 'error_'
        generate_example_script = True
        example_filename = py_file_path.rsplit(
            '/', 1
        )[-1][:-len('.py')] + '_' + obj_name_snake_case[len(prefix):] + '.py'
    if obj_name_snake_case[:len('example_')] == 'example_':
        prefix = 'example_'
        generate_example_script = True
        example_filename = py_file_path.rsplit(
            '/', 1
        )[-1][:-len('.py')] + '_' + obj_name_snake_case[len(prefix):] + '.py'
    return generate_example_script, example_filename, prefix


def export_examples():
    # Python 3.9: use removesuffix
    package_path = inspect.getfile(pkg)[:-len('__init__.py')]
    examples_path = package_path + examples_dir
    test_examples_path = package_path + test_examples_subdir
    doc_examples_path = package_path + doc_examples_subdir

    for example in os.listdir(examples_path):
        suffix = '.py'
        if example[-len(suffix):] == suffix:
            py_file_path = (examples_path + example)

            # gather imports
            import_statements = []
            with open(py_file_path, 'r') as f:
                import_statements = []
                for line in f:
                    l = line.lstrip()
                    if re.match('import', l) or re.match('from', l):
                        import_statements.append(l)

            # Python 3.9: use removesuffix
            py_module = importlib.import_module(examples_module_path + '.' +
                                                example[:-len(suffix)])
            members = inspect.getmembers(py_module)
            for obj in dict(members).values():
                if inspect.isclass(obj):
                    generate_example_script, example_filename, prefix = get_example_file_name(
                        obj, py_file_path)

                    if generate_example_script == True:

                        # collect params
                        docstring = parse(obj.__doc__)
                        var_names = []
                        options = []
                        for param in docstring.params:
                            if param.arg_name == 'var':
                                var_names.append(param.description)
                            if param.arg_name == 'option':
                                options.append(param.description)

                        file_path = ''
                        if prefix == 'error_':
                            file_path = test_examples_path + example_filename
                        elif prefix == 'example_':
                            file_path = doc_examples_path + example_filename
                        if file_path != '':
                            print('writing to file', file_path)
                            with open(file_path, 'w') as f:

                                # write import statements
                                f.write('from openmdao.api import Problem\n')
                                for stmt in import_statements:
                                    f.write(stmt)
                                f.write('\n\n')

                                # write example class
                                source = re.sub('.*:param.*:.*\n', '',
                                                inspect.getsource(obj))
                                source = re.sub('\n.*"""\n.*"""', '', source)
                                f.write(source)
                                f.write('\n\n')

                                write_setup(f, obj, options)

                                # output values
                                if len(var_names) > 0:
                                    f.write('\n')
                                for var in var_names:
                                    f.write('print(\'' + var + '\', prob[\'' +
                                            var + '\'].shape)\n')
                                    f.write('print(prob[\'' + var + '\'])')
                                    f.write('\n')
                                f.close()


export_examples()
