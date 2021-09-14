import os
import sys
from yapf.yapflib.yapf_api import FormatCode
import inspect
import csdl
from io import StringIO
from contextlib import redirect_stdout

lang_pkg = csdl

# set paths
style_path = inspect.getfile(
    lang_pkg)[:-len('__init__.py')][:-len(lang_pkg.__name__ +
                                          '/')] + '.style.yapf'
"""
Examples are written as functions that take ``Simulator`` class as an
argument. This script makes new files containing only the contents of
the function definition so that Jupyter Sphinx can display and run the
examples using the reference implementation back end.
"""


def unindent(s: str) -> str:
    if len(s) > 0:
        if s[0] == '\t':
            return s[1:] + '\n'
        elif s[:4] == "    ":
            return s[4:] + '\n'
        else:
            return s + '\n'
    else:
        return s + '\n'


def trimlines(l):
    while l[-1] == "\n" or l[-1] == '':
        _ = l.pop()


do_not_run = [
    '__pycache__', '__init__.py', 'ex_dedup_simple.py',
    'ex_matmat_mat_vec_product.py',
    'ex_implicit_with_subsystems_visualize_internal_model.py'
]

CSDLPATH = sys.argv[1]
examples_directory = CSDLPATH + '/csdl/examples/valid/'
clean_examples_directory = CSDLPATH + '/docs/docs/worked_examples/'
print('Cleaning examples in {}'.format(examples_directory))
print('Clean examples will be in {}'.format(clean_examples_directory))
for filename in os.listdir(examples_directory):
    print(filename)
    if filename not in do_not_run:
        if filename[-3:] == '.py':
            filestr = open(examples_directory + filename,
                           'r').read().split('\n')
            trimlines(filestr)
            clean_filestr_lines = ['from csdl_om import Simulator\n']
            for line in filestr[1:-1]:
                clean_filestr_lines.append(unindent(line))
            trimlines(clean_filestr_lines)
            new_path = clean_examples_directory + filename[:-3] + '.mdx'
            print(new_path)
            outfile = open(new_path, 'w')
            clean_filestr, changed = FormatCode(
                "".join(clean_filestr_lines),
                style_config=style_path,
            )
            # print(changed)

            s = StringIO()
            with redirect_stdout(s):
                exec(clean_filestr)
            outputstr = s.getvalue()

            file_content = '```py\n' + clean_filestr + '```\n\n\n```' + outputstr + '```\n'
            outfile.writelines(file_content)
