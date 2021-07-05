import os
import sys
from yapf.yapflib.yapf_api import FormatCode


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


CSDLPATH = sys.argv[1]
examples_directory = CSDLPATH + 'csdl/examples/valid/'
clean_examples_directory = CSDLPATH + 'docs/_build/html/examples/'
for filename in os.listdir(examples_directory):
    # if filename == "ex_explicit_with_subsystems.py":
    if filename[-3:] == '.py':
        filestr = open(examples_directory + filename, 'r').read().split('\n')
        trimlines(filestr)
        clean_filestr = ['from csdl_om import Simulator\n']
        for line in filestr[1:-1]:
            clean_filestr.append(unindent(line))
        trimlines(clean_filestr)
        outfile = open(clean_examples_directory + filename, 'w')
        FormatCode("\n".join(clean_filestr))
        outfile.writelines(clean_filestr)
