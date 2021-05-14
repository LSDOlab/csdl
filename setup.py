from distutils.core import setup

setup(
    name='csdl',
    version='1',
    packages=[
        'csdl',
    ],
    install_requires=[
        'numpy',
        'dash==1.2.0',
        'dash-daq==0.1.0',
        'pint',
        'guppy3',
        'sphinx-rtd-theme',
        'sphinx-code-include',
        'jupyter-sphinx',
        'numpydoc',
    ],
)
