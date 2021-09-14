from setuptools import setup

setup(
    name='csdl',
    packages=[
        'csdl',
    ],
    install_requires=[
        'numpy',
        'dash==1.2.0',
        'dash-daq==0.1.0',
        'pint',
        'guppy3',
        'matplotlib',
    ],
    version_config=True,
    setup_requires=['setuptools-git-versioning'],
)
