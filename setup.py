from setuptools import setup

setup(
    name='csdl',
    packages=[
        'csdl',
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'pint',
        'guppy3',
        'networkx',
        'pytest',
        'scipy',
        'matplotlib',
    ],
    tests_require=['pytest'],
    version_config=True,
    setup_requires=['setuptools-git-versioning'],
)
