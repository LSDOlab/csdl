from setuptools import setup

setup(
    name='csdl',
    packages=[
        'csdl',
    ],
    # python_requires='>=3.10',
    install_requires=[
        'numpy',
        'dash==1.2.0',
        'dash-daq==0.1.0',
        'pint',
        'guppy3',
        'networkx',
    ],
    tests_require=['pytest'],
    version_config=True,
    setup_requires=['setuptools-git-versioning'],
)
