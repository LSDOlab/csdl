from setuptools import setup

setup(
    name='csdl',
    packages=[
        'csdl',
    ],
    install_requires=[
        'numpy',
        'pint',
        'guppy3',
        'matplotlib',
    ],
    version_config=True,
    setup_requires=['setuptools-git-versioning'],
)
