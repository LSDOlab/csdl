from setuptools import setup, find_packages

setup(
    name='csdl',
    packages=find_packages(),
    #packages=['csdl'],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'pint',
        'guppy3',
        'networkx>=2.7',
        'pytest',
        # 'scipy==1.8.0',
        # 'scipy',
        'scipy==1.10.1',
        'matplotlib',
    ],
    tests_require=['pytest'],
    version_config=True,
    setup_requires=['setuptools-git-versioning'],
)
