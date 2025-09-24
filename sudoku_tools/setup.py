from setuptools import setup, Extension
import numpy

module = Extension(
    'sudoku_tools',
    sources=['main.c','src/jcz.c','src/sudoku_path.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-Wall", "-O3"]
)

setup(
    name='sudoku_tools',
    version='1.0',
    description='A list of utils for Solving Sudoku',
    ext_modules=[module]
)