from setuptools import setup, Extension
import numpy

module = Extension(
    'sat_tools',
    sources=['main.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-Wall", "-O3"]
)

setup(
    name='sat_tools',
    version='1.0',
    description='A list of utils for Solving sat',
    ext_modules=[module]
)