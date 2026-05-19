import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

package = Extension(
    'libkrylov',
    sources=['krylov.pyx'],
    libraries=['blas', 'lapack', 'gfortran'],
    # library_dirs=['lib'],
    extra_objects=['lib/libkrylov.a'],
    include_dirs=['include', np.get_include()],
    runtime_library_dirs=['lib']
)
setup(ext_modules=cythonize([package]))