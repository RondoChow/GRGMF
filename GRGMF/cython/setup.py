from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
ext_modules = [Extension("cython_func",["cython_func.pyx"])]
setup(
    name = "cython_func pyx",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    include_dirs=[numpy.get_include()]
)