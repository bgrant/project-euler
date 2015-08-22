from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension(name="_euler", sources=["_euler.pyx"])

setup(name="_euler",
      ext_modules=cythonize(ext))
