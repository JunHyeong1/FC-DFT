from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import fcdft

extensions = [
    Extension(
        "fcdft.lib.pbe_helper",
        ["fcdft/lib/pbe_helper.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "fcdft.lib.fcdft_helper",
        ["fcdft/lib/fcdft_helper.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        include_dirs=[numpy.get_include()],        
    )
]

setup(
    name='fcdft',
    version=fcdft.__version__,
    packages=find_packages(),
    include_package_data=True,
    ext_modules = cythonize(extensions),
)
