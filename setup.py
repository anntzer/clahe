import numpy as np
from setuptools import find_packages, setup
from Cython.Build import cythonize


setup(
    name="clahe",
    ext_modules=cythonize("**/*.pyx"),
    include_dirs=[np.get_include()],
    packages=find_packages("lib"),
    package_dir={"": "lib"}
)
