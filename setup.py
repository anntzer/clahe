from setuptools import find_packages, setup
from Cython.Build import cythonize


setup(
    name="clahe",
    ext_modules=cythonize("**/*.pyx"),
    packages=find_packages("lib"),
    package_dir={"": "lib"}
)
