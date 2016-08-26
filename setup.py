from setuptools import find_packages, setup
from Cython.Build import cythonize


if __name__ == "__main__":
    setup(
        name="clahe",
        ext_modules=cythonize("**/*.pyx"),
        packages=find_packages(),
    )
