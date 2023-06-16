from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup


setup(
    ext_modules=[
        Pybind11Extension(
            "clahe._clahe_impl", ["ext/_clahe_impl.cpp"], cxx_std=11),
    ],
)
