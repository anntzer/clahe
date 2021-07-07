from setupext import find_namespace_packages, setup


@setup.add_extensions
def make_extension():
    from pybind11.setup_helpers import Pybind11Extension
    yield Pybind11Extension(
        "clahe._clahe_impl", ["src/_clahe_impl.cpp"], cxx_std=11)


setup(
    name="clahe",
    description="Exact contrast-limited adaptive histogram equalization",
    long_description=open("README.rst", encoding="utf-8").read(),
    author="Antony Lee",
    url="https://github.com/anntzer/clahe",
    license="zlib",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: zlib/libpng License",
        "Programming Language :: Python :: 3",
    ],
    packages=find_namespace_packages("lib"),
    package_dir={"": "lib"},
    python_requires=">=3",
    setup_requires=[
        "pybind11>=2.6",
        "setuptools_scm>=3.3",  # fallback_version support.
    ],
    use_scm_version={
        "version_scheme": "post-release",
        "local_scheme": "node-and-date",
        "fallback_version": "0+unknown",
    },
    install_requires=[
        "numpy",
    ]
)
