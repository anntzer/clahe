from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class build_ext(build_ext):
    def finalize_options(self):
        from Cython.Build import cythonize
        import numpy as np

        self.distribution.ext_modules[:] = cythonize("**/*.pyx")
        for ext in self.distribution.ext_modules:
            ext.include_dirs = [np.get_include()]

        super().finalize_options()


setup(
    name="clahe",
    description="Exact contrast-limited adaptive histogram equalization",
    long_description=open("README.rst", encoding="utf-8").read(),
    author="Antony Lee",
    url="https://github.com/anntzer/clahe",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    cmdclass={"build_ext": build_ext},
    packages=find_packages("lib"),
    package_dir={"": "lib"},
    ext_modules=[Extension("", [])],
    python_requires=">=3",
    setup_requires=[
        "Cython",
        "numpy",
        "setuptools_scm",
    ],
    use_scm_version={  # xref __init__.py
        "version_scheme": "post-release",
        "local_scheme": "node-and-date",
        "write_to": "lib/clahe/_version.py",
    },
    install_requires=[
        "numpy",
    ]
)
