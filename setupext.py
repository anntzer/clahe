"""setuptools helpers."""

from pathlib import Path

import setuptools
# find_namespace_packages itself bounds support to setuptools>=40.1.
from setuptools import Distribution, Extension, find_namespace_packages


__all__ = ["Extension", "find_namespace_packages", "setup"]


def add_extensions(ext_gen):
    """
    ::
        # Define extension modules with the ability to import setup_requires.
        @setup.add_extensions
        def make_extensions():
            import some_setup_requires
            yield Extension(...)
    """
    _build_ext_mixin._ext_gens.append(ext_gen)


class _build_ext_mixin:
    _ext_gens = []
    _ext_gens_called = False

    def finalize_options(self):
        if self._ext_gens and not self._ext_gens_called:
            self.distribution.ext_modules[:] = [
                ext for ext_gen in self._ext_gens for ext in ext_gen()]
            if len(self.distribution.ext_modules) == 1:
                ext, = self.distribution.ext_modules
                if (not ext.depends
                        and all(src.parent == Path("src")
                                for src in map(Path, ext.sources))):
                    ext.depends = ["setup.py", *Path("src").glob("*.*")]
            self._ext_gens_called = True
        super().finalize_options()


def _prepare_build_ext(kwargs):
    cmdclass = kwargs.setdefault("cmdclass", {})
    get = Distribution({"cmdclass": cmdclass}).get_command_class
    cmdclass["build_ext"] = type(
        "build_ext_with_extensions", (_build_ext_mixin, get("build_ext")), {})
    if _build_ext_mixin._ext_gens:
        # Don't tag wheels as dist-specific if no extension.
        kwargs.setdefault("ext_modules", [Extension("", [])])


def setup(**kwargs):
    _prepare_build_ext(kwargs)
    setuptools.setup(**kwargs)


setup.add_extensions = add_extensions
