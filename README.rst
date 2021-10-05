Exact contrast-limited adaptive histogram equalization
======================================================

|PyPI|

.. |PyPI|
   image:: https://img.shields.io/pypi/v/clahe.svg
   :target: https://pypi.python.org/pypi/clahe

As usual, install using pip:

.. code-block:: sh

   $ pip install clahe  # from PyPI
   $ pip install git+https://github.com/anntzer/clahe  # from Github

Run tests with unittest (or pytest).

This package uses a simple moving window implementation.  It may be worth
trying an implementation based on Perreault, S. & Hebert, P., *Median Filtering
in Constant Time* (2007).

See also the `discussion on the scikit-image issue tracker`__.

.. __: https://github.com/scikit-image/scikit-image/issues/2219#issuecomment-516791949
