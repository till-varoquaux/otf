On-the-fly distributed python workflows
=======================================

|PyPI| |PyPI - Python Version| |License: CC0-1.0| |Tests| |codecov|
|Documentation Status| |binder|

OTF is a framework to programmatically write, run and debug workflows.

Notebooks:
----------

OTF is still in its infancy. We currently mostly rely on notebook to demonstrate
how it works:

+ `Introduction <https://nbviewer.org/github/till-varoquaux/otf/blob/HEAD/docs/examples/introduction.ipynb>`_
+ `Serialisation <https://nbviewer.org/github/till-varoquaux/otf/blob/HEAD/docs/examples/serialisation.ipynb>`_


Installing
----------

OTF is currently in pre-alpha. If you really want to play with it you
can check install the latest build via:

.. code:: bash

   $ pip install -i https://test.pypi.org/simple/ otf

.. |PyPI| image:: https://img.shields.io/pypi/v/otf.svg
   :target: https://pypi.org/project/otf/
.. |PyPI - Python Version| image:: https://img.shields.io/pypi/pyversions/otf
.. |License: CC0-1.0| image:: https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg
   :target: http://creativecommons.org/publicdomain/zero/1.0/
.. |Tests| image:: https://github.com/till-varoquaux/otf/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/till-varoquaux/otf/actions/workflows/ci.yml
.. |codecov| image:: https://codecov.io/gh/till-varoquaux/otf/branch/main/graph/badge.svg?token=ahhI117oFg
   :target: https://codecov.io/gh/till-varoquaux/otf
.. |Documentation Status| image:: https://readthedocs.org/projects/otf/badge/?version=latest
   :target: https://otf.readthedocs.io/en/latest/?badge=latest
.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/till-varoquaux/otf/HEAD?labpath=docs%2Fexamples%2Fintroduction.ipynb
