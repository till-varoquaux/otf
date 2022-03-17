Testing
-------

We use `tox <https://tox.wiki/en/latest/>`__ and
`poetry <https://python-poetry.org/>`__ to manage dependencies. You can
run the tests from the project direcotry (on a machine where tox and
python 3.10 are already installed) by calling:

.. code:: bash

   $ tox

Development environment
~~~~~~~~~~~~~~~~~~~~~~~

To setup your dev environment you should install the extra dev-tools
package in poetry.

.. code:: bash

   $ poetry install -E dev-tools

This will install all the tools required to work on the code base
including:

-  `Grip <https://github.com/joeyespo/grip>`__: Instant preview of the
   ``README.md`` file

.. code:: bash

   $ poetry run grip . 0.0.0.0:8080

-  `Pytest-watch <https://github.com/joeyespo/pytest-watch>`__:
   Continuous pytest runner

.. code:: bash

   $ poetry run ptw

-  `Black <https://black.readthedocs.io/en/stable/index.html>`__: Code
   formatter

.. code:: bash

   $ poetry run black --line-length=80 .

-  `Isort <https://pycqa.github.io/isort>`__: Sort the python imports

.. code:: bash

   $ poetry run isort .

-  `Flake8 <https://flake8.pycqa.org/en/latest>`__: Linting and style

.. code:: bash

   $ poetry run flake8 .

-  `Mypy <http://mypy-lang.org/>`__: Static type checker

.. code:: bash

   $ poetry run mypy src tests
