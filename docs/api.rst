API
===

.. testsetup::

   import math

   import otf

.. module:: otf

This is the full API reference for all public classes and functions.

Decorators
----------

.. autofunction:: function

.. autofunction:: environment


Runtime Classes
---------------

.. autoclass:: Closure
  :members:

.. autoclass:: Environment
  :members:

.. autoclass:: Suspension
  :members:

.. autoclass:: Workflow
  :members:

All the runtime classes support :mod:`pickle` but we also have our own
serialization format.

Serialisation
-------------

The otf serialisation format is designed to be read by humans. It is a subset of
python's syntax:

.. doctest::

   >>> otf.dump_text([1, 2, 3, 4, None])
   '[1, 2, 3, 4, None]'

   >>> otf.load_text("""
   ... # You can have comments in the values you load
   ... {
   ...    tuple([2, 1]): 5,
   ...    tuple([3, 1]): 13,
   ...    tuple([4, 1]): 65533,
   ... }
   ... """)
   {(2, 1): 5, (3, 1): 13, (4, 1): 65533}

The library is extensible and can handle arbitrary python values.

Shared references:

   >>> v = []
   >>> otf.dump_text([v, v])
   '[[], ref(1)]'

Non finite floats:

   >>> otf.dump_text([1., 2., -math.inf])
   '[1.0, 2.0, -inf]'

Adding support for new types:

   >>> import fractions
   ...
   >>> @otf.register
   ... def _(fraction: fractions.Fraction):
   ...   return fractions.Fraction, str(fraction)
   ...
   >>> otf.dump_text(fractions.Fraction(227, 73))
   "fractions.Fraction('227/73')"

Serialisation functions and classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: dump_text
  :noindex:

.. autofunction:: load_text
  :noindex:

.. autofunction:: register
  :noindex:

.. autoclass:: NamedReference

Modules
-------

.. toctree::
   :maxdepth: 1
   :glob:

   modules/*
