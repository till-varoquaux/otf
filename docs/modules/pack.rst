``otf.pack``: Serialisation library
===================================

.. module:: otf.pack

.. testsetup::

   import otf


.. automodule:: otf.pack

API
---

.. _pack_api:

Text format
^^^^^^^^^^^^

The text format uses the same syntax as python, we just support a very
restricted subset of the language and have a different set of builtins (e.g.:
``ref`` and ``nan``).

.. todo:: The text format doesn't have a proper spec.

.. autofunction:: dump_text

Valid arguments for the **format** keyword of the :func:`dump_text` are:

.. data:: COMPACT

.. data:: PRETTY

.. data:: EXECUTABLE

--------------

.. autofunction:: load_text

Binary format
^^^^^^^^^^^^^

The binary format used by *OTF* is MessagePack with a couple of extensions. See
:ref:`Description of the binary format`

.. autofunction:: dump_bin

.. autofunction:: load_bin

Adding support for new types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: register


Converting between formats
^^^^^^^^^^^^^^^^^^^^^^^^^^

:mod:`otf.pack` is built upon the concept of *reducers* and *accumulator* (you
might know those as `Fold
<https://en.wikipedia.org/wiki/Fold_(higher-order_function)>`_ and `Unfold
<https://en.wikipedia.org/wiki/Unfold_(higher-order_function)>`_ if you're
functional programmer).

Documents are converted from one format to another by calling perform a *reduce*
on them from their source format with an *accumulator* for their destination
format. For instance if you wanted to go from the text representation of a
document to its binary representation you could do:

.. doctest::

  >>> otf.pack.reduce_text('[1, 2, 3, 4]', otf.pack.BinPacker())
  b'\x94\x01\x02\x03\x04'

This doesn't use the runtime representation as an intermediate value. This has
the advantage that it lets us introspect and fix documents that might rely on
constructors that don't exist anymore. Let's say you have a binary value that
won't load because it relies on constructor that doesn't exist anymore:

.. doctest::

  >>> bin_doc = (
  ...   b'\x94\x92\xc7\x14\x02mymath.angle:degrees\x00\x92\xd4\x03\x02Z\x92'
  ...   b'\xd4\x03\x02\xcc\xb4\x92\xd4\x03\x02\xcd\x01\x0e'
  ... )
  >>> otf.pack.load_bin(bin_doc)
  Traceback (most recent call last):
    ...
  LookupError: Constructor not found: 'mymath.angle'

You can convert that binary representation into a format that is easier to read
and can be fixed in an editor:

.. testsetup:: convert

  import otf

  bin_doc = (
    b'\x94\x92\xc7\x14\x02mymath.angle:degrees\x00\x92\xd4\x03\x02Z\x92\xd4\x03'
    b'\x02\xcc\xb4\x92\xd4\x03\x02\xcd\x01\x0e'
  )

.. doctest:: convert

  >>> print(otf.pack.reduce_bin(bin_doc, otf.pack.PrettyPrinter()))
  [
      mymath.angle(degrees=0),
      mymath.angle(degrees=90),
      mymath.angle(degrees=180),
      mymath.angle(degrees=270)
  ]


Let's say that you know that the new system uses angles represented as time on a
clock. You create a new binary value that can be loaded by that system even if
you don't have the ``mymath`` package installed on your machine:

.. doctest:: convert

  >>> doc = """[
  ...     mymath.clock_angle(hours=3),
  ...     mymath.clock_angle(hours=12),
  ...     mymath.clock_angle(hours=9),
  ...     mymath.clock_angle(hours=6),
  ... ]"""
  ...
  >>> new_bin_doc = otf.pack.reduce_text(doc, otf.pack.BinPacker())

-------------------

The defined reducers are:


.. autofunction:: reduce_runtime_value

.. autofunction:: reduce_text

.. autofunction:: reduce_bin


The Accumulators are:

.. autoclass:: RuntimeValueBuilder

.. autoclass:: CompactPrinter

.. autoclass:: PrettyPrinter

.. autoclass:: ExecutablePrinter

.. autoclass:: BinPacker


Utility functions
^^^^^^^^^^^^^^^^^

.. autofunction:: copy

.. autofunction:: dis

Description of the text format
------------------------------

*OTF*'s text representation is a subset of python. In fact we use the cpython
parser to read values. The easiest to specify the language is to use an ASDL
[ASDL97]_ representation based on the one for the `python grammar
<https://docs.python.org/3/library/ast.html#abstract-grammar>`_:

.. literalinclude:: grammar.asdl
  :language: asdl


An *OTF* serialised values consists of 3 parts:

+ **imports**: which are ignored by our parser
+ **bindings**: in form of `<name> = <value>`
+ **value**: an expression that can use any of the values defined in the
  bindings.

.. todo::
  This part needs to be filled in properly.

Description of the binary format
--------------------------------

**OTF**'s binary format is valid `MessagePack <https://msgpack.org>`_:

.. doctest::

  >>> import msgpack
  >>> packed = otf.pack.dump_bin([None, {1: 1}, -0.])
  >>> msgpack.unpackb(packed, strict_map_key=False)
  [None, {1: 1}, -0.0]

MessagePack allows for application specific `extension types
<https://github.com/msgpack/msgpack/blob/master/spec.md#extension-types>`_. Here
are the ones used by *OTF*:

Extension ``0``: Arbitrary precision ints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MessagePack only supports encoding integers in the :math:`[-2^{63}, 2^{64}-1]`
interval. Python, on the other hand, supports arbitrarily large integers. We
encode the integers outside the native MessagePack as 2's-complement,
little-endian payload_text inside an Extension Type of code 0:

.. doctest::

  >>> import msgpack
  >>> msgpack.unpackb(otf.pack.dump_bin(2**72))
  ExtType(code=0, data=b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01')


Extension ``1``: Shared references
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A shared value is represented by a reference to a node that was previously
visited while de-serialising the value:

.. doctest::

  >>> import msgpack
  >>> v = {'a': 'b'}
  >>> value = [v, v]
  >>> packed = otf.pack.dump_bin(value)
  >>> msgpack.unpackb(packed)
  [{'a': 'b'}, ExtType(code=1, data=b'\x03')]


The ``ExtType(...)`` translates to a reference with an offset of 3:

.. doctest::

  >>> print(otf.dump_text(value))
  [{'a': 'b'}, ref(3)]


This means that we are referencing the object that appeared three instructions
ago in the *OTF* bin code. The best way to see what we are talking about is to
disassemble the bin code:

.. testsetup:: shared_ref

  import otf
  v = {'a': 'b'}
  value = [v, v]
  packed = otf.pack.dump_bin(value)


.. doctest:: shared_ref

  >>> otf.pack.dis(packed)
  0001: LIST:
  0002:   MAP:
  0003:     'a'
  0004:     'b'
  0005:   REF(3)

Here we can clearly see that the ref points to ``MAP`` defined on instruction
``0002``.

In most cases it's easier to just translate the binary value to a text format:

.. doctest:: shared_ref

  >>> print(otf.pack.reduce_bin(packed, otf.pack.text.ExecutablePrinter()))
  _0 = {'a': 'b'}
  <BLANKLINE>
  [_0, _0]
  <BLANKLINE>

Extension ``2``: Custom constructors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**OTF** can be extended to support arbitrary types via
:func:`~otf.pack.register`. For those we need to save:

+ The full name of the constructor.
+ The names of keyword arguments.
+ The full list of values to pass back to the constructor.

.. doctest::

  >>> @otf.pack.register
  ... def _(c: complex):
  ...   return complex, (c.real,), {"imag": c.imag}
  >>>
  >>> packed = otf.pack.dump_bin(complex(1, .5))
  >>> otf.pack.dis(packed)
  0001: CUSTOM('complex:imag'):
  0002:   1.0
  0003:   0.5

We will call the constructor ``complex`` with one keyword argument ``imag`` and
the full list of arguments is ``[1.0, 0.5]``. The last arguments of this list
are the keyword arguments.

.. testsetup:: custom

  import otf

  @otf.pack.register
  def _(c: complex):
    return complex, (c.real,), {"imag": c.imag}

  packed = otf.pack.dump_bin(complex(1, .5))


.. doctest:: custom

  >>> print(otf.pack.reduce_bin(packed, otf.pack.text.PrettyPrinter()))
  complex(1.0, imag=0.5)

In raw MessagePack this is encoded as an array where the first element is an
``ExtType`` of code 2 and the rest of the list are the arguments:

.. doctest:: custom

  >>> import msgpack
  >>> msgpack.unpackb(packed)
  [ExtType(code=2, data=b'complex:imag'), 1.0, 0.5]



Extension ``3``: Interned constructor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The argument passed to the ``ExtType`` for custom constructors (the full name of
the constructor and the names of keyword arguments) is called the "shape" of a
custom constructors. In order to save space if the same shape appears twice in a
document the encoder uses the "interned_custom" instruction to reuse a previous
shape declaration.

.. doctest::

  >>> @otf.pack.register
  ... def _(c: complex):
  ...   return complex, (), {"real": c.real, "imag": c.imag}
  >>>
  >>> packed = otf.pack.dump_bin([complex(1, .5), complex(2)])
  >>> otf.pack.dis(packed)
  0001: LIST:
  0002:   CUSTOM('complex:real:imag'):
  0003:     1.0
  0004:     0.5
  0005:   INTERNED_CUSTOM(3):
  0006:     2.0
  0007:     0.0


.. testsetup:: interned

  import otf

  @otf.pack.register
  def _(c: complex):
    return complex, (), {"real": c.real, "imag": c.imag}

  packed = otf.pack.dump_bin([complex(1, .5), complex(2)])


This results in smaller MessagePack payload_text:

.. doctest:: interned

  >>> import msgpack, pprint
  >>>
  >>> pprint.pp(msgpack.unpackb(packed), width=60)
  [[ExtType(code=2, data=b'complex:real:imag'), 1.0, 0.5],
   [ExtType(code=3, data=b'\x03'), 2.0, 0.0]]


*OTF*'s serialised values should be self-descriptive; interning shapes
encourages clients to focus on the readability of their output format.

Design choice
-------------

Not prioritising speed
^^^^^^^^^^^^^^^^^^^^^^

The main focus is to provide an easy and safe way to serialise arbitrary python
values. Fast python serialisation libraries (e.g.: `msgpack
<https://github.com/msgpack/msgpack-python>`_ and `_pickle
<https://github.com/python/cpython/blob/main/Modules/_pickle.c>`_) significant
part should be written in C. Both MsgPack and Pickle offer fallbacks in pure
python. Maintaining a big a C stub (let alone a fallback in pure python) would
be a significant cost. *OTF* is still a young project and being nimble is
important.

If you want to save large amounts of data we recommend you use a format that is
tailored to the type of data you are saving. Some good examples would be:

+ `Arrow <https://arrow.apache.org/>`_ and `Parquet
  <https://parquet.apache.org/>`_ for column oriented data.
+ `Protocol Buffer <https://developers.google.com/protocol-buffers/>`_ for
  structured data (i.e.: data with a fixed schema).

Relative references
^^^^^^^^^^^^^^^^^^^

There are several ways to encode shared references. Here are a couple of
options we rejected:

+ **Marking**: You could "mark" the values that will be used later as shared
  references either by encoding them in a separate section and referring to them
  later or by having an instruction to save them the first time the encoder sees
  them. We chose to avoid this route because:

  1. It requires two different instructions (one to save the value and one to
     refer to it).
  2. You have to do an additional pass over the value you're about to encode to
     detect all the shared references.

+ **Absolute references**: instead of using an offset relative to the current
  position to encode the references we could have used a position relative to
  the start of the document. We chose to use a relative position because that
  allows us to insert an existing document without having to rewrite it.

Not allowing recursive values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recursive values are somewhat of a rare corner case and they can cause a lot of
headaches. The naive approach is to assume you can construct values in two
stages: initialise them as empty, create their children and then fill them.
This works fine if you look at a value like::

  value = [[],]
  value[0].append(0)

But now let's consider this case::

  value = ([],)
  value[0].append(0)

In this case you can't create ``value`` as an empty tuple and they fill
it... The corner cases cases get even more problematic when you consider that we
rely heavily on user provided serialisation function (via :func:`register`). We
don't actually know how much of its argument a constructor introspects...


Not relying on runtime representation or inheritance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the main philosophical differences between :mod:`pickle` and ``otf.pack``
is that we are a lot more explicit. ``otf.pack`` only supports values of types
that it explicitly knows how to handle. ``pickle``, on the other hand, will try
to serialise values based upon their runtime representation (e.g.: classes whose
``__dict__`` is picklable). These classes can, in turn, have instances of other
classes that will also be pickled based on their runtime representation. In
practice this ties the output of pickle pretty tightly with the exact version of
the code that generated it. On the other hand, by forcing clients to think
carefully about how values are serialised, we make it much easier to reason
about backward compatibility.

Using python and MessagePack for the serialisation languages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Learning a new syntax, even the syntax for a domain specific language is
tedious. We made it a point to re-use python for our text
representation. Similarly, MessagePack is one of the most widespread languages
to save unstructured data. Most languages already have libraries to read
MessagePack. Depending on how much custom types you use in your applications,
reading back values written by *OTF* in another language should be relatively
easy. The same definitely cannot be said for pickle.


.. [ASDL97] The Zephyr abstract syntax description language [`PDF
            <https://www.cs.princeton.edu/~appel/papers/asdl97.pdf>`_]
