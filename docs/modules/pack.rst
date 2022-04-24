``otf.pack``: Serialisation library
===================================

.. automodule:: otf.pack
  :members:

FAQ:
----

Is it fast?
  No. The main focus is to provide an easy and safe way to serialise arbitrary
  python values. If you want to save large amounts of data we recommend you
  use a format that is tailored to the type of data you are saving. Some good
  examples would be:

  + `Arrow <https://arrow.apache.org/>`_ and
    `Parquet <https://parquet.apache.org/>`_ for column oriented data.
  + `Protocol Buffer <https://developers.google.com/protocol-buffers/>`_ for
    structured data (i.e.: data with a fixed schema).

..
  Why are references relative?
    Referential transparency

  Why no inheritance?
    Because they cause problems...
