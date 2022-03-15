OTF: On-the-fly python
=======================

.. toctree::
  :hidden:
  :maxdepth: 1

  license

The ``OTF`` framework makes it easy to write portable python functions on the
fly and to send them to remote machine. Here's an example of a portable python
function that can be used to count words in strings::

  import re
  import collections
  
  @function
  @environment(re=re, collections=collections)
  def count_words(document: str) -> collections.Counter[str]:
      """Count all the words in a string.
      """
      words = re.findall("\w+", document)
      return collections.Counter(words)


API Reference
-------------

The full API documentation is here:

.. toctree::
   :maxdepth: 2

   api
