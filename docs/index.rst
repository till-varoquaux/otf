OTF: On-the-fly python
=======================

.. toctree::
  :hidden:
  :maxdepth: 1

  license
  deep_dives

The ``OTF`` framework is a framework to write, run and debug complex machine
distributed workflows. Unlike the traditional DAG based frameworks (e.g.: `Luigi
<https://github.com/spotify/luigi>`_, `Airflow <https://airflow.apache.org/>`_,
`Kubeflow <https://www.kubeflow.org/>`_...) OTF allows you to use normal python
control flow statements in your workflow (e.g.: ``if`` and ``while``).


Features
--------

**Define by run control flow**

  Because we rewrite the workflows we can provide ``async/await`` semmantics
  without requiring a python interpreter to keep on running on a machine while
  we're awaiting for results.

  See :doc:`deep_dives/workflow_compilation` for an in-depth dive.

**Explicit serialisation**

  The :py:func:`~otf.function` and :py:func:`~otf.environement` decorators make
  it easy to write code that can be shipped over the network. Serialisation
  framework cannot always capture the intention of the author of the data/code
  they are serializing. Whereas `cloudpickle
  <https://github.com/cloudpipe/cloudpickle>`_ and `dill
  <https://github.com/uqfoundation/dill>`_ implicitly extend the `pickle
  <https://docs.python.org/3/library/pickle.html>`_ serialisation mechanism, we
  require code authors to tell us what will actually get sent.

..
  TODO:
  **Record-replay debugging**

API Reference
-------------

The full API documentation is here:

.. toctree::
   :maxdepth: 2

   api
