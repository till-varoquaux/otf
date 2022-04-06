Compiling workflows
===================


The :func:`~otf.Environment.workflow` decorator rewrites the code of the
underlying function to a state machine. The state machine is then compiled to a
big switch statement. This enables us to resume the execution of the workflow
from any ``await`` instruction. Even on another machine.

Let's consider the following workflow:

.. literalinclude:: examples/workflow.py


This would be compiled to this machine:

.. otf:graph:: examples/workflow.py

Which, in turn, would be compiled to these python cases:

.. otf:match_case:: examples/workflow.py

This is just a peak inside OTF's internal machinery.

The generated function returns :class:`~otf.Suspension` whenever it hits an
``await``. This :class:`~otf.Suspension` can be used to resume the workflow once
the result we're awaiting on is available. All the states are identified by
numbers. The public states (the one we can reenter the function through) have
positive id. The state ``0`` correspond to the start of the function, the state
``1`` to the continuation of the first ``await`` that appears in the source
function, the state ``2`` to the second ``await``...

All the functions and variables starting with ``_otf_`` are used for internal
purposes only. In order to understand this code here's a quick explainer of the
ones used here:

+ ``_otf_pos``: target state in the state machine.
+ ``_otf_val``: value that we were just awaiting on.
+ ``_otf_suspend()``: build a :class:`~otf.Suspension` that captures everything
  weneed to resume the workflow after awaiting on a variable.

