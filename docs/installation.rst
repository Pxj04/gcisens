Installation
============

Requirements
------------

``gcisens`` requires Python 3.11 or newer. Using a virtual environment keeps
its scientific Python dependencies isolated from other projects.

Install from PyPI
-----------------

.. code-block:: bash

   python -m pip install gcisens

To use the plotting methods, no additional package is needed; Matplotlib is a
standard dependency.

Install from source
-------------------

For development or to test the current repository checkout:

.. code-block:: bash

   git clone https://github.com/Pxj04/gcisens.git
   cd gcisens
   python -m pip install -e ".[dev,docs]"

Verify the installation
-----------------------

.. code-block:: python

   import gcisens

   print(gcisens.__version__)

If this import succeeds, the library and its runtime dependencies are ready.
