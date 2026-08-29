Installation
============

Install the latest stable release from PyPI:

.. code-block:: bash

   python -m pip install gcisens

This is the recommended option for regular use and includes the dependencies
needed for plotting. Cloning the repository is only necessary if you want to
modify the source code, run the test suite or build the documentation locally:

.. code-block:: bash

   git clone https://github.com/Pxj04/gcisens.git
   cd gcisens
   python -m pip install -e ".[dev,docs]"
