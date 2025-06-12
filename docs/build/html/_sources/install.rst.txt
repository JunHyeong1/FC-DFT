
Installation
*************
This page explains how to install FC-DFT code.

Requirements
=============
* GeomeTRIC
* PySCF
* PyAMG
* PyAMGCL
* GPU4PySCF (optional)

How to install
==============
Download the lastest version of FC-DFT from the repository::

1. Clone the repository::

    $ git clone https://github.com/Yang-Laboratory/FC-DFT.git

2. Change the directory to the cloned repository::

    $ cd FC-DFT

3. Install the package using `pip`::

    $ pip install .

4. Change directory to `$PYTHONPATH/fcdft/lib` and create build directory.
5. Go into `build` and compile the C shared libraries by `cmake ..` and `make`.