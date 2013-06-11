===========
ForceBalance
===========

If you're reading this, you are probably looking for help. :)

There is documentation available online at:

http://www.simtk.org/home/forcebalance

=========
Quick help
=========

To build the package:

python setup.py build

To install the package:

python setup.py install

To install the package locally (i.e. if you don't have root permissions), try the following:

python setup.py install --user
python setup.py install --prefix=$HOME/.local
python setup.py install --prefix=/your/favorite/directory
export PYTHONPATH=/your/favorite/directory

The python packages "setuptools" and "virtualenv" are helpful in setting up a Python environment in your local directory.

To turn the package into a distributable .tar.bz2 file:

python setup.py sdist

Arthur was here
