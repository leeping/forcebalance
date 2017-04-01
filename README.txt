#==================#
#|  ForceBalance  |#
#==================#

If you're reading this, you are probably looking for help. :D

The ForceBalance project website is located at
https://simtk.org/home/forcebalance

There is documentation available online in html at:
http://leeping.github.io/forcebalance/doc/html/index.html

You can also download the documentation in pdf format here:
http://leeping.github.io/forcebalance/doc/ForceBalance-Manual.pdf

#==================#
#|   Quick Help   |#
#==================#

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

#========================#
#| Published Parameters |#
#========================#

If you are here to obtain data for a published force field (e.g. AMBER-FB15), you can
find it in the Products folder.  Please email me at leeping (at) ucdavis (dot) edu
if you notice anything that should be added.

