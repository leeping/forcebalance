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

#=======================#
#| Citing ForceBalance |#
#=======================#

Published work that utilizes ForceBalance should include the following references:

Wang L-P, Chen J and Van Voorhis T. "Systematic Parametrization of Polarizable Force Fields from Quantum Chemistry Data." 
J. Chem. Theory Comput. 2013, 9, 452-460. DOI: 10.1021/ct300826t

Wang L-P, Martinez TJ and Pande VS. "Building Force Fields: An Automatic, Systematic, and Reproducible Approach." 
J. Phys. Chem. Lett. 2014, 5, 1885-1891. DOI: 10.1021/jz500737m

Qiu Y, Nerenberg PS, Head-Gordon T and Wang L-P. "Systematic Optimization of Water Models Using Liquid/Vapor Surface Tension Data."
J. Phys. Chem. B 2019, 123, 7061-7073. DOI: 10.1021/acs.jpcb.9b05455

#===========================#
#| Funding Acknowledgement |#
#===========================#

The development of this code has been supported in part by the following grants and awards:

NIH Grant R01 AI130684-02
ACS-PRF 58158-DNI6
NIH Grant U54 GM072970
Open Force Field Consortium
