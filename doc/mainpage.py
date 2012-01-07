""" 

@mainpage

@section preface_sec Preface: How to use this documentation

This documentation exists in two forms: a web page and a PDF manual.
They contain equivalent content.  The most up-to-date version can
always be found at https://simtk.org/home/forcebalance/ under the
'Documents' tab.

<em>Users</em> of the program should read the <em>Introduction,
Installation</em>, <em>Usage</em>, and <em>Tutorial</em> sections on
the main page (Chapter 1 in the PDF manual).

<em>Developers and contributors</em> should read the whole
Introduction chapter, including the <em>Program Layout</em> and
<em>Creating Documentation</em> sections.  The <em>API
documentation</em>, which constitutes the bulk of this documentation,
is mainly intended to be a reference for contributors who are writing
code.

@section intro_sec Introduction

Welcome to ForceBalance! :)

This is a <em> theoretical and computational chemistry </em> program
primarily developed by Lee-Ping Wang during graduate school at MIT and
postdoctoral work at Stanford.  Contributors to the code include
Jiahao Chen, Vijay Pande, Troy Van Voorhis, and Matt Welborn.

The function of this program is <em> automatic potential
optimization</em>; given a force field (a.k.a. empirical potential)
and a set of reference data, the program tunes the empirical
parameters in the potential such that it reproduces the reference data
as closely as possible.  

The philosophy of this program is to
present force field optimization in a unified and easily extensible
framework.  Since there are many different ways in theoretical
chemistry to compute the potential energy of a collection of atoms,
and similarly many types of reference data to fit these potentials to,
we do our best to provide an infrastructure which allows a user or a
contributor to fit any type of potential to any type of reference
data.

@section install_sec Installation

This section covers how to install ForceBalance.  Currently only Linux
is supported, though Mac OS X installation should also be straightforward.

@subsection Prerequisites

The only software you're really required to have are Python and NumPy.  However, there are some more packages you might like to have, especially if you're interested in creating your own documentation.  Note that you don't need to create your own documentation unless you're a developer, since you're already reading it!

@li Python version 2.7.1
@li NumPy version 1.5.0
@li SciPy version 0.9.0 (optional; needed for some of the non-default optimizers)
@li Doxygen version 1.7.6.1 (optional; for creating documentation)
@li Doxypy plugin for Doxygen (for creating documentation)
@li LaTeX software (for creating PDF documentation)

@subsection Installing

ForceBalance is an ordinary Python module, so if you know how to install Python
modules, you shouldn't have any trouble with this.

To install the package, first unzip the tarball that you downloaded from the
webpage using the command:

@code tar xvzf ForceBalance-[version].tar.gz @endcode

Upon extracting the distribution you will notice three directories:
'bin', 'doc', and 'forcebalance'.

The 'bin' directory contains all executable scripts and
programs, the 'forcebalance' directory contains all Python modules and
libraries, and the 'doc' directory contains documentation.

To install the code into your default Python location, run this (you might need to be root):

@code python setup.py install @endcode

Alternatively, you can do a local install by running:

@code python setup.py install --prefix=/home/your_username/local_directory @endcode

where you would of course replace your_username and local_directory with your username and preferred install location.  The executable scripts will be placed into <tt>/home/your_username/local_directory/bin</tt> and the module will be placed into <tt>/home/your_username/local_directory/lib/python[version]/site-packages/forcebalance</tt>.

Note that if you do a local installation, for Python to recognize the newly installed module you may need to append your PYTHONPATH environment variable using a command like the one below:

@code export PYTHONPATH=$PYTHONPATH:/home/your_username/local_directory/lib/python[version] @endcode

@section Glossary

It is useful to define several terms for the
sake of our discussion.

@li <b> Empirical Potential </b> : A formula that contains empirical
parameters and computes the potential energy of a collection of atoms.
Note that in ForceBalance we use this term very loosely; even a DFT
functional may contain many empirical parameters, and we have the
ability to optimize these as well!

@li <b> Force field </b> : This term is used interchangeably with
empirical potential; it is more prevalent in the biomolecular simulation
community.

@li <b> Functional form </b> : The mathematical functions in the force
field.  For instance, a CHARMM-type functional form has harmonic interactions
for bonds and angles, a cosine expansion for the dihedrals, Coulomb interactions
between point charges and Lennard-Jones terms for van der Waals interactions.

@li <b> Empirical parameter </b> : Any adjustable parameter in the
empirical potential that affects the potential energy, such as the
partial charge on an atom, the equilibrium length of a chemical
bond, or the fraction of Hartree-Fock exchange in a density functional.

@li <b> Reference data </b> : In general, any accurately known
quantity that we would like our force field to reproduce.  Reference
data can come from either theory or experiment.  For instance,
energies and forces from a high-level QM method can be used as
reference data (for instance, we could fit a CHARMM-type force field
to a DFT or MP2 calculation), or we can try to reproduce the experimental
density of a liquid, its enthalpy of vaporization or the solvation
free energy of a solute.

@section usage_sec Usage

To run this program, you may execute the scripts located in the 'bin'
directory.  The main script that performs force field optimization
is <tt>OptimizePotential.py</tt>.

<tt>OptimizePotential.py</tt> takes only one argument - an input file.
An example input file is given in the source distribution.

@section create_doc_sec Creating documentation



@image latex ForceBalance.pdf width=2cm

"""

