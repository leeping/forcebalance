""" 

@mainpage

@section preface_sec Preface: How to use this document

The documentation for ForceBalance exists in two forms: a web page and
a PDF manual.  They contain equivalent content.  The newest versions
of the software and documentation, along with relevant literature, can
be found on the <a href=https://simtk.org/home/forcebalance/>SimTK website</a>.

\b Users of the program should read the <em>Introduction,
Installation</em>, <em>Usage</em>, and <em>Tutorial</em> sections on
the main page (Chapter 1 in the PDF manual).

<b>Developers and contributors</b> should read the
Introduction chapter, including the <em>Program Layout</em> and
<em>Creating Documentation</em> sections.  The <em>API
documentation</em>, which describes all of the modules, classes and
functions in the program, is intended as a reference
for contributors who are writing code.

ForceBalance is a work in progress; using the program is nontrivial
and many features are still being actively developed.  Thus, users and
developers are highly encouraged to contact me through
the <a href=https://simtk.org/home/forcebalance/>SimTK website</a>, either by sending me email or posting to the
public forum, in order to get things up and running.

Thanks!

Lee-Ping Wang

@section intro_sec Introduction

Welcome to ForceBalance! :)

This is a <em> theoretical and computational chemistry </em> program
primarily developed by Lee-Ping Wang.  The full list of people who
made this project possible are given in the \ref credits.

The function of ForceBalance is <em>automatic potential
optimization</em>.  It addresses the problem of parameterizing
empirical potential functions, colloquially called <em>force
fields</em>.  Here I will provide some background, which for the sake
of brevity and readability will lack precision and details.  In the
future, this documentation will include literature citations which
will guide further reading.

@subsection background Background: Empirical Potentials

In theoretical and computational chemistry, there are many methods for
computing the potential energy of a collection of atoms and molecules
given their positions in space.  For a system of \a N particles, the
potential energy surface (or <em>potential</em> for short) is a
function of the \a 3N variables that specify the atomic coordinates.
The potential is the foundation for many types of atomistic
simulations, including molecular dynamics and Monte Carlo, which are
used to simulate all sorts of chemical and biochemical processes
ranging from protein folding and enzyme catalysis to reactions between
small molecules in interstellar clouds.

The true potential is given by the energy eigenvalue of the
time-independent Schrodinger's equation, but since the exact solution
is intractable for virtually all systems of interest, approximate
methods are used.  Some are <em>ab initio</em> methods ('from first
principles') since they are derived directly from approximating
Schrodinger's equation; examples include the independent electron
approximation (Hartree-Fock) and perturbation theory (MP2).  However,
the vast majority of methods contain some tunable constants or
<em>empirical parameters</em> which are carefully chosen to make the
method as accurate as possible.  Three examples: the widely used B3LYP
approximation in density functional theory (DFT) contains three
parameters, the semiempirical PM3 method has 10-20 parameters per
chemical element, and classical force fields have hundreds to
thousands of parameters.  All such formulations require an accurate
parameterization to properly describe reality.

\image html ladder_sm.png "An arrangement of simulation methods by accuracy vs. computational cost."
\image latex ladder.png "An arrangement of simulation methods by accuracy vs. computational cost." width=10cm

A major audience of ForceBalance is the scientific community that uses
and develops classical force fields.  These force fields do not use
the Schrodinger's equation as a starting point; instead, the potential
is entirely specified using elementary mathematical functions.  Thus,
the rigorous physical foundation is sacrificed but the computational
cost is reduced by a factor of millions, enabling atomic-resolution
simulations of large biomolecules on long timescales and allowing the
study of problems like protein folding.

In classical force fields, relatively few parameters may be determined
directly from experiment - for instance, a chemical bond may be
described using a harmonic spring with the experimental bond length
and vibrational frequency.  More often there is no experimentally
measurable counterpart to a parameter - for example, electrostatic
interactions are often described as interactions between pairs of
point charges on atomic centers, but the fractional charge assigned to
each atom has no rigorous experimental of theoretical definition.  To
complicate matters further, most molecular motions arise from a
combination of interactions and are sensitive to many parameters at
once - for example, the dihedral interaction term is intended to
govern torsional motion about a bond, but these motions are modulated
by the flexibility of the nearby bond and angle interactions as well
as the nonbonded interactions on either side.

\image html interactions_sm.png "An illustration of some interactions typically found in classical force fields."
\image latex interactions.png "An illustration of some interactions typically found in classical force fields." width=10cm

For all of these reasons, force field parameterization is difficult.
In the current practice, parameters are often determined by fitting to
results from other calculations (for example, restrained electrostatic
potential fitting (RESP) for determining the partial charges) or
chosen to reproduce experimental measurements which depend indirectly
on the parameters (for example, adjusting the partial charges on a
solvent molecule to reproduce the bulk dielectric constant.)
Published force fields have been modified by hand over decades to
maximize their agreement with experimental observations (for example,
adjusting some parameters in order to reproduce a protein crystal
structure) at the expense of reproducibility and predictive power.

@subsection mission_statement Purpose and brief description of this program 

Given this background, I can make the following statement.
<b>ForceBalance aims to advance the methods of empirical potential
development by applying a highly general and systematic process with
explicitly specified input data and mathematical optimization
algorithms, paving the way to higher accuracy potentials, improved
reproducibility of potential development, and well-defined scopes of
validity and error estimation for the parameters. </b>

At a high level, ForceBalance takes an empirical potential and a set of
reference data as inputs, and tunes the parameters such that the
reference data is reproduced as accurately as possible.  Examples of
reference data include energy and forces from high-level QM
calculations, experimentally known molecular properties
(e.g. polarizabilities and multipole moments), and experimentally
measured bulk properties (e.g. density and dielectric constant).

ForceBalance presents the problem of potential optimization in a
unified and easily extensible framework.  Since there are many
empirical potentials in theoretical chemistry and similarly many types
of reference data, significant effort is taken to provide an
infrastructure which allows a researcher to fit any type of
potential to any type of reference data.

Conceptually, a set of reference data (usually a physical quantity of
some kind), in combination with a method for computing the
corresponding quantity with the empirical potential, is called a
<b>fitting simulation</b>.  For example:

- A force field can predict the density of a liquid by running NPT
molecular dynamics, and this computed value can be compared against
the experimental density.

- A force field can be used to evaluate the energies and forces at
several molecular geometries, and these can be compared against
energies and forces from higher-level quantum chemistry calculations
using these same geometries.  This is known as <tt>force and energy
matching</tt>.

- A force field can predict the multipole moments and polarizabilities
of a molecule isolated in vacuum, and these can be compared against
experimental measurements.

Within the context of a fitting simulation, the accuracy of the force
field can be optimized by tuning the parameters to minimize the
difference between the computed and reference quantities.  One or more
fitting simulations can be combined to produce an aggregate
<b>objective function</b> whose domain is the <b>parameter space</b>.
This objective function, which typically depends on the parameters in
a complex way, is minimized using nonlinear optimization algorithms.
The result is a force field with high accuracy in all of the fitting
simulations.

\image html flowchart_sm.png "The division of the potential optimization problem into three parts; the force field, fitting simulations and optimization algorithm."
\image latex flowchart.png "The division of the potential optimization problem into three parts; the force field, fitting simulations and optimization algorithm." height=10cm

The problem is now split into three main components; the force field,
the fitting simulations, and the optimization algorithm.  ForceBalance
uses this conceptual division to define three classes with minimal
interdependence.  Thus, if a researcher wishes to explore a new
functional form, incorporate a new type of reference data or try a new
optimization algorithm, he or she would only need to contribute to one
branch of the program without having to restructure the entire code
base.

The scientific problems and concepts that this program is based upon
are further described in my Powerpoint presentations and publications,
which can be found on the <a href=https://simtk.org/home/forcebalance/>SimTK website</a>.

@section credits Credits

- Lee-Ping Wang is the principal developer and author.

- Troy Van Voorhis provided scientific guidance and many of
the central ideas as well as financial support.

- Jiahao Chen contributed the call graph generator, the QTPIE
fluctuating-charge force field (which Lee-Ping implemented into
GROMACS), the interface to the MOPAC semiempirical code, and many
helpful discussions.

- Matt Welborn contributed the parallelization-over-snapshots
functionality in the general force matching module.

- Vijay Pande provided scientific guidance and financial support,
and through the SimBios program gave this software a home on the Web
at the <a href=https://simtk.org/home/forcebalance/>SimTK website</a>.

- Todd Martinez provided scientific guidance and financial support.

\page installation Installation

This section covers how to install ForceBalance.  Currently only Linux
is supported, though Mac OS X installation should also be straightforward.

@section Prerequisites

The only required software for installing ForceBalance are Python and
NumPy.  However, <em>ForceBalance does not contain any simulation
software or methods for generating the reference data</em>.  Fitting
simulations are performed by interfacing ForceBalance with simulation
software like GROMACS, TINKER or OpenMM; reference data is obtained
from experimental measurements (consult the literature), or from
quantum chemistry software packages such as Q-Chem or TeraChem.

I have provided a heavily modified version of GROMACS (dubbed version
4.0.7-X2) on the <a href=https://simtk.org/home/forcebalance/>SimTK
website</a> which interfaces with ForceBalance through the
forceenergymatch_gmx module.  Although interfacing with unmodified
simulation software should be straightforward, GROMACS 4.0.7-X2 is
optimized for our task and makes things much faster.  Soon, I will
also implement functions for grid-scale computation of reference
energies and forces using Q-Chem (a commercial software).  However,
you should be prepared to write some simple code to interface with a
fitting simulation or quantum chemistry software of your choice.  If
you choose to do so, please contact me as I would be happy to include
your contribution in the main distribution.

Additionally, there are some more packages you might like to add,
especially if you'd like to \ref create_doc_sec.  Here is a list of
Python packages and software:

@li Python version 2.7.1
@li NumPy version 1.5.0
@li SciPy version 0.9.0 (optional; needed for some of the non-default optimizers)
@li Doxygen version 1.7.6.1 (optional; for creating documentation)
@li Doxypy plugin for Doxygen (for creating documentation)
@li LaTeX software (for creating PDF documentation)
@li GROMACS 4.0.7-X2 (for force and energy matching)
@li Q-Chem 3.2 (for computing reference energies and forces)

@section Installing

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

@section create_doc_sec Create documentation

This documentation is created by Doxygen with the Doxypy plugin.
To create new documentation or expand on what's here, follow the
examples on the source code or visit the Doxygen home page.

To create this documentation from the source files, go to the \c doc
directory in the distribution and run <tt> doxygen doxygen.cfg </tt>
to generate the HTML documentation and LaTeX source files.  Run the \c
add-tabs.py script to generate the extra navigation tabs for the HTML
documentation.  Then go to the \c latex directory and type in <tt> make
</tt> to build the PDF manual (You might need a LaTeX distribution for
this.)

\page usage Usage

This page describes how to use the ForceBalance software.

A good starting point for using this software package is to run
the scripts in the \c bin directory of the distribution.

\c OptimizePotential.py is the executable script that performs
force field optimization.  It requires an input file and a
\ref directory_structure.

@section input_file Input file

A typical input file for ForceBalance might look something like this:

@code
$options
jobtype                  bfgs
gmxpath                  /home/leeping/opt/gromacs-4.0.7-x2/bin
forcefield               water.itp
penalty_multiplicative   0.01
convergence_objective    1e-6
convergence_step         1e-6
convergence_gradient     1e-4
$end

$simulation
simtype                  forceenergymatch_gmx
name                     water12_gen1
weight                   1
efweight                 0.5
shots                    300
fd_ptypes                VSITE
fdhessdiag               1
covariance               0
$end

$simulation
simtype                  forceenergymatch_gmx
name                     water12_gen2
weight                   1
efweight                 0.5
shots                    300
fd_ptypes                VSITE
fdhessdiag               1
covariance               0
$end
@endcode

Global options for a ForceBalance job are given in the \c $options
section while the settings for each fitting simulation are given in
the \c $simulation sections.  At this time, these are the only two
section types.

The most important general options to note are: \c jobtype specifies
the optimization algorithm to use and \c forcefield specifies the
force field file name (there may be more than one of these).  The most
important simulation options to note are: \c simtype specifies the
type of fitting simulation and \c name specifies the simulation name
(must correspond to a subdirectory in \c simulations/ ).  All options
are explained in the Option Index.

@section directory_structure Directory structure

The directory structure for our example job would look like:

@code
<root>
  +- forcefield
  |   |- water.itp
  +- simulations
  |   +- water12_gen1
  |   |   |- all.gro (containing 300 geometries)
  |   |   |- qdata.txt
  |   |   |- shot.mdp
  |   |   |- topol.top
  |   +- water12_gen2
  |   |   |- all.gro (containing 300 geometries)
  |   |   |- qdata.txt
  |   |   |- shot.mdp
  |   |   |- topol.top
  |- input_file.in
@endcode

The top-level directory names \b forcefield and \b simulations are
fixed and cannot be changed.  \b forcefield contains the force field
files that you're optimizing, and \b simulations contains all of the
fitting simulations and reference data.  Each subdirectory in \b
simulations corresponds to a single fitting simulation, and its
contents depend on the specific kind of simulation and its
corresponding \c FittingSimulation subclass.

The \b temp directory is the temporary workspace of the program, and
the \b result directory is where the optimized force field files are
deposited after the optimization job is done.  These two directories
are created if they're not already there.

Note the force field file, \c water.itp and the two fitting
simulations \c water12_gen1 and \c water12_gen2 correspond to the
entries in the input file.  There are two energy and force matching
simulations here; each directory contains the relevant geometries (in
\c all.gro ) and reference data (in \c qdata.txt ).

\page glossary Glossary

This is a glossary page containing useful terms for the discussion of
potential optimization.

@li <b> Empirical Potential </b> : A formula that contains empirical
parameters and computes the potential energy of a collection of atoms.
Note that in ForceBalance this is used very loosely; even a DFT
functional may contain many empirical parameters, and ForceBalance has the
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
quantity that the force field is optimized to reproduce.  Reference
data can come from either theory or experiment.  For instance,
energies and forces from a high-level QM method can be used as
reference data (for instance, a CHARMM-type force field can be fitted
to reproduce forces from a DFT or MP2 calculation), or a force field
can be optimized to reproduce the experimental density of a liquid,
its enthalpy of vaporization or the solvation free energy of a solute.

@li <b> Fitting simulation </b> : A simulation protocol that allows
a force field to predict a physical quantity, paired with some reference
data.  The accuracy of the force field is given by its closeness 

@image latex ForceBalance.pdf "ForceBalance logo" width=2cm


"""

