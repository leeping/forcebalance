""" 

@mainpage

@section preface_sec Preface: How to use this document

The documentation for ForceBalance exists in two forms: a web page and
a PDF manual.  They contain equivalent content.  The newest versions
of the software and documentation, along with relevant literature, can
be found on the <a href=https://simtk.org/home/forcebalance/>SimTK website</a>.

\b Users of the program should read the <em>Introduction,
Installation</em>, <em>Usage</em>, and <em>Tutorial</em> sections on
the main page.

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

This section covers how to install ForceBalance and its companion
software GROMACS-X2.  Currently only Linux is supported, though
installation on other Unix-based systems (e.g. Mac OS) should also be
straightforward.

Importantly, note that <em>ForceBalance does not contain any simulation
software or methods for generating the reference data</em>.  Fitting
simulations are performed by interfacing ForceBalance with simulation
software like GROMACS, TINKER or OpenMM; reference data is obtained
from experimental measurements (consult the literature), or from
simulation / quantum chemistry software (for example, NWChem or Q-Chem).

I have provided a specialized version of GROMACS (dubbed version
4.0.7-X2) on the <a href=https://simtk.org/home/forcebalance/>SimTK
website</a> which interfaces with ForceBalance through the
forceenergymatch_gmxx2 module.  Although interfacing with unmodified
simulation software is straightforward, GROMACS-X2 is optimized
for our task and makes things much faster.  Soon, I will also
implement functions for grid-scale computation of reference energies
and forces using Q-Chem (a commercial software).  However, you should
be prepared to write some simple code to interface with a fitting
simulation or quantum chemistry software of your choice.  If you
choose to do so, please contact me as I would be happy to include your
contribution in the main distribution.

@section installing_forcebalance Installing ForceBalance

ForceBalance is an ordinary Python module, so if you know how to install Python
modules, you shouldn't have any trouble with this.

@subsection installing_forcebalance_prereq Prerequisites

The only required software for installing ForceBalance are Python and
NumPy.  ForceBalance also allows the usage of SciPy optimizers; they
aren't as effective as the internal optimizer but still often helpful
- if you want to use these, then SciPy is needed.  A few more packages
are required if you want to \ref create_doc.  Here is a list of Python
packages and software:

Needed for ForceBalance:
@li Python version 2.7.1
@li NumPy version 1.5.0
@li SciPy version 0.9.0 (optional; needed for some of the non-default optimizers)
Needed for making documentation:
@li Doxygen version 1.7.6.1
@li Doxypy plugin for Doxygen
@li LaTeX software like TeXLive

@subsection installing_forcebalance_install Installing

To install the package, first extract the tarball that you downloaded from the
webpage using the command:

@verbatim tar xvzf ForceBalance-[version].tar.gz @endverbatim

Upon extracting the distribution you will notice this directory structure:

@verbatim
<root>
  +- bin
  |   |- <Executable scripts>
  +- forcebalance
  |   |- <Python module files>
  +- test
  |   +- <ForceBalance example jobs>
  +- doc
  |   +- callgraph
  |   |   |- <Stuff for making a call graph>
  |   +- Images
  |   |   |- <Images for the website and PDF manual>
  |   |- mainpage.py (Contains most user documentation and this text)
  |   |- header.tex (Customize the LaTex documentation)
  |   |- add-tabs.py (Adds more navigation tabs to the webpage)
  |   |- DoxygenLayout.xml (Removes a navigation tab from the webpage)
  |   |- doxygen.cfg (Main configuration file for Doxygen)
  |   |- ForceBalance-Manual.pdf (PDF manual, but the one on the SimTK website is probably newer)
  |- PKG-INFO (Auto-generated package information)
  |- README.txt (Points to the SimTK website)
  |- setup.py (Python script for installation)
@endverbatim

To install the code into your default Python location, run this (you might need to be root):

@verbatim python setup.py install @endverbatim

Alternatively, you can do a local install by running:

@verbatim python setup.py install --prefix=/home/your_username/local_directory @endverbatim

where you would of course replace your_username and local_directory with your username and preferred install location.  The executable scripts will be placed into <tt>/home/your_username/local_directory/bin</tt> and the module will be placed into <tt>/home/your_username/local_directory/lib/python[version]/site-packages/forcebalance</tt>.

Note that if you do a local installation, for Python to recognize the newly installed module you may need to append your PYTHONPATH environment variable using a command like the one below:

@verbatim export PYTHONPATH=$PYTHONPATH:/home/your_username/local_directory/lib/python[version] @endverbatim

@section install_gmxx2 Installing GROMACS-X2

GROMACS-X2 contains major modifications from GROMACS 4.0.7.
Most importantly, it enables computation of the objective function
<a>and its analytic derivatives</a> for rapid force matching.  There
is also an implementation of the QTPIE fluctuating-charge polarizable
force field, and the beginnings of a GROMACS/Q-Chem interface
(carefully implemented but not extensively tested).  Most of the
changes were added in several new source files (less than ten): \c
qtpie.c, \c fortune.c, \c fortune_utils.c, \c fortune_vsite.c, \c
fortune_nb_utils.c, \c zmatrix.c and their corresponding header files,
and \c fortunerec.h for the force matching struct.  The name 'fortune'
derives from back when this code was called ForTune.

The force matching functions are turned on by calling \c mdrun with
the command line argument \c '-fortune' ; without this option, there
should be no impact on the performance of normal MD simulations.

ForceBalance interfaces with GROMACS-X2 by calling the program
with special options and input files; the objective function and
derivatives are computed and printed to output files.  The interface
is defined in \c fortune.c on the GROMACS side and \c
forceenergymatch_gmxx2 on the Python side.  ForceBalance needs to know
where the GROMACS-X2 executables are located, and this is specified
using the \c gmxpath option in the input file.

@subsection install_gmxx2_prerequisites Prerequisites for GROMACS-X2

GROMACS-X2 needs the base GROMACS requirements and several other libraries.

@li FFTW version 3.3
@li GLib version 2.0
@li Intel MKL library

GLib is the utility library provided by the GNOME foundation (the
folks who make the GNOME desktop manager and GTK+ libraries).
GROMACS-X2 requires GLib for its hash table (dictionary)
implementation.

GLib and FFTW can be compiled from source, but it is much easier if
you're using a Linux distribution with a package manager.  If you're
running Ubuntu or Debian, run <tt>sudo apt-get install libglib2.0-dev
libfftw3-dev</tt>; if you're using CentOS or some other distro with
the yum package manager, run <tt>sudo yum install glib2-devel.x86_64
fftw3-devel.x86_64</tt> (or replace \c x86_64 with \c i386 if you're
not on a 64-bit system.

GROMACS-X2 requires the Intel Math Kernel Library (MKL) for linear algebra.
In principle this requirement can be lifted if I rewrite the source
code, but it's a lot of trouble, plus MKL is faster than other
implementations of BLAS and LAPACK.

The Intel MKL can be obtained from the Intel website, free of charge
for noncommercial use.  Currently GROMACS-X2 is built with MKL version
10.2, which ships with compiler version 11.1/072 ; this is not the
newest version, but it can still be obtained from the Intel website
after you register for a free account.

After installing these packages, extract the tarball that you downloaded
from the website using the command:

@verbatim tar xvjf gromacs-[version]-x2.tar.bz2 @endverbatim

The directory structure is identical to GROMACS 4.0.7, but I added
some shell scripts. \c Build.sh will run the configure script using
some special options, compile the objects, create the executables and
install them; you will probably need to modify it slightly for your
environment.  The comments in the script will help further
with installation.

Don't forget to specify the install location of the GROMACS-X2 executables
in the ForceBalance input file!

@section create_doc Create documentation

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

\c OptimizePotential.py is the executable script that performs force
field optimization.  It requires an input file and a \ref
directory_structure.  \c MakeInputFile.py will create an example input
file that contains all options, their default values, and a short
description for each option.  There are plans to automatically
generate the correct input file from the provided directory structure,
but for now the autogenerated input file only provides the hardcoded
default options.

@todo The MakeInputFile.py script 

@section input_file Input file

A typical input file for ForceBalance might look something like this:

@verbatim
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
@endverbatim

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

@todo I need to make the option index.

@section directory_structure Directory structure

The directory structure for our example job would look like:

@verbatim
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
@endverbatim

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

\page tutorial Tutorial

This is a tutorial page, but if you haven't installed ForceBalance yet
please go to the Installation page first.  It is very much in process,
and there are many more examples to come.

\section tip4p Fitting a TIP4P potential using two fitting simulations
After everything is installed, go to the \c test directory in the distribution
and run:

@verbatim
cd 001_water12_tip4p/
OptimizePotential.py 01_bfgs_from_start.in | tee my_job.out
@endverbatim

If the installation was successful, you will get an output file
similar to \c 01_bfgs_from_start.out .  \c OptimizePotential.py begins
by taking the force field files from the \c forcefield directory and
the fitting simulations / reference data from the \c simulations
directory.  Then it calls GROMACS-X2 to compute the objective function
and its derivatives, uses the internal optimizer (based on BFGS) to
take a step in the parameter space, and repeats the process until
convergence criteria were made.

At every step, you will see output like:
@verbatim
  Step       |k|        |dk|       |grad|       -=X2=-     Stdev(X2)
    35   6.370e-01   1.872e-02   9.327e-02   2.48773e-01   1.149e-04

Sim: water12_sixpt   E_err(kJ/mol)=     8.8934 F_err(%)=    29.3236
Sim: water12_fourpt  E_err(kJ/mol)=    14.7967 F_err(%)=    39.2558
@endverbatim

The first line reports the step number, the length of the parameter
displacement vector, the gradient of the objective function, the
objective function itself, and the standard deviation of the last ten
\a improved steps in the objective function.  There are three kinds of
convergence criteria - the step size, the gradient, and the objective
function itself; all of them can be specified in the input file.

The next two lines report on the two fitting simulations in this job,
both of which use force/energy matching.  First, note that there are
two fitting simulations named \c water12_sixpt and \c water12_fourpt;
the names are because one set of geometries was sampled using a
six-site QTPIE force field, and the other was sampled using the TIP4P
force field.  However, the TIP4P force field is what we are fitting
for this ForceBalance job.  This shows how only one force field or
parameter set is optimized for each ForceBalance job, but the method
for sampling the configuration space is completely up to the user.
The geometries can be seen in the \c all.gro files, and the reference
data is provided in \c qdata.txt .  Note that the extra virtual sites
in \c water12_sixpt have been replaced with a single TIP4P site.

\c E_err and \c F_err report the RMS energy error in kJ/mol and the
percentage force error; note the significant difference in the quality
of agreement!  This illustrates that the quality of fit depends not
only on the functional form of the potential but also the
configurations that are sampled.  \c E_err and \c F_err are
'indicators' of our progress - that is, they are not quantities to be
optimized but they give us a mental picture of how we're doing.

The other input files in the directory use the same fitting
simulations, but they go through the various options of
reading/writing checkpoint files, testing gradients and Hessians by
finite difference, and different optimizers in SciPy.  Feel free to
explore some optimization jobs of your own - for example, vary the
weights on the fitting simulations and see what happens.  You will
notice that the optimizer will try very hard to fit one simulation but
not the other.

\page glossary Glossary

This is a glossary page containing useful terms for the discussion of
potential optimization.

@li <b> Empirical parameter </b> : Any adjustable parameter in the
empirical potential that affects the potential energy, such as the
partial charge on an atom, the equilibrium length of a chemical
bond, or the fraction of Hartree-Fock exchange in a density functional.

@li <b> Empirical Potential </b> : A formula that contains empirical
parameters and computes the potential energy of a collection of atoms.
Note that in ForceBalance this is used very loosely; even a DFT
functional may contain many empirical parameters, and ForceBalance has the
ability to optimize these as well!

@li <b> Fitting simulation </b> : A simulation protocol that allows
a force field to predict a physical quantity, paired with some reference
data.  The accuracy of the force field is given by its closeness 

@li <b> Force field </b> : This term is used interchangeably with
empirical potential; it is more prevalent in the biomolecular simulation
community.

@li <b> Functional form </b> : The mathematical functions in the force
field.  For instance, a CHARMM-type functional form has harmonic interactions
for bonds and angles, a cosine expansion for the dihedrals, Coulomb interactions
between point charges and Lennard-Jones terms for van der Waals interactions.

@li <b> Reference data </b> : In general, any accurately known
quantity that the force field is optimized to reproduce.  Reference
data can come from either theory or experiment.  For instance,
energies and forces from a high-level QM method can be used as
reference data (for instance, a CHARMM-type force field can be fitted
to reproduce forces from a DFT or MP2 calculation), or a force field
can be optimized to reproduce the experimental density of a liquid,
its enthalpy of vaporization or the solvation free energy of a solute.

@image latex ForceBalance.pdf "ForceBalance logo" width=2cm


"""

