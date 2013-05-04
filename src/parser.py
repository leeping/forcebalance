""" @package parser Input file parser for ForceBalance jobs.  Additionally, the location for all default options.

Although I will do my best to write good documentation,
for many programs the input parser becomes the most up-to-date
source for documentation.  So this is a great place to write
lots of comments for those who implement new functionality.

There are two types of sections for options - GENERAL and TARGET.
Since there can be many fitting targets within a single job (i.e. we
may wish to fit water trimers and hexamers, which constitutes two
fitting targets) the input is organized into sections, like so:

$options\n
gen_option_1 Big\n
gen_option_2 Mao\n
$target\n
tgt_option_1 Sniffy\n
tgt_option_2 Schmao\n
$target\n
tgt_option_1 Nifty\n
tgt_option_2 Jiffy\n
$end

In this case, two sets of target options are generated in addition to the general option.

(Note: "Target" used to be called "Simulation".  Backwards compatibility is maintained.)

Each option is meant to be parsed as a certain variable type.

- String option values are read in directly; note that only the first two words in the line are processed
- Some strings are capitalized when they are read in; this is mainly for function tables like OptTab and TgtTab
- List option types will pick up all of the words on the line and use them as values,
plus if the option occurs more than once it will aggregate all of the values.
- Integer and float option types are read in a pretty straightforward way
- Boolean option types are always set to true, unless the second word is '0', 'no', or 'false' (not case sensitive)
- Section option types are meant to treat more elaborate inputs, such
as the user pasting in output parameters from a previous job as input,
or a specification of internal coordinate system.  I imagine that for
every section type I would have to write my own parser.  Maybe a
ParsTab of parsing functions would work. :)

To add a new option, simply add it to the dictionaries below and give it a default value if desired.
If you add an entirely new type, make sure to implement the interpretation of that type in the parse_inputs function.

@author Lee-Ping Wang
@date 11/2012
"""

import os
import re
import sys
import itertools
import traceback
from nifty import printcool, printcool_dictionary, which
from copy import deepcopy
from collections import OrderedDict

## Default general options.
## Note that the documentation is included in part of the key; this will aid in automatic doc-extraction. :)
## In the 5-tuple we have: Default value, priority (larger number means printed first), short docstring, description of scope, list of filter strings for pulling out pertinent targets (MakeInputFile.py)
gen_opts_types = {
    'strings' : {"gmxpath"      : (which('mdrun'), 60, 'Path for GROMACS executables (if not the default)', 'All targets that use GROMACS', ['GMX']),
                 "gmxsuffix"    : ('', 60, 'The suffix of GROMACS executables', 'All targets that use GROMACS', ['GMX']),
                 "tinkerpath"   : (which('testgrad'), 60, 'Path for TINKER executables (if not the default)', 'All targets that use TINKER', ['TINKER']),
                 "penalty_type" : ("L2", 100, 'Type of the penalty, L2 or Hyp in the optimizer', 'All optimizations'),
                 "scan_vals"    : (None, -100, 'Values to scan in the parameter space, given like this: -0.1:0.1:11', 'Job types scan_mvals and scan_pvals'),
                 "readchk"      : (None, -50, 'Name of the restart file we read from', 'Restart jobtype "newton" with "writechk" set'),
                 "writechk"     : (None, -50, 'Name of the restart file we write to (can be same as readchk)', 'Main optimizer'),
                 "ffdir"        : ('forcefield', 100, 'Directory containing force fields, relative to project directory', 'All'),
                 "amoeba_polarization"        : ('direct', 0, 'The AMOEBA polarization type, either direct, mutual, or nonpolarizable.', 'Targets in OpenMM / TINKER that use the AMOEBA force field', ['OPENMM','TINKER'])
                 },
    'allcaps' : {"jobtype"      : ("single", 200, 'The calculation type, defaults to a single-point evaluation of objective function.', 
                                   'All (important); choose "single", "gradient", "hessian", "newton" (Main Optimizer), "bfgs", "powell", "simplex", "anneal", "genetic", "conjugategradient", "scan_mvals", "scan_pvals", "fdcheck[gh]"'),
                 },
    'lists'   : {"forcefield"     : ([],  200, 'The names of force fields, corresponding to directory forcefields/file_name.(itp,xml,prm,frcmod,mol2)', 'All (important)'),
                 "scanindex_num"  : ([], -100, 'Numerical index of the parameter to scan over', 'Job types scan_mvals and scan_pvals'),
                 "scanindex_name" : ([], -100, 'Parameter name to scan over (should convert to a numerical index)', 'Job types scan_mvals and scan_pvals')
                 },
    'ints'    : {"maxstep"      : (100, 50, 'Maximum number of steps in an optimization', 'Main Optimizer'),
                 "objective_history"  : (2, 20, 'Number of good optimization steps to average over when checking the objective convergence criterion', 'Main Optimizer (jobtype "newton")'),
                 "wq_port"   : (0, 0, 'The port number to use for Work Queue', 'Targets that use Work Queue (advanced usage)'),
                 },
    'bools'   : {"backup"           : (1,  10,  'Write temp directories to backup before wiping them'),
                 "writechk_step"    : (1, -50,  'Write the checkpoint file at every optimization step'),
                 "have_vsite"       : (0, -150, 'Specify whether there are virtual sites in the simulation (being fitted or not).  Enforces calculation of vsite positions.', 'Experimental feature in ESP fitting', ['ABINITIO']),
                 "constrain_charge" : (0,  10,  'Specify whether to constrain the charges on the molecules.', 'Printing the force field (all calculations)'),
                 "print_gradient"   : (1,  20,  'Print the objective function gradient at every step', 'Main Optimizer'),
                 "logarithmic_map"  : (0, -150, 'Optimize in the space of log-variables', 'Creating the force field (all calculations, advanced usage)'),
                 "print_hessian"    : (0,  20, 'Print the objective function Hessian at every step', 'Main Optimizer'),
                 "print_parameters" : (1,  20, 'Print the mathematical and physical parameters at every step', 'Main Optimizer'),
                 "normalize_weights": (1, 100, 'Normalize the weights for the fitting targets', 'Objective function (all calculations)'),
                 "verbose_options"  : (0, 150, 'Set to false to suppress printing options that are equal to their defaults', 'Printing output'),
                 "rigid_water"      : (0, -150, 'Perform calculations using rigid water molecules.', 'Currently used in AMOEBA parameterization (advanced usage)', ['OPENMM','TINKER']),
                 "use_pvals"        : (0, -150, 'Bypass the transformation matrix and use the physical parameters directly', 'Creating the force field; advanced usage, be careful.'),
                 "asynchronous"     : (0, 0, 'Execute Work Queue tasks and local calculations asynchronously for improved speed', 'Targets that use Work Queue (advanced usage)'),
                 },
    'floats'  : {"trust0"                 : (1e-1, 100, 'Levenberg-Marquardt trust radius; set to negative for nonlinear search', 'Main Optimizer'),
                 "mintrust"               : (0.0,   10, 'Minimum trust radius (if the trust radius is tiny, then noisy optimizations become really gnarly)', 'Main Optimizer'),
                 "convergence_objective"  : (1e-4, 100, 'Convergence criterion of objective function (in MainOptimizer this is the stdev of X2 over [objective_history] steps)', 'Main Optimizer'),
                 "convergence_gradient"   : (1e-4, 100, 'Convergence criterion of gradient norm', 'Main Optimizer'),
                 "convergence_step"       : (1e-4, 100, 'Convergence criterion of step size (just needs to fall below this threshold)', 'Main Optimizer'),
                 "eig_lowerbound"         : (1e-4,  10, 'Minimum eigenvalue for applying steepest descent correction', 'Main Optimizer'),
                 "lm_guess"               : (1.0,    9, 'Guess value for bracketing line search in trust radius algorithm', 'Main Optimizer'),
                 "finite_difference_h"    : (1e-3,  50, 'Step size for finite difference derivatives in many functions', 'pretty much everywhere'),
                 "penalty_additive"       : (0.0,   55, 'Factor for additive penalty function in objective function', 'Objective function, all penalty types'),
                 "penalty_multiplicative" : (0.0,   55, 'Factor for multiplicative penalty function in objective function', 'Objective function, all penalty types'),
                 "penalty_alpha"          : (1e-3,  53, 'Extra parameter for fusion penalty function.  Dictates position of log barrier or L1-L0 switch distance', 
                                             'Objective function, FUSION_BARRIER or FUSION_L0 penalty type, advanced usage in basis set optimizations'),
                 "penalty_hyperbolic_b"   : (1e-6,  54, 'Cusp region for hyperbolic constraint; for x=0, the Hessian is a/2b', 'Penalty type L1'),
                 "adaptive_factor"        : (0.25,  10, 'The step size is increased / decreased by up to this much in the event of a good / bad step; increase for a more variable step size.', 'Main Optimizer'),
                 "adaptive_damping"       : (0.5,   10, 'Damping factor that ties down the trust radius to trust0; decrease for a more variable step size.', 'Main Optimizer'),
                 "error_tolerance"        : (0.0,   10, 'Error tolerance; the optimizer will only reject steps that increase the objective function by more than this number.', 'Main Optimizer'),
                 "search_tolerance"       : (1e-4, -10, 'Search tolerance; used only when trust radius is negative, dictates convergence threshold of nonlinear search.', 'Main Optimizer with negative mintrust; advanced usage')
                 },
    'sections': {"read_mvals" : (None, 100, 'Paste mathematical parameters into the input file for them to be read in directly', 'Restarting an optimization'),
                 "read_pvals" : (None, 100, 'Paste physical parameters into the input file for them to be read in directly', 'Restarting an optimization (recommend use_mvals instead)'),
                 "priors"     : (OrderedDict(), 150, 'Paste priors into the input file for them to be read in directly', 'Scaling and regularization of parameters (important)')
                 }
    }

## Default fitting target options.
tgt_opts_types = {
    'strings' : {"name"      : (None, 200, 'The name of the target, corresponding to the directory targets/name', 'All targets (important)'),
                 "masterfile": ('interactions.txt', 0, 'The name of the master file containing interacting systems', 'Binding energy target', 'BindingEnergy'),
                 "force_map" : ('residue', 0, 'The resolution of mapping interactions to net forces and torques for groups of atoms.  In order of resolution: molecule > residue > charge-group', 'Force Matching', 'AbInitio'),
                 "fragment1" : ('', 0, 'Interaction fragment 1: a selection of atoms specified using atoms and dashes, e.g. 1-6 to select the first through sixth atom (i.e. list numbering starts from 1)', 'Interaction energies', 'Interaction'),
                 "fragment2" : ('', 0, 'Interaction fragment 2: a selection of atoms specified using atoms and dashes, e.g. 7-11 to select atoms 7 through 11.', 'Interaction energies', 'Interaction'),
                 "openmm_cuda_precision" : ('', -10, 'Precision of local OpenMM calculation.  Choose either single, double or mixed ; defaults to the OpenMM default.', 'Targets that use OpenMM', 'OpenMM'),
                 },
    'allcaps' : {"type"   : (None, 200, 'The type of fitting target, for instance AbInitio_GMX ; this must correspond to the name of a Target subclass.', 'All targets (important)' ,'')
                 },
    'lists'   : {"fd_ptypes" : ([], -100, 'The parameter types that are differentiated using finite difference', 'In conjunction with fdgrad, fdhess, fdhessdiag; usually not needed')
                 },
    'ints'    : {"shots"              : (-1, 0, 'Number of snapshots; defaults to all of the snapshots', 'Energy + Force Matching', 'AbInitio'),
                 "fitatoms"           : (0, 0, 'Number of fitting atoms; defaults to all of them', 'Energy + Force Matching', 'AbInitio'),
                 "sleepy"             : (0, -50, 'Wait a number of seconds every time this target is visited (gives me a chance to ctrl+C)', 'All targets (advanced usage)'),
                 "liquid_prod_steps"  : (20000, 0, 'Number of time steps for the liquid production run.', 'Condensed phase property targets', 'liquid'),
                 "liquid_equ_steps"   : (10000, 0, 'Number of time steps for the liquid equilibration run.', 'Condensed phase property targets', 'liquid'),
                 },
    'bools'   : {"whamboltz"        : (0, -100, 'Whether to use WHAM Boltzmann Weights', 'Ab initio targets with Boltzmann weights (advanced usage)', 'AbInitio'),
                 "sampcorr"         : (0, -150, 'Whether to use the archaic sampling correction', 'Energy + Force Matching, very old option, do not use', 'AbInitio'),
                 "covariance"       : (0, -100, 'Whether to use the quantum covariance matrix', 'Energy + Force Matching, only if you know what you are doing', 'AbInitio'),
                 "batch_fd"         : (0, -150, 'Whether to batch and queue up finite difference jobs, defaults to False', 'Currently unused'),
                 "fdgrad"           : (0, -100, 'Finite difference gradient of objective function w/r.t. specified parameters', 'Use together with fd_ptypes (advanced usage)'),
                 "fdhess"           : (0, -100, 'Finite difference Hessian of objective function w/r.t. specified parameters', 'Use together with fd_ptypes (advanced usage)'),
                 "fdhessdiag"       : (0, -100, 'Finite difference Hessian diagonals w/r.t. specified parameters (costs 2np times a objective calculation)', 'Use together with fd_ptypes (advanced usage)'),
                 "all_at_once"      : (1, -50, 'Compute all energies and forces in one fell swoop where possible(as opposed to calling the simulation code once per snapshot)', 'Various QM targets and MD codes', 'AbInitio'),
                 "run_internal"     : (1, -50, 'For OpenMM or other codes with Python interface: Compute energies and forces internally', 'OpenMM interface', 'OpenMM'),
                 "energy"           : (1, 0, 'Enable the energy objective function', 'All ab initio targets', 'AbInitio'), 
                 "force"            : (1, 0, 'Enable the force objective function', 'All ab initio targets', 'AbInitio'), 
                 "resp"             : (0, -150, 'Enable the RESP objective function', 'Ab initio targets with RESP; experimental (remember to set espweight)'),
                 "do_cosmo"         : (0, -150, 'Call Q-Chem to do MM COSMO on MM snapshots.', 'Currently unused, but possible in AbInitio target'),
                 "optimize_geometry": (1, 0, 'Perform a geometry optimization before computing properties', 'Monomer properties', 'moments'),
                 "absolute"         : (0, -150, 'When matching energies in AbInitio, do not subtract the mean energy gap.', 'Energy matching (advanced usage)', 'abinitio'),
                 "cauchy"           : (0, 0, 'Normalize interaction energies each using 1/(denom**2 + reference**2) which resembles a Cauchy distribution', 'Interaction energy targets', 'interaction'),
                 "attenuate"        : (0, 0, 'Normalize interaction energies using 1/(denom**2 + reference**2) only for repulsive interactions greater than denom.', 'Interaction energy targets', 'interaction'),
                 "manual"           : (0, -150, 'Give the user a chance to fill in condensed phase stuff on the zeroth step', 'Condensed phase property targets (advanced usage)', 'liquid'),
                 "hvap_subaverage"  : (0, -150, 'Don\'t target the average enthalpy of vaporization and allow it to freely float (experimental)', 'Condensed phase property targets (advanced usage)', 'liquid'),
                 "force_cuda"       : (0, -150, 'Force the external npt.py script to crash if CUDA Platform not available', 'Condensed phase property targets (advanced usage)', 'liquid_openmm'),
                },
    'floats'  : {"weight"       : (1.0, 150, 'Weight of the target (determines its importance vs. other targets)', 'All targets (important)'),
                 "w_rho"        : (1.0, 0, 'Weight of experimental density', 'Condensed phase property targets', 'liquid'),
                 "w_hvap"       : (1.0, 0, 'Weight of enthalpy of vaporization', 'Condensed phase property targets', 'liquid'),
                 "w_alpha"      : (1.0, 0, 'Weight of thermal expansion coefficient', 'Condensed phase property targets', 'liquid'),
                 "w_kappa"      : (1.0, 0, 'Weight of isothermal compressibility', 'Condensed phase property targets', 'liquid'),
                 "w_cp"         : (1.0, 0, 'Weight of isobaric heat capacity', 'Condensed phase property targets', 'liquid'),
                 "w_eps0"       : (1.0, 0, 'Weight of dielectric constant', 'Condensed phase property targets', 'liquid'),
                 "w_energy"     : (1.0, 0, 'Weight of energy', 'Ab initio targets', 'liquid'),
                 "w_force"      : (1.0, 0, 'Weight of atomistic forces', 'Ab initio targets', 'liquid'),
                 "w_netforce"   : (0.0, 0, 'Weight of net forces (condensed to molecules, residues, or charge groups)', 'Ab initio targets', 'abinitio'),
                 "w_torque"     : (0.0, 0, 'Weight of torques (condensed to molecules, residues, or charge groups)', 'Ab initio targets', 'abinitio'),
                 "w_resp"       : (0.0, -150, 'Weight of RESP', 'Ab initio targets with RESP (advanced usage)', 'abinitio'),
                 "resp_a"       : (0.001, -150, 'RESP "a" parameter for strength of penalty; 0.001 is strong, 0.0005 is weak', 'Ab initio targets with RESP (advanced usage)', 'abinitio'),
                 "resp_b"       : (0.1, -150, 'RESP "b" parameter for hyperbolic behavior; 0.1 is recommended', 'Ab initio targets with RESP (advanced usage)', 'abinitio'),
                 "energy_upper" : (30.0, 0, 'Upper energy cutoff (in kcal/mol); super-repulsive interactions are given zero weight', 'Interaction energy targets', 'interaction'),
                 "qmboltz"      : (0.0, -100, 'Fraction of Quantum Boltzmann Weights (ab initio), 1.0 for full reweighting, 0.5 for hybrid', 'Ab initio targets with Boltzmann weights (advanced usage)', 'abinitio'),
                 "qmboltztemp"  : (298.15, -100, 'Temperature for Quantum Boltzmann Weights (ab initio), defaults to room temperature', 'Ab initio targets with Boltzmann weights (advanced usage)', 'abinitio'),
                 "energy_denom"   : (1.0, 0, 'Energy normalization for binding energies in kcal/mol (default is to use stdev)', 'Binding energy targets', 'binding'),
                 "rmsd_denom"     : (0.1, 0, 'RMSD normalization for optimized geometries in Angstrom', 'Binding energy targets', 'binding'),
                 "wavenumber_tol" : (10.0, 0, 'Frequency normalization (in wavenumber) for vibrational frequencies', 'Vibrational frequency targets', 'vibration'),
                 "dipole_denom"   : (1.0, 0, 'Dipole normalization (Debye) ; set to 0 if a zero weight is desired', 'Monomer property targets', 'monomer'),
                 "quadrupole_denom"   : (1.0, 0, 'Quadrupole normalization (Buckingham) ; set to 0 if a zero weight is desired', 'Monomer property targets', 'monomer'),
                 "polarizability_denom"   : (1.0, 0, 'Dipole polarizability tensor normalization (cubic Angstrom) ; set to 0 if a zero weight is desired', 'Monomer property targets with polarizability', 'monomer'),
                 "liquid_timestep"  : (0.5, 0, 'Time step size for the liquid simulation.', 'Condensed phase property targets', 'liquid'),
                 "liquid_interval"  : (0.05, 0, 'Time interval for saving coordinates for the liquid production run.', 'Condensed phase property targets', 'liquid'),
                 },
    'sections': {}
    }

## Default general options - basically a collapsed veresion of gen_opts_types.
gen_opts_defaults = {}
for t in gen_opts_types:
    subdict = {}
    for i in gen_opts_types[t]:
        subdict[i] = gen_opts_types[t][i][0]
    gen_opts_defaults.update(subdict)

## Default target options - basically a collapsed version of tgt_opts_types.
tgt_opts_defaults = {}
for t in tgt_opts_types:
    subdict = {}
    for i in tgt_opts_types[t]:
        subdict[i] = tgt_opts_types[t][i][0]
    tgt_opts_defaults.update(subdict)

## Option maps for maintaining backward compatibility.
bkwd = {"simtype" : "type"}

## Listing of sections in the input file.
mainsections = ["SIMULATION","TARGET","OPTIONS","END","NONE"]

def read_mvals(fobj):
    Answer = []
    for line in fobj:
        if re.match("(/read_mvals)|(^\$end)",line):
            break
        Answer.append(float(line.split('[')[-1].split(']')[0].split()[-1]))
    return Answer
        
def read_pvals(fobj):
    Answer = []
    for line in fobj:
        if re.match("(/read_pvals)|(^\$end)",line):
            break
        Answer.append(float(line.split('[')[-1].split(']')[0].split()[-1]))
    return Answer

def read_priors(fobj):
    Answer = OrderedDict()
    for line in fobj:
        line = line.split("#")[0]
        if re.match("(/priors)|(^\$end)",line):
            break
        Answer[line.split()[0]] = float(line.split()[-1])
    return Answer

def read_internals(fobj):
    return

## ParsTab that refers to subsection parsers.
ParsTab  = {"read_mvals" : read_mvals,
            "read_pvals" : read_pvals,
            "priors"     : read_priors,
            "internal"   : read_internals
            }

def printsection(heading,optdict,typedict):
    """ Print out a section of the input file in a parser-compliant and readable format.

    At the time of writing of this function, it's mainly intended to be called by MakeInputFile.py.
    The heading is printed first (it is something like $options or $target).  Then it loops
    through the variable types (strings, allcaps, etc...) and the keys in each variable type.
    The one-line description of each key is printed out as a comment, and then the key itself is
    printed out along with the value provided in optdict.  If optdict is None, then the default
    value is printed out instead.

    @param[in] heading Heading, either $options or $target
    @param[in] optdict Options dictionary or None.
    @param[in] typedict Option type dictionary, either gen_opts_types or tgt_opts_types specified in this file.
    @return Answer List of strings for the section that we are printing out.
    
    """
    from forcebalance.implemented import Implemented_Targets
    from forcebalance.optimizer import Optimizer

    def FilterTargets(search):
        if type(search) == str:
            search = [search]
        list_out = []
        for key in sorted(Implemented_Targets.keys()):
            if any([i.lower() in key.lower() for i in search]):
                list_out.append(Implemented_Targets[key].__name__)
        return ', '.join(sorted(list_out))

    Answer = [heading]
    firstentry = 1
    Options = []
    for i in ['strings','allcaps','lists','ints','bools','floats','sections']:
        vartype = re.sub('s$','',i)
        for j in typedict[i]:
            Option = []
            val = optdict[j] if optdict != None else typedict[i][j][0]
            if firstentry:
                firstentry = 0
            else:
                Option.append("")
            Priority = typedict[i][j][1]
            Option.append("# (%s) %s" % (vartype, typedict[i][j][2]))
            if len(typedict[i][j]) >= 4:
                Relevance = typedict[i][j][3]
                str2 = "# used in: %s" % Relevance
                if len(typedict[i][j]) >= 5:
                    TargetName = FilterTargets(typedict[i][j][4])
                    str2 += " (%s)" % TargetName
                else:
                    TargetName = "None"
                Option.append(str2)
            else:
                Relevance = "None"
            Option.append("%s %s" % (str(j),str(val)))
            Options.append((Option, Priority, TargetName, j))
    def key1(o):
        return o[1]
    def key2(o):
        return o[2]
    def key3(o):
        return o[3]
    Options.sort(key=key3)
    Options.sort(key=key2)
    Options.sort(key=key1, reverse=True)
    for o in Options:
        Answer += o[0]

    # PriSet = sorted(list(set(Priorities)))[::-1]
    # TgtSet = sorted(list(set(TargetNames)))
    # RelSet = sorted(list(set(Relevances)))
    # for p0 in PriSet:
    #     ogrp = []
    #     rgrp = []
    #     tgrp = []
    #     for o, p, r, t in zip(Options, Priorities, Relevances, TargetNames):
    #         if p == p0:
    #             ogrp.append(o)
    #             rgrp.append(r)
    #             tgrp.append(t)
    #     ogrp2 = []
    #     rgrp2 = []
    #     for t0 in TgtSet:
    #         for o, r, t in zip(ogrp, rgrp, tgrp):
    #             if t == t0:
    #                 ogrp2.append(
                
    Answer.append("$end")
    return Answer

def parse_inputs(input_file=None):
    """ Parse through the input file and read all user-supplied options.

    This is usually the first thing that happens when an executable script is called.
    Our parser first loads the default options, and then updates these options as it
    encounters keywords.

    Each keyword corresponds to a variable type; each variable type (e.g. string,
    integer, float, boolean) is treated differently.  For more elaborate inputs,
    there is a 'section' variable type.

    There is only one set of general options, but multiple sets of target options.
    Each target has its own section delimited by the \em $target keyword,
    and we build a list of target options.  

    @param[in]  input_file The name of the input file.
    @return     options    General options.
    @return     tgt_opts   List of fitting target options.
    
    @todo Implement internal coordinates.
    @todo Implement sampling correction.
    @todo Implement charge groups.
    """
    
    print "Reading options from file: %s" % input_file
    section = "NONE"
    # First load in all of the default options.
    options = {'root':os.getcwd()}
    options.update(gen_opts_defaults)
    tgt_opts = []
    this_tgt_opt = deepcopy(tgt_opts_defaults)
    # Give back a bunch of default options if input file isn't specified.
    if input_file == None:
        return options, tgt_opts
    fobj = open(input_file)
    for line in fobj:
        try:
            # Anything after "#" is a comment
            line = line.split("#")[0].strip()
            s = line.split()
            # Skip over blank lines
            if len(s) == 0:
                continue
            key = s[0].lower()
            if key in bkwd: # Do option replacement for backward compatibility.
                key = bkwd[key]
            # If line starts with a $, this signifies that we're in a new section.
            if re.match('^\$',line):
                newsection = re.sub('^\$','',line).upper()
                if section in ["SIMULATION","TARGET"] and newsection in mainsections:
                    tgt_opts.append(this_tgt_opt)
                    this_tgt_opt = deepcopy(tgt_opts_defaults)
                section = newsection
            elif section in ["OPTIONS","SIMULATION","TARGET"]:
                ## Depending on which section we are in, we choose the correct type dictionary
                ## and add stuff to 'options' and 'this_tgt_opt'
                (this_opt, opts_types) = (options, gen_opts_types) if section == "OPTIONS" else (this_tgt_opt, tgt_opts_types)
                ## Note that "None" is a special keyword!  The variable will ACTUALLY be set to None.
                if len(s) > 1 and s[1].upper() == "NONE":
                    this_opt[key] = None
                elif key in opts_types['strings']:
                    this_opt[key] = s[1]
                elif key in opts_types['allcaps']:
                    this_opt[key] = s[1].upper()
                elif key in opts_types['lists']:
                    for word in s[1:]:
                        this_opt.setdefault(key,[]).append(word)
                elif key in opts_types['ints']:
                    this_opt[key] = int(s[1])
                elif key in opts_types['bools']:
                    if len(s) == 1:
                        this_opt[key] = True
                    elif s[1].upper() in ["0", "NO", "FALSE"]:
                        this_opt[key] = False
                    else:
                        this_opt[key] = True
                elif key in opts_types['floats']:
                    this_opt[key] = float(s[1])
                elif key in opts_types['sections']:
                    this_opt[key] = ParsTab[key](fobj)
                else:
                    print "Unrecognized keyword: --- \x1b[1;91m%s\x1b[0m --- in %s section" \
                          % (key, section)
                    print "Perhaps this option actually belongs in %s section?" \
                          % (section == "OPTIONS" and "a TARGET" or "the OPTIONS")
                    sys.exit(1)
            elif section not in mainsections:
                print "Unrecognized section: %s" % section
                sys.exit(1)
        except:
            print "Failed to read in this line! Check your input file."
            print line,
            traceback.print_exc()
            sys.exit(1)
    if section == "SIMULATION" or section == "TARGET":
        tgt_opts.append(this_tgt_opt)
    if not options['verbose_options']:
        printcool("Options at their default values are not printed\n Use 'verbose_options True' to Enable", color=5)
    return options, tgt_opts
