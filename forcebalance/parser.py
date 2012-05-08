""" @package parser Input file parser for ForceBalance projects.  Additionally, the location for all default options.

Although I will do my best to write good documentation,
for many programs the input parser becomes the most up-to-date
source for documentation.  So this is a great place to write
lots of comments for those who implement new functionality.

Basically, the way my program is structured there is a set of
GENERAL options and also a set of SIMULATION options.  Since
there can be many fitting simulations within a single project
(i.e. we may wish to fit water trimers and hexamers, which
constitutes two fitting simulations) the input is organized
into sections, like so:

$options\n
gen_option_1 Big\n
gen_option_2 Mao\n
$simulation\n
sim_option_1 Sniffy\n
sim_option_2 Schmao\n
$simulation\n
sim_option_1 Nifty\n
sim_option_2 Jiffy\n
$end

In this case, two sets of simulation options are generated in addition to the general option.

Each option is meant to be parsed as a certain variable type.

- String option values are read in directly; note that only the first two words in the line are processed
- Some strings are capitalized when they are read in; this is mainly for function tables like OptTab and SimTab
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
@date 12/2011
"""

import os
import re
import sys
import itertools
from nifty import printcool, printcool_dictionary
from copy import deepcopy

## Default general options.
## Note that the documentation is included in part of the key; this will aid in automatic doc-extraction. :)
gen_opts_types = {
    'strings' : {"gmxpath"      : (None, 'Path for GROMACS executables'),
                 "gmxsuffix"    : ('',   'The suffix of GROMACS executables'),
                 "tinkerpath"   : (None, 'Path for TINKER executables'),
                 "penalty_type" : ("L2", 'Type of the penalty, L2 or L1 in the optimizer'),
                 "scan_vals"    : (None, 'Values to scan in the parameter space for job type "scan[mp]vals", given like this: -0.1:0.1:11'),
                 "readchk"      : (None, 'Name of the restart file we read from'),
                 "writechk"     : (None, 'Name of the restart file we write to (can be same as readchk)'),
                 "ffdir"        : ('forcefield', 'Directory containing force fields, relative to project directory')
                 },
    'allcaps' : {"jobtype"      : ("single", 'The job type, defaults to a single-point evaluation of objective function'),
                 },
    'lists'   : {"forcefield"     : ([], 'The names of force fields, corresponding to directory forcefields/file_name.(itp|gen)'),
                 "scanindex_num"  : ([], 'Numerical index of the parameter to scan over in job type "scan[mp]vals"'),
                 "scanindex_name" : ([], 'Parameter name to scan over (should convert to a numerical index) in job type "scan[mp]vals"')
                 },
    'ints'    : {"maxstep"      : (100, 'Maximum number of steps in an optimization'),
                 },
    'bools'   : {"backup"         : (1, 'Write temp directories to backup before wiping them'),
                 "writechk_step"  : (0, 'Write the checkpoint file at every optimization step')
                 },
    'floats'  : {"trust0"                 : (1e-2, 'Trust radius for the MainOptimizer'),
                 "convergence_objective"  : (1e-4, 'Convergence criterion of objective function (in MainOptimizer this is the stdev of x2 over 10 steps)'),
                 "convergence_gradient"   : (1e-4, 'Convergence criterion of gradient norm'),
                 "convergence_step"       : (1e-4, 'Convergence criterion of step size (just needs to fall below this threshold)'),
                 "eig_lowerbound"         : (1e-4, 'Minimum eigenvalue for applying steepest descent correction in the MainOptimizer'),
                 "finite_difference_h"    : (1e-2, 'Step size for finite difference derivatives in many functions (get_(G/H) in fitsim, FDCheckG)'),
                 "penalty_additive"       : (0.0, 'Factor for additive penalty function in objective function'),
                 "penalty_multiplicative" : (0.1, 'Factor for multiplicative penalty function in objective function'),
                 "adaptive_factor"        : (0.5, 'The step size is increased / decreased by up to this much in the event of a good / bad step; increase for a more variable step size.'),
                 "adaptive_damping"       : (0.5, 'Damping factor that ties down the trust radius to trust0; decrease for a more variable step size.')
                 },
    'sections': {"read_mvals" : (None, 'Paste mathematical parameters into the input file for them to be read in directly'),
                 "read_pvals" : (None, 'Paste physical parameters into the input file for them to be read in directly'),
                 "priors"     : ({}, 'Paste priors into the input file for them to be read in directly')
                 }
    }

## Default general options - basically a collapsed veresion of gen_opts_types.
gen_opts_defaults = {}
for t in gen_opts_types:
    subdict = {}
    for i in gen_opts_types[t]:
        subdict[i] = gen_opts_types[t][i][0]
    gen_opts_defaults.update(subdict)

## Default fitting simulation options.
sim_opts_types = {
    'strings' : {"name"      : (None, 'The name of the simulation, which corresponds to the directory simulations/dir_name')
                 },
    'allcaps' : {"simtype"   : (None,      'The type of fitting simulation, for instance ForceEnergyMatch_GMXX2')
                 },
    'lists'   : {"fd_ptypes" : ([], 'The parameter types that need to be differentiated using finite difference')
                 },
    'ints'    : {"shots"     : (-1, 'Number of snapshots (force+energy matching); defaults to all of the snapshots'),
                 "fitatoms"  : (0, 'Number of fitting atoms (force+energy matching); defaults to all of them'),
                 "wq_port"   : (9230, 'The port number to use for Work Queue')
                 },
    'bools'   : {"whamboltz"  : (0, 'Whether to use WHAM Boltzmann Weights (force+energy match), defaults to False'),
                 "sampcorr"   : (0, 'Whether to use the (archaic) sampling correction (force+energy match), defaults to False'),
                 "covariance" : (0, 'Whether to use the quantum covariance matrix (force+energy match), defaults to False'),
                 "batch_fd"   : (0, 'Whether to batch and queue up finite difference jobs, defaults to False'),
                 "fdgrad"     : (1, 'Finite difference gradients'),
                 "fdhess"     : (0, 'Finite difference Hessian diagonals (costs np times a gradient calculation)'),
                 "fdhessdiag" : (1, 'Finite difference Hessian diagonals (cheap; costs 2np times a objective calculation)'),
                 "use_pvals"  : (0, 'Bypass the transformation matrix and use the physical parameters directly'),
                 "all_at_once": (1, 'Compute all energies and forces in one fell swoop (as opposed to calling the simulation code once per snapshot)'),
                 "run_internal": (1,'For OpenMM or other codes with Python interface: Compute energies and forces internally'),
                 },
    'floats'  : {"weight"      : (1.0, 'Weight of the current simulation (with respect to other simulations)'),
                 "efweight"    : (0.5, '1.0 for all energy and 0.0 for all force (force+energy match), defaults to 0.5'),
                 "qmboltz"     : (0.0, 'Fraction of Quantum Boltzmann Weights (force+energy match), 1.0 for full reweighting, 0.0 < 1.0 for hybrid'),
                 "qmboltztemp" : (298.15, 'Temperature for Quantum Boltzmann Weights (force+energy match), defaults to room temperature')
                 },
    'sections': {}
    }

## Default simulation options - basically a collapsed version of sim_opts_types.
sim_opts_defaults = {}
for t in sim_opts_types:
    subdict = {}
    for i in sim_opts_types[t]:
        subdict[i] = sim_opts_types[t][i][0]
    sim_opts_defaults.update(subdict)

## Listing of sections in the input file.
mainsections = ["SIMULATION","OPTIONS","END","NONE"]

def read_mvals(fobj):
    Answer = []
    for line in fobj:
        if re.match("(/read_mvals)|(^\$end)",line):
            break
        Answer.append(float(line.split('[')[-1].split(']')[0].split()[0]))
    return Answer
        
def read_pvals(fobj):
    Answer = []
    for line in fobj:
        if re.match("(/read_pvals)|(^\$end)",line):
            break
        Answer.append(float(line.split('[')[-1].split(']')[0].split()[0]))
    return Answer

def read_priors(fobj):
    Answer = {}
    for line in fobj:
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
    The heading is printed first (it is something like $options or $simulation).  Then it loops
    through the variable types (strings, allcaps, etc...) and the keys in each variable type.
    The one-line description of each key is printed out as a comment, and then the key itself is
    printed out along with the value provided in optdict.  If optdict is None, then the default
    value is printed out instead.

    @param[in] heading Heading, either $options or $simulation
    @param[in] optdict Options dictionary or None.
    @param[in] typedict Option type dictionary, either gen_opts_types or sim_opts_types specified in this file.
    @return Answer List of strings for the section that we are printing out.
    
    """
    Answer = [heading]
    firstentry = 1
    for i in ['strings','allcaps','lists','ints','bools','floats','sections']:
        vartype = re.sub('s$','',i)
        for j in typedict[i]:
            val = optdict[j] if optdict != None else typedict[i][j][0]
            if firstentry:
                firstentry = 0
            else:
                Answer.append("")
            Answer.append("# (%s) %s" % (vartype, typedict[i][j][1]))
            Answer.append("%s %s" % (str(j),str(val)))
    Answer.append("$end")
    return Answer

def parse_inputs(input_file):
    """ Parse through the input file and read all user-supplied options.

    This is usually the first thing that happens when an executable script is called.
    Our parser first loads the default options, and then updates these options as it
    encounters keywords.

    Each keyword corresponds to a variable type; each variable type (e.g. string,
    integer, float, boolean) is treated differently.  For more elaborate inputs,
    there is a 'section' variable type.

    There is only one set of general options, but multiple sets of fitting simulation options.
    Each fitting simulation has its own section delimited by the \em $simulation keyword,
    and we build a list of simulation options.  

    @param[in]  input_file The name of the input file.
    @return     options    General options.
    @return     sim_opts   List of fitting simulation options.
    
    @todo Implement internal coordinates.
    @todo Implement sampling correction.
    @todo Implement charge groups.
    """
    
    print "Reading options from file: %s" % input_file
    section = "NONE"
    # First load in all of the default options.
    options = {'root':os.getcwd()}
    options.update(gen_opts_defaults)
    sim_opts = []
    this_sim_opt = deepcopy(sim_opts_defaults)
    fobj = open(input_file)
    for line in fobj:
        # Anything after "#" is a comment
        line = line.split("#")[0].strip()
        s = line.split()
        # Skip over blank lines
        if len(s) == 0:
            continue
        key = s[0].lower()
        # If line starts with a $, this signifies that we're in a new section.
        if re.match('^\$',line):
            newsection = re.sub('^\$','',line).upper()
            if section == "SIMULATION" and newsection in mainsections:
                sim_opts.append(this_sim_opt)
                this_sim_opt = deepcopy(sim_opts_defaults)
            section = newsection
        elif section in ["OPTIONS","SIMULATION"]:
            ## Depending on which section we are in, we choose the correct type dictionary
            ## and add stuff to 'options' and 'this_sim_opt'
            (this_opt, opts_types) = (options, gen_opts_types) if section == "OPTIONS" else (this_sim_opt, sim_opts_types)
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
                      % (key, section == "OPTIONS" and "general" or "simulation")
                print "Perhaps this option actually belongs in the %s section?" \
                      % (section == "OPTIONS" and "simulation" or "general")
                sys.exit(1)
        elif section not in mainsections:
            print "Unrecognized section: %s" % section
            sys.exit(1)
    if section == "SIMULATION":
        sim_opts.append(this_sim_opt)
    return options, sim_opts
