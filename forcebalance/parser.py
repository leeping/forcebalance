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
- Section option types haven't been implemented yet.  They are meant to treat more elaborate inputs, such as
the user pasting in output parameters from a previous job as input, or a specification of internal coordinate system.
I imagine that for every section type I would have to write my own parser.  Maybe a ParsTab of parsing functions would work. :)

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
gen_opts_types = {
    'strings' : {"gmxpath"                   : None,    # Overall path for GROMACS executables (i.e. if we ran "make install" for gmx) for *_gmx functionality
                 "gmxrunpath"                : None,    # Path for GROMACS executables grompp and mdrun, may override GMXPATH (i.e. if we only ran "make")
                 "gmxtoolpath"               : None,    # Path for GROMACS tools, may override GMXPATH
                 "gmxsuffix"                 : "",      # The suffix of GROMACS executables
                 "penalty_type"              : "L2",    # Type of the penalty, L2 or L1 in the optimizer
                 "scan_vals"                 : None     # Values to scan in the parameter space for job type "scan[mp]vals", given like this: -0.1:0.01:0.1
                 },
    'allcaps' : {"jobtype"                   : "sp"     # The job type, defaults to a single-point evaluation of objective function
                 },
    'lists'   : {"forcefield"                : [],      # The names of force fields, corresponding to directory forcefields/file_name.(itp|gen)
                 "scanindex_num"             : [],      # Numerical index of the parameter to scan over in job type "scan[mp]vals"
                 "scanindex_name"            : []       # Parameter name to scan over (should convert to a numerical index) in job type "scan[mp]vals"
                 },
    'ints'    : {"maxstep"                   : 100      # Maximum number of steps in an optimization
                 },
    'bools'   : {"backup"                    : 1,       # Write temp directories to backup before wiping them (always used)
                 "readchk"                   : 1,       # Read in a checkpoint file (for the optimizer)
                 "writechk"                  : 1        # Write the checkpoint file (for the optimizer)
                 },
    'floats'  : {"trust0"                    : 1e-4,    # Trust radius for the MainOptimizer
                 "convergence_objective"     : 1e-4,    # Convergence criterion of objective function (in MainOptimizer this is the stdev of x2 over 10 steps)
                 "convergence_step"          : 1e-4,    # Convergence criterion of step size (just needs to fall below this threshold)
                 "eig_lowerbound"            : 1e-4,    # Minimum eigenvalue for applying steepest descent correction in the MainOptimizer
                 "finite_difference_h"       : 1e-4,    # Step size for finite difference derivatives in many functions (get_(G/H) in fitsim, FDCheckG)
                 "penalty_additive"          : 0.0,     # Factor for additive penalty function in objective function
                 "penalty_multiplicative"    : 0.1      # Factor for multiplicative penalty function in objective function
                 },
    'sections': {"readmvals"                 : None,    # Paste mathematical parameters into the input file for them to be read in directly
                 "readpvals"                 : None     # Paste physical parameters into the input file for them to be read in directly
                 }
    }

## Default general options - basically a collapsed veresion of gen_opts_types.
gen_opts_defaults = {}
for t in gen_opts_types:
    gen_opts_defaults.update(gen_opts_types[t])

# NOT IMPLEMENTED YET: Internal coordinates, 'Sampling correction', charge groups, parallel run

## Default fitting simulation options.
sim_opts_types = {
    'strings' : {"name"                      : None,    # The name of the simulation, which corresponds to the directory simulations/dir_name
                 },
    'allcaps' : {"simtype"                   : None     # The type of fitting simulation, for instance ForceEnergyMatching_GMX
                 },
    'lists'   : {"fd_ptypes"                 : []       # The parameter types that need to be differentiated using finite difference
                 },
    'ints'    : {"shots"                     : -1,      # Number of snapshots (force+energy matching); defaults to all of the snapshots
                 "fitatoms"                  : 0        # Number of fitting atoms (force+energy matching); defaults to all of them
                 },
    'bools'   : {"whamboltz"                 : 0,       # Whether to use WHAM Boltzmann Weights (force+energy match), defaults to False
                 "sampcorr"                  : 0,       # Whether to use the (archaic) sampling correction (force+energy match), defaults to False
                 "covariance"                : 1,       # Whether to use the quantum covariance matrix (force+energy match), defaults to True
                 "batch_fd"                  : 0,       # Whether to batch and queue up finite difference jobs, defaults to False
                 "fdgrad"                    : 1,       # Finite difference gradients
                 "fdhess"                    : 1,       # Finite difference Hessian diagonals (costs np times a gradient calculation)
                 "fdhessdiag"                : 1,       # Finite difference Hessian diagonals (cheap; costs 2np times a objective calculation)
                 "use_pvals"                 : 0        # Bypass the transformation matrix and use the physical parameters directly
                 },
    'floats'  : {"weight"                    : 1.0,     # Weight of the current simulation (with respect to other simulations)
                 "efweight"                  : 0.5,     # 1.0 for all energy and 0.0 for all force (force+energy match), defaults to 0.5
                 "qmboltz"                   : 0.0,     # Fraction of Quantum Boltzmann Weights (force+energy match), 1.0 for full reweighting, 0.0 < 1.0 for hybrid
                 "qmboltztemp"               : 298.15   # Temperature for Quantum Boltzmann Weights (force+energy match), defaults to room temperature
                 },
    'sections': {}
    }

## Default simulation options - basically a collapsed version of sim_opts_types.
sim_opts_defaults = {}
for t in sim_opts_types:
    sim_opts_defaults.update(sim_opts_types[t])

## Listing of sections in the input file.
mainsections = ["SIMULATION","OPTIONS","END","NONE"]
## Listing of subsections.
subsections  = {"OPTIONS":["READ_MVALS", "READ_PVALS"]}

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
    
    @todo The section variable type hasn't been implemented yet.
    """
    
    print "Reading options from file: %s" % input_file
    section = "NONE"
    # First load in all of the default options.
    options = {'root':os.getcwd()}
    options.update(gen_opts_defaults)
    sim_opts = []
    this_sim_opt = deepcopy(sim_opts_defaults)
    for line in open(input_file):
        # Anything after "#" is a comment
        line = line.split("#")[0].strip()
        s = line.split()
        # Capitalize the first letter to make the first field insensitive
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
            # Depending on which section we are in, we choose the correct type dictionary
            # and add stuff to 'options' and 'this_sim_opt'
            (this_opt, opts_types) = section == "OPTIONS" and (options, gen_opts_types) or (this_sim_opt, sim_opts_types)
            if key in opts_types['strings']:
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
                pass # Not implemented yet
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
