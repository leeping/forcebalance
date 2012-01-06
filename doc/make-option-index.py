#!/usr/bin/env python

from forcebalance import parser

""" The OptionDoc dictionary is a key : dictionary dictionary.
Each variable can have several parts of the documentation: 'pert' = 
"""

GenOptionDoc = {"gmxpath" : {"scope" : "Fitting simulations that use GROMACS (GROMACS-X2 for ForceEnergyMatch_GMX)",
                             "required" : True,
                             "long" : "Specify the path where GROMACS executables are installed, most likely ending in 'bin'.
                             Note that executables are only installed 'bin' if the program is installed using 'make install';
                             this will NOT be the case if you simply ran 'make'.  Built executables are located in the 'kernel'
                             and 'tools' directories prior to installing, in those instances use 'gmxrunpath' and 'gmxtoolpath'
                             respectively.",
                             "recommend" : "Depends on your local installation and environment."
                             },
                "gmxrunpath" : {"scope" : "Fitting simulations that use GROMACS",
                                "required" : False,
                                "long" : "The path for built GROMACS executables, most likely ending in 'kernel'.  You only
                                need to use this if you didn't install the built executables (which LPW often did during
                                development of GROMACS-X2).  If specified, this overrides 'gmxpath' for the programs grompp
                                and mdrun.",
                                "recommend" : "Specify if GROMACS is built but not installed."
                                },
                "gmxtoolpath" : {"scope" : "Fitting simulations that use GROMACS",
                                 "required" : False,
                                 "long" : "The path for built GROMACS tools, most likely ending in 'tools'.  If specified, this
                                 overrides 'gmxpath' for the programs trjconv, g_energy and some others.",
                                 "recommend" : "Specify if GROMACS is built but not installed."
                                 },
                "gmxsuffix" : {"scope" : "Fitting simulations that use GROMACS",
                               "required" : False,
                               "long" : "Depending on how GROMACS is configured and installed, a suffix may be appended to executable
                               names.  If there is a suffix, it needs to be specified here (or else ForceBalance will not find the
                               GROMACS executable and it will crash.",
                               "recommend" : "Depends on your local installation and environment."
                               },
                "penalty_type" : {"scope" : "All force field optimizations",
                                  "required" : False,
                                  "long" : "To prevent the optimization from changing the parameters too much, an additional penalty
                                  is applied to the objective function that depends linearly (L1) or quadratically (L2) on the norm
                                  of the parameter displacement vector.  L1 corresponds to LASSO regularization while L2 is known as
                                  Tikhonov regularization or ridge regression."
                                  "recommend" : "L2; tested and known to be working.  Implementation of L1 in progress."
                                  },
                "scan_vals" : {"scope" : "scan_mvals and scan_pvals job types",
                               "required" : True,
                               "long" : "This specifies a range of parameter values to scan in a uniform grid.  scan_mvals works in
                               the mathematical parameter space while scan_pvals works in the physical parameter space.  The syntax
                               is lower:step:upper .  Both lower and upper limits are included in the range."
                               "recommend" : "For scan_mvals, a range of values between -1 and +1 is recommended; for scan_pvals, choose
                               values close to the physical parameter value."
                               },
                "readchk" : {"scope" : "Main optimizer",
                             "required" : False,
                             "long" : "The main optimizer has the ability to pick up where it left off by reading / writing checkpoint
                             files.  Here you may specify the checkpoint file to read in from a previous optimization run."
                             },
                "writechk" : {"scope" : "Main optimizer",
                              "required" : False,
                              "long" : "The main optimizer has the ability to pick up where it left off by reading / writing checkpoint
                              files.  Here you may specify the checkpoint file to write after the job is finished."
                              },
                "ffdir" : {"scope" : "All force field optimizations",
                           "required" : True
                           },
                "jobtype" : {"scope" : "All force field optimizations",
                             "required" : True,
                             "long" : "Here you may specify the type of ForceBalance job.  This ranges from gradient-based and stochastic
                             optimizations to simple scans over the parameter space and finite difference checking of gradients.",
                             "recommend" : "See the Optimizer class documentation for which optimizer is best suited for you."
                             }
                "forcefield" : {"scope" : "All force field optimizations",
                                "required" : True,
                                
                              
                              
                }

def main():
    for t in gen_opts_types:
        
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


if __name__ == "__main__":
    main()
