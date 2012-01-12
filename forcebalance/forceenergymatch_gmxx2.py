""" @package forceenergymatch_gmxx2 Force and energy matching with interface to modified GROMACS.

In order for us to obtain the objective function in force and energy
matching, we loop through the snapshots, compute the energy and force
(as well as its derivatives), and sum them up.  The details of the
process are complicated and I won't document them here.  The contents
of this package (mainly the ForceEnergyMatch_GMXX2 class) allows us to
call the modified GROMACS to compute the objective function for us.

@author  Lee-Ping Wang
@date    12/2011

"""

import os
import shutil
from nifty import col, flat, floatornan, remove_if_exists
from numpy import append, array, mat, zeros
from gmxio import gmxx2_print, rm_gmx_baks
from re import match
import subprocess
from subprocess import PIPE
from forceenergymatch import ForceEnergyMatch

class ForceEnergyMatch_GMXX2(ForceEnergyMatch):

    """ForceBalance class for force and energy matching with the modified GROMACS.

    This class allows us to use a heavily modified version of GROMACS
    (a major component of this program) to compute the objective
    function contribution.  The modified GROMACS does the looping
    through snapshots, computes the interactions as well as the
    derivatives, and sums them up to build the objective function.
    I will write that documentation elsewhere, perhaps when I port
    GROMACS over to version 4.5.4.

    This class implements the 'get' method.  When 'get' is called, the
    force field is printed to the temporary directory along with
    several files containing the information needed by the modified
    GROMACS (the Boltzmann weights, the parameters that need
    derivatives and their values, the QM energies and forces, and the
    energy / force weighting.)

    The modified GROMACS is called with the arguments '-rerun all.gro
    -fortune -rerunvsite' to loop over the snapshots, turn on force
    matching functionality and reconstruct virtual site positions (an
    important consideration if we're changing the virtual site
    positions in the optimization).  Its outputs are 'e2f2bc' which
    means 'energy squared, force squared, boltzmann corrected' and
    contains the objective function, 'a1dbc' and 'a2dbc' containing
    analytic first and second derivatives, 'gmxboltz' containing the
    Boltzmann weights used, and possibly some other stuff.

    Most importantly, 'e2f2bc', 'a1dbc' and 'a2dbc' are read by 'get'
    after GROMACS is called and returned directly as the objective
    function contributions.

    Other methods implemented in this class are related to the
    preparation of the temp directory.
    
    """

    def __init__(self,options,sim_opts,forcefield):
        """Instantiation of ForceEnergyMatch_GMXX2.

        Several important things happen here:
        - We load in the coordinates from 'all.gro'.
        - We prepare the temporary directory.
        
        """
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## Set the software to GROMACS no matter what
        self.trajfnm = "all.gro"
        ## Initialize the superclass. :)
        super(ForceEnergyMatch_GMXX2,self).__init__(options,sim_opts,forcefield)
        ## Put stuff for GROMACS-X2 into the temp directory.
        #self.prepare_gmxx2()
        
    def prepare_temp_directory(self,options,sim_opts,tempdir=None):
        """ Prepare the temporary directory for running the modified GROMACS.

        This method creates the temporary directory, links in the
        necessary files for running (except for the force field), and
        writes the coordinate file for the snapshots we've chosen.

        There are also files that specific to our *modified* GROMACS, including:
        - qmboltz   : The QM Boltzmann weights
        - bp        : The QM vs. MM Boltzmann weight proportionality factor
        - whamboltz : The WHAM Boltzmann weights (i.e. MM Boltzmann weights passed from outside)
        - sampcorr  : Boolean for the 'sampling correction', i.e. updating the Boltzmann factors
        when the force field is updated.  This required a TON of implementation into the
        modified Gromacs, but in the end we didn't find it to be very useful.  It basically
        emphasizes energy minima and gets barrier heights wrong.  Blah! :)
        - fitatoms  : The number of atoms that we're fitting, which may be less than the total
        number in the QM calculation (i.e. if we are fitting something to be compatible with
        a water model ...)
        - energyqm  : QM reference energies
        - forcesqm  : QM reference forces
        - ztemp     : Template for Z-matrix coordinates (for internal coordinate forces)
        - pids      : Information for building interaction name -> parameter number hashtable

        @param[in] tempdir The temporary directory to be prepared.
        @todo Someday I'd like to use WHAM to put AIMD simulations in. :)
        @todo The fitatoms shouldn't be the first however many atoms, it should be a list.

        """
        if tempdir == None:
            tempdir = self.tempdir
        # Create the temporary directory
        abstempdir = os.path.join(self.root,self.tempdir)
        # Link the necessary programs into the temporary directory
        os.symlink(os.path.join(options['gmxpath'],"mdrun"+options['gmxsuffix']),os.path.join(abstempdir,"mdrun"))
        os.symlink(os.path.join(options['gmxpath'],"grompp"+options['gmxsuffix']),os.path.join(abstempdir,"grompp"))
        os.symlink(os.path.join(options['gmxpath'],"g_energy"+options['gmxsuffix']),os.path.join(abstempdir,"g_energy"))
        os.symlink(os.path.join(options['gmxpath'],"g_traj"+options['gmxsuffix']),os.path.join(abstempdir,"g_traj"))
        # Link the run files
        os.symlink(os.path.join(self.root,self.simdir,"shot.mdp"),os.path.join(abstempdir,"shot.mdp"))
        # Write the trajectory to the temp-directory
        self.traj.write(os.path.join(abstempdir,"all.gro"))
        os.symlink(os.path.join(self.root,self.simdir,"topol.top"),os.path.join(abstempdir,"topol.top"))
        # Print out the first conformation in all.gro to use as conf.gro
        self.traj.write(os.path.join(abstempdir,"conf.gro"),subset=[0])
        if self.qmboltz > 0.0:
            # QM Boltzmann Weights
            gmxx2_print(os.path.join(abstempdir,"qmboltz"),self.qmboltz_wts,"double")
            with open(os.path.join(abstempdir,"bp"),'w') as f: f.write("%.3f" % self.qmboltz)
        if self.whamboltz == True:
            # WHAM Boltzmann Weights
            # Might as well note here, they are compatible with QM Boltzmann weights! :)
            gmxx2_print(os.path.join(abstempdir,"whamboltz"),self.whamboltz_wts[:self.ns],"double")
        if self.sampcorr == True:
            # Create a file called "sampcorr" if the sampcorr option is turned on.
            # This functionality is VERY VERY OLD
            open(os.path.join(abstempdir,"sampcorr"),'w').close()
        if self.covariance == False:
            # Gromacs-X2 defaults to using the covariance, so we can turn it off here
            open(os.path.join(abstempdir,"NoCovariance"),'w').close()
        if self.fitatoms != self.natoms:
            # The number of fitting atoms (since the objective function is built internally)
            with open(os.path.join(abstempdir,"fitatoms"),'w') as f: f.write("%i\n" % self.fitatoms)
        # Print the QM energies and forces in such a way that Gromacs understands.
        gmxx2_print(os.path.join(abstempdir,"energyqm"),self.eqm, "double")
        gmxx2_print(os.path.join(abstempdir,"forcesqm"),self.fqm.reshape(-1), "double")
        if os.path.exists(os.path.join(self.simdir,"ztemp")):
            shutil.copy2(os.path.join(self.simdir,"ztemp"),os.path.join(abstempdir,"ztemp"))
        # Print the parameter ID information.  
        # Todo: Inactivate certain pids for energy decomposition analysis.  I shouldn't have to worry about that here.
        pidsfile = open(os.path.join(abstempdir,"pids"),'w')
        print >> pidsfile, len(self.FF.map)+1,
        #============================================================#
        # E0 (mean energy gap) is no longer an adjustable parameter  #
        # because we subtract it out.  However, a "zero" entry is    #
        # required by the hash tables in gromacs, because a "zero"   #
        # dictionary lookup means that the parameter does not exist. #
        # For now let's keep it, but if there's a better solution we #
        # should get rid of it.                                      #
        #============================================================#
        print >> pidsfile, "E0 0",
        for i in self.FF.map:
            # LPW For the very reason described above, we add one to the
            # parameter number.
            print >> pidsfile, i, self.FF.map[i]+1,
        print >> pidsfile
        pidsfile.close()
        
    def get(self,mvals,AGrad=False,AHess=False,tempdir=None):
        """ Calls the modified GROMACS and collects the objective function contribution.

        First we create the force field using the parameter values that were
        passed in.  Note that we may pass in physical parameters directly
        and bypass the coordinate transformation by setting self.usepvals to True.

        The physical parameters are printed to 'pvals' for GROMACS to read -
        of course GROMACS knows the parameters already, but this facilitates
        retrieval from the low level subroutines.

        Several switches are printed to files, such as:
        - 'FirstDerivativesOnly' to prevent computation of the Hessian
        - 'NoDerivatives' to prevent computation of the Hessian AND the gradient

        GROMACS is called in the callgmxx2() method.

        The output files are then parsed for the objective function and its
        derivatives are read in.  The answer is passed out as a dictionary:
        {'X': Objective Function, 'G': Gradient, 'H': Hessian}

        @param[in] mvals Mathematical parameter values
        @param[in] AGrad Switch to turn on analytic gradient
        @param[in] AHess Switch to turn on analytic Hessian
        @param[in] tempdir Temporary directory for running computation
        @return Answer Contribution to the objective function

        @todo Some of these files don't need to be printed, they can be passed
        to GROMACS as arguments.  Let's think about this some more.
        @todo Currently I have no way to pass out the qualitative indicators.

        """
        if tempdir == None:
            tempdir = self.tempdir
        Answer = {}
        cwd = os.getcwd()
        # Create the new force field!!
        pvals = self.FF.make(tempdir,mvals,self.usepvals)
        gmxx2_print(os.path.join(os.path.join(self.root,tempdir,"pvals")),append([0],pvals),"double")
        os.chdir(os.path.join(self.root,tempdir))
        if AHess:
            AGrad = True
            remove_if_exists("FirstDerivativesOnly")
            remove_if_exists("NoDerivatives")
        elif AGrad:
            with open("FirstDerivativesOnly",'w') as f: f.close()
            remove_if_exists("NoDerivatives")
        else:
            with open("NoDerivatives",'w') as f: f.close()
        print "GMXX2: %s\r" % tempdir,
        self.callgmxx2()
        # Parse the output files
        for line in open('e2f2bc').readlines():
            sline = line.split()
            if match('^Objective Function',line):
                X = floatornan(sline[-1])
                Answer['X'] = X
                # Pass out the qualitative indicator
            elif match('^Energy Error \(kJ/mol\)',line):
                E = floatornan(sline[-1])
                self.e_err = E
            elif match('^Force Error',line):
                F = floatornan(sline[-1])
                self.f_err = F
        # Derivatives in the physical parameter values
        GR = array([float(i.split()[2]) for i in open('a1dbc').readlines()])
        # If we're working in the rescaled space, rescale and transform to the 'mathematical' derivatives
        # Otherwise just use the raw values.
        if self.usepvals:
            Answer['G'] = col(GR.copy())
        else:
            G = flat(mat(self.FF.tm)*col(GR.copy()))
            Answer['G'] = G
        # Reads in second derivatives from GROMACS
        # Even if we're just calling the gradients, we want the diagonal Hessian elements as well!
        HR = zeros((self.FF.np, self.FF.np))
        for line in open('a2dbc').readlines():
            i = int(line.split()[1])
            j = int(line.split()[2])
            HR[i, j] = float(line.split()[3])
        if self.usepvals:
            Answer['H'] = mat(HR.copy())
        else:
            H = mat(self.FF.tm)*mat(HR.copy())*mat(self.FF.tmI)                       # Projects charge constraints out of second derivative
            Answer['H'] = H
        os.chdir(cwd)
        return Answer
    
    def callgmxx2(self):
        """ Call the modified GROMACS! """
        rm_gmx_baks('.')
        # Call grompp followed by mdrun.
        o, e = subprocess.Popen(["./grompp", "-f", "shot.mdp"],stdout=PIPE,stderr=PIPE).communicate()
        o, e = subprocess.Popen(["./mdrun", "-fortune", "-o", "shot.trr", "-rerun", "all.gro", "-rerunvsite"], stdout=PIPE, stderr=PIPE).communicate()
