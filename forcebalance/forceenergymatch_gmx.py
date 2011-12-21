""" @package forceenergymatch_gmx Force and energy matching with interface to modified GROMACS.

In order for us to obtain the objective function in force and energy
matching, we loop through the snapshots, compute the energy and force
(as well as its derivatives), and sum them up.  The details of the
process are complicated and I won't document them here.  The contents
of this package (mainly the ForceEnergyMatch_GMX class) allows us to
call the modified GROMACS to compute the objective function for us.

@author  Lee-Ping Wang
@date    12/2011

"""

import os
import shutil
from nifty import col, flat, floatornan
from numpy import append, array, mat, zeros
from gmxio import gmxprint
from re import match
import subprocess
from subprocess import PIPE
from forceenergymatch import ForceEnergyMatch

def set_gmx_paths(me,options):
    """ Set the gmxrunpath, gmxtoolpath and gmxsuffix attributes of a class.
    
    \param[in] me The class whose attributes we want to set.
    \param[in] options Simulation options dictionary
    
    """
    ##  The path for main GROMACS executables like mdrun, grompp (if linking to just-built binaries)
    me.gmxrunpath  = options['gmxrunpath'] != None and options['gmxrunpath'] or options['gmxpath']
    ##  The path for GROMACS tools like g_energy (if linking to just-built binaries)
    me.gmxtoolpath = options['gmxrunpath'] != None and options['gmxtoolpath'] or options['gmxpath']
    ##  Suffix for GROMACS executables
    me.gmxsuffix   = options['gmxsuffix']

class ForceEnergyMatch_GMX(ForceEnergyMatch):

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
        """Instantiation of ForceEnergyMatch_GMX.

        Several important things happen here:
        - We load in the coordinates from 'all.gro'.
        - We prepare the temporary directory.
        
        """
        
        # Initialize the superclass. :)
        super(ForceEnergyMatch_GMX,self).__init__(options,sim_opts,forcefield)
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## The number of particles (includes atoms, Drudes, and virtual sites)
        self.nparticles  = 0
        ## The number of true atoms 
        self.natoms      = 0
        set_gmx_paths(self, options)
        ## Path to the all.gro file
        self.grofnm      = os.path.join(self.simdir,"all.gro")
        ## The all.gro file is loaded into memory
        self.allgro      = self.load_gro_files(self.grofnm)

        # Backup the temporary directory if desired, delete what was there
        # and prepare it for this job
        if 'backup_temp_directory' in options:
            self.backup_temp_directory()
        shutil.rmtree(os.path.join(self.root,self.tempdir),ignore_errors=True)
        self.prepare_temp_directory()
        
    def load_gro_files(self,grofnm):
        """ Load the entire trajectory into memory.  The data is not
        interpreted, but the trajectory is split into multiple .gro
        files.
        
        Actually this is not very useful, but I prefer to perform the
        coordinate file manipulations in-house rather than relying on
        a program like trjconv.

        @param[in] grofnm The .gro file to be loaded.
        """
        ln = 0
        sn = 0
        thisgro = []
        allgro  = []
        for line in open(os.path.join(self.root,grofnm)):
            thisgro.append(line)
            if ln == 1:
                self.nparticles = int(line.strip())
            elif ln == self.nparticles + 2:
                ln = -1
                allgro.append(thisgro)
                thisgro = []
                sn += 1
                if sn == self.ns:
                    break
            ln += 1
        return allgro
    
    def backup_temp_directory(self):
        """ Back up the temporary directory.

        LPW Todo: This is a candidate for moving up to the superclass.
        
        """
        cwd = os.getcwd()
        if not os.path.exists(os.path.join(self.root,'backups')):
            os.makedirs(os.path.join(self.root,'backups'))
        if os.path.exists(os.path.join(self.root,self.tempdir)):
            print "Backing up:", self.tempdir
            os.chdir(os.path.join(self.root,"temp"))
            # I could use the tarfile module here
            subprocess.call(["tar","cjf",os.path.join(self.root,'backups',"%s.tar.bz2" % self.name),self.name,"--remove-files"])
            os.chdir(cwd)
            
    def prepare_temp_directory(self,tempdir=None):
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
        @todo Currently we don't have a switch to turn the covariance off. Should be simple to add in.
        @todo Someday I'd like to use WHAM to put AIMD simulations in. :)
        @todo The fitatoms shouldn't be the first however many atoms, it should be a list.

        """
        if tempdir == None:
            tempdir = self.tempdir
        # Create the temporary directory
        abstempdir = os.path.join(self.root,self.tempdir)
        os.makedirs(abstempdir)
        # Link the necessary programs into the temporary directory
        os.symlink(os.path.join(self.gmxrunpath,"mdrun"+self.gmxsuffix),os.path.join(abstempdir,"mdrun"))
        os.symlink(os.path.join(self.gmxrunpath,"grompp"+self.gmxsuffix),os.path.join(abstempdir,"grompp"))
        os.symlink(os.path.join(self.gmxtoolpath,"g_energy"+self.gmxsuffix),os.path.join(abstempdir,"g_energy"))
        os.symlink(os.path.join(self.gmxtoolpath,"g_traj"+self.gmxsuffix),os.path.join(abstempdir,"g_traj"))
        # Link the run files
        os.symlink(os.path.join(self.root,self.simdir,"shot.mdp"),os.path.join(abstempdir,"shot.mdp"))
        allgro_out = open(os.path.join(abstempdir,"all.gro"),'w')
        for gro in self.allgro:
            for line in gro:
                allgro_out.write(line)
        os.symlink(os.path.join(self.root,self.simdir,"topol.top"),os.path.join(abstempdir,"topol.top"))
        # Print out the first conformation in all.gro to use as conf.gro
        confgro = open(os.path.join(abstempdir,"conf.gro"),'w')
        for line in self.allgro[0]:
            confgro.write(line)
        confgro.close()
        # Print a number of things that Gromacs likes to use
        if self.qmboltz > 0.0:
            # QM Boltzmann Weights
            gmxprint(os.path.join(abstempdir,"qmboltz"),self.qmboltz_wts,"double")
            with open(os.path.join(abstempdir,"bp"),'w') as f: f.write("%.3f" % self.qmboltz)
        if self.whamboltz == True:
            # WHAM Boltzmann Weights
            # Might as well note here, they are compatible with QM Boltzmann weights! :)
            gmxprint(os.path.join(abstempdir,"whamboltz"),self.whamboltz_wts[:self.ns],"double")
        if self.sampcorr == True:
            # Create a file called "sampcorr" if the sampcorr option is turned on.
            # This functionality is VERY VERY OLD
            open(os.path.join(abstempdir,"sampcorr"),'w').close()
        if self.covariance == False:
            # Gromacs-X2 defaults to using the covariance, so we can turn it off here
            open(os.path.join(abstempdir,"NoCovariance"),'w').close()
        fqmm = self.fqm.reshape(self.ns, -1)
        if self.fitatoms > 0:
            # Indicate to Gromacs that we're only fitting the first however-many atoms.
            print "We're only fitting the first %i atoms" % self.fitatoms
            with open(os.path.join(abstempdir,"fitatoms"),'w') as f: f.write("%i\n" % self.fitatoms)
            print "The quantum force matrix appears to contain more components (%i) than those being fit (%i)." % (fqmm.shape[1], 3*self.fitatoms)
            print "Pruning the quantum force matrix..."
        else:
            self.fitatoms = self.natoms
        fqmprint    = fqmm[:, :3*self.fitatoms].copy().reshape(-1)
        # Print the QM energies and forces in such a way that Gromacs understands.
        gmxprint(os.path.join(abstempdir,"energyqm"),self.eqm, "double")
        gmxprint(os.path.join(abstempdir,"forcesqm"),fqmprint, "double")
        if os.path.exists(os.path.join(self.simdir,"ztemp")):
            shutil.copy2(os.path.join(self.simdir,"ztemp"),os.path.join(abstempdir,"ztemp"))
        # Print the parameter ID information.  
        # Todo: Inactivate certain pids for energy decomposition analysis.  I shouldn't have to worry about that here.
        pidsfile = open(os.path.join(abstempdir,"pids"),'w')
        print >> pidsfile, len(self.FF.map)+1,
        # E0 (mean energy gap) is no longer an adjustable
        # parameter because we subtract it out.  However, a "zero"
        # entry is required by the hash tables in gromacs, because a
        # "zero" dictionary lookup means that the parameter does not
        # exist.  For now let's keep it, but if there's a better
        # solution we should get rid of it.
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

        GROMACS is called in the callgmx() method.

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
        self.FF.make(tempdir,mvals,self.usepvals)
        gmxprint(os.path.join(os.path.join(self.root,tempdir,"pvals")),append([0],self.FF.pvals),"double")
        os.chdir(os.path.join(self.root,tempdir))
        if AHess:
            AGrad = True
        elif AGrad:
            with open("FirstDerivativesOnly",'w') as f: f.close()
        else:
            with open("NoDerivatives",'w') as f: f.close()
        print "GMX: %s\r" % tempdir,
        self.callgmx()
        # Parse the output files
        for line in open('e2f2bc').readlines():
            sline = line.split()
            if match('^Objective Function',line):
                X = floatornan(sline[-1])
                Answer['X'] = X
            # Do we want to pass out the qualitative indicators?
            # elif match('^Energy Error \(kJ/mol\)',line):
            #     E = floatornan(sline[-1])
            #     Answer['E'] = E
            # elif match('^Force Error',line):
            #     F = floatornan(sline[-1])
            #     Answer['F'] = F
        #if AGrad or AHess:
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
    
    def callgmx(self):
        """ Call the modified GROMACS! """
        # Delete the #-prepended files that GROMACS likes to make
        for root, dirs, files in os.walk('.'):
            for file in files:
                if match('^#',file):
                    os.remove(file)
        # Call grompp followed by mdrun.
        o, e = subprocess.Popen(["./grompp", "-f", "shot.mdp"],stdout=PIPE,stderr=PIPE).communicate()
        o, e = subprocess.Popen(["./mdrun", "-fortune", "-o", "sshot.trr", "-rerun", "all.gro", "-rerunvsite"], stdout=PIPE, stderr=PIPE).communicate()
