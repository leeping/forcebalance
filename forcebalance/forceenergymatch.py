"""Force and energy matching module.

Performs actions that are relevant to any force and energy
matching job (currently this means reading in the reference
data and determining the Boltzmann weights).

@author Lee-Ping Wang
@date 12/2011
"""

import os
import shutil
from nifty import col, eqcgmx, flat, floatornan, fqcgmx, kb, printcool
from numpy import append, array, exp, log, mat, mean, where, zeros
from gmxio import gmxprint
from fitsim import FittingSimulation
from re import match
import subprocess
from subprocess import PIPE

class ForceEnergyMatch(FittingSimulation):

    """ Subclass of FittingSimulation for force and energy matching.

    In force and energy matching, we introduce the following concepts:
    - The number of snapshots
    - The reference energies and forces (eqm, fqm) and the file they belong in (qdata.txt)
    - The sampling simulation energies (emd0)
    - The WHAM Boltzmann weights (these are computed externally and passed in)
    - The QM Boltzmann weights (computed internally using the difference between eqm and emd0)

    There are also these little details:
    - Switches for whether to turn on certain Boltzmann weights (they stack)
    - Temperature for the QM Boltzmann weights
    - Whether to fit a subset of atoms

    Note that this subclass does not contain the 'get' method.  """
    
    def __init__(self,options,sim_opts,forcefield):
        """Instantiation of the subclass.

        We begin by instantiating the superclass here and also
        defining a number of core concepts for energy / force
        matching.

        @todo Obtain the number of true atoms (or the particle -> atom mapping)
        from the force field.
        """
        
        # Initialize the SuperClass!
        super(ForceEnergyMatch,self).__init__(options,sim_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        
        ## Number of snapshots
        self.ns            = sim_opts['shots']
        ## Whether to use WHAM Boltzmann weights
        self.whamboltz     = sim_opts['whamboltz']
        ## Whether to use the Sampling Correction
        self.sampcorr      = sim_opts['sampcorr']
        ## Whether to use the Covariance Matrix
        self.covariance    = sim_opts['covariance']
        ## Whether to use QM Boltzmann weights
        self.qmboltz       = sim_opts['qmboltz']
        ## The temperature for QM Boltzmann weights
        self.qmboltztemp   = sim_opts['qmboltztemp']
        ## Number of atoms that we are fitting
        self.fitatoms      = sim_opts['fitatoms']
        ## The proportion of energy vs. force.
        self.efweight      = sim_opts['efweight']
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        
        ## WHAM Boltzmann weights
        self.whamboltz_wts = []
        ## QM Boltzmann weights
        self.qmboltz_wts   = []
        ## Reference (QM) energies
        self.eqm           = []
        ## Energies of the sampling simulation
        self.emd0          = []
        ## Reference (QM) forces
        self.fqm           = []
        ## The qdata.txt file that contains the QM energies and forces
        self.qfnm = os.path.join(self.simdir,"qdata.txt")
        ## The number of true atoms 
        self.natoms      = 0
        ## Qualitative Indicator: average energy error (in kJ/mol)
        self.e_err = 0.0
        ## Qualitative Indicator: average force error (fractional)
        self.f_err = 0.0
        
        # Read in the reference data
        self.readrefdata()
        
    def readrefdata(self):
        
        """ Read the reference data (for force and energy matching)
        from a file such as qdata.txt.

        @todo Add an option for picking any slice out of
        qdata.txt, helpful for cross-validation
        
        @todo Closer integration of reference data with program -
        leave behind the qdata.txt format?  (For now, I like the
        readability of qdata.txt)

        After reading in the information from qdata.txt, it is converted
        into the GROMACS energy units (kind of an arbitrary choice);
        forces (kind of a misnomer in qdata.txt) are multipled by -1
        to convert gradients to forces.

        We also subtract out the mean energies of all energy arrays
        because energy/force matching does not account for zero-point
        energy differences between MM and QM (i.e. energy of electrons
        in core orbitals).

        The configurations in force/energy matching typically come
        from a the thermodynamic ensemble of the MM force field at
        some temperature (by running MD, for example), and for many
        reasons it is helpful to introduce non-Boltzmann weights in
        front of these configurations.  There are two options: WHAM
        Boltzmann weights (for combining the weights of several
        simulations together) and QM Boltzmann weights (for converting
        MM weights into QM weights).  Note that the two sets of weights
        'stack'; i.e. they can be used at the same time.

        A 'hybrid' ensemble is possible where we use 50% MM and 50% QM
        weights.  Please read more in LPW and Troy Van Voorhis, JCP
        Vol. 133, Pg. 231101 (2010), doi:10.1063/1.3519043.
        
        @todo As of now (Dec 2011) the WHAM Boltzmann weights are
        generated by external scripts (wanalyze.py and
        make-wham-data.sh) and passed in; tighter integration would be
        nice.

        Finally, note that using non-Boltzmann weights degrades the
        statistical information content of the snapshots.  This
        problem will generally become worse if the ensemble to which
        we're reweighting is dramatically different from the one we're
        sampling from.  We end up with a set of Boltzmann weights like
        [1e-9, 1e-9, 1, 1e-9, 1e-9 ... ] and this is essentially just
        one snapshot.  I believe Troy is working on something to cure
        this problem.

        Here, we have a measure for the information content of our snapshots,
        which comes easily from the definition of information entropy:

        S = -1*Sum_i(P_i*log(P_i))
        InfoContent = exp(-S)

        With uniform weights, InfoContent is equal to the number of snapshots;
        with horrible weights, InfoContent is closer to one.

        """
        # Parse the qdata.txt file
        for line in open(os.path.join(self.root,self.qfnm)):
            sline = line.split()
            if len(sline) == 0: continue
            elif sline[0] == 'ENERGY':
                self.eqm.append(float(sline[1]))
            elif sline[0] == 'EMD0':
                self.emd0.append(float(sline[1]))
            elif sline[0] == 'FORCES':
                self.fqm.append([float(i) for i in sline[1:]])
            if len(self.eqm) and len(self.emd0) and len(self.fqm) == self.ns:
                # Break out of the loop if we've already gotten "ns" number of energies and forces.
                break
        self.ns = len(self.eqm)
        # Turn everything into arrays, convert to kJ/mol, and subtract the mean energy from the energy arrays
        self.eqm = array(self.eqm)
        self.eqm *= eqcgmx
        self.eqm -= mean(self.eqm)
        self.fqm = array(self.fqm)
        self.fqm *= fqcgmx
        self.natoms = self.fqm.shape[1]/3
        self.emd0 = array(self.emd0)
        self.emd0 -= mean(self.emd0)
        if self.whamboltz == True:
            self.whamboltz_wts = array([float(i.strip()) for i in open(os.path.join(self.root,self.simdir,"wham-weights.txt")).readlines()])
            #   This is a constant pre-multiplier in front of every snapshot.
            bar = printcool("Using WHAM MM Boltzmann weights.", color=3)
            if os.path.exists(os.path.join(self.root,self.simdir,"wham-master.txt")):
                whaminfo = open(os.path.join(self.root,self.simdir,"wham-master.txt")).readlines()
                print "From wham-master.txt, I can see that you're using %i generations" % len(whaminfo)
                print "Relative weight of each generation:"
                shotcounter = 0
                for line in whaminfo:
                    sline = line.split()
                    genshots = int(sline[2])
                    weight = sum(self.whamboltz_wts[shotcounter:shotcounter+genshots])/sum(self.whamboltz_wts)
                    print " %s, %i snapshots, weight %.3e" % (sline[0], genshots, weight)
                    shotcounter += genshots
            else:
                print "Oops... WHAM files don't exist?"
            print bar
        if self.qmboltz > 0:
            # LPW I haven't revised this section yet
            # Do I need to?
            logboltz = -(self.eqm - mean(self.eqm) - self.emd0 + mean(self.emd0)) / kb / self.qmboltztemp
            logboltz -= max(logboltz) # Normalizes boltzmann weights
            self.qmboltz_wts = exp(logboltz)
            self.qmboltz_wts /= sum(self.qmboltz_wts)
            # where simply gathers the nonzero values otherwise we get lots of NaNs
            qbwhere = self.qmboltz_wts[where(self.qmboltz_wts)[0]]
            # Compute ze InfoContent!
            qmboltzent = -sum(qbwhere*log(qbwhere))
            print "Quantum Boltzmann weights are ON, the formula is exp(-b(E_qm-E_mm)),",
            print "distribution entropy is %.3f, equivalent to %.2f snapshots" % (qmboltzent, exp(qmboltzent))
            print "%.1f%% is mixed into the MM boltzmann weights." % (self.qmboltz*100)
            
    def indicate(self):
        print "Sim: %-15s E_err(kJ/mol)= %10.4f F_err(%%)= %10.4f" % (self.name, self.e_err, self.f_err*100)
