"""@package forcebalance.counterpoise

Match an empirical potential to the counterpoise correction for basis set superposition error (BSSE).

Here we test two different functional forms: a three-parameter
Gaussian repulsive potential and a four-parameter Gaussian which
goes smoothly to an exponential.  The latter can be written in two
different ways - one which gives us control over the exponential,
the switching distance and the Gaussian decay constant, and
another which gives us control over the Gaussian and the switching
distance.  They are called 'CPGAUSS', 'CPEXPG', and 'CPGEXP'.  I think
the third option is the best although our early tests have
indicated that none of the force fields perform particularly well
for the water dimer.

This subclass of Target implements the 'get' method.

@author Lee-Ping Wang
@date 12/2011
"""

import os
import sys
from re import match
from forcebalance.target import Target
from nifty import *
import numpy as np
from forcebalance.output import getLogger
logger = getLogger(__name__)

class Counterpoise(Target):
    """ Target subclass for matching the counterpoise correction."""

    def __init__(self,options,tgt_opts,forcefield):
        """ To instantiate Counterpoise, we read the coordinates and counterpoise data."""
        # Initialize the superclass. :)
        super(Counterpoise,self).__init__(options,tgt_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        ## Number of snapshots
        self.set_option(tgt_opts,'shots','ns')
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## XYZ elements and coordinates
        self.elem, self.xyzs = self.loadxyz(os.path.join(self.root,self.tgtdir,'all.xyz'))
        ## Counterpoise correction data
        self.cpqm = self.load_cp(os.path.join(self.root,self.tgtdir,'cp.dat'))
        
    def loadxyz(self,fnm):
        """ Parse an XYZ file which contains several xyz coordinates, and return their elements.

        @param[in] fnm The input XYZ file name
        @return elem A list of chemical elements in the XYZ file
        @return xyzs A list of XYZ coordinates (number of snapshots times number of atoms)
        @todo I should probably put this into a more general library for reading coordinates.
        """
        
        logger.info("Loading XYZ file!\n")
        xyz  = []
        xyzs = []
        elem = []
        an   = 0
        sn   = 0
        for line in open(fnm):
            strip = line.strip()
            sline = line.split()
            if match('^[0-9]+$',strip):
                ## Number of atoms
                self.na = int(strip)
            elif match('[A-Z][a-z]*( +[-+]?([0-9]*\.[0-9]+|[0-9]+)){3}$',strip):
                xyz.append([float(i) for i in sline[1:]])
                if len(elem) < self.na:
                    elem.append(sline[0])
                an += 1
                if an == self.na:
                    xyzs.append(np.array(xyz))
                    sn += 1
                    if sn == self.ns:
                        break
                    xyz = []
                    an  = 0
        self.ns = len(xyzs)
        return elem, xyzs

    def load_cp(self,fnm):
        """ Load in the counterpoise data, which is easy; the file
        consists of floating point numbers separated by newlines.  """
        logger.info("Loading CP Data!\n")
        return np.array([float(i.strip()) for i in open(fnm).readlines()])[:self.ns]

    def get(self,mvals,AGrad=False,AHess=False):
        """Gets the objective function for fitting the counterpoise correction.

        As opposed to AbInitio_GMXX2, which calls an external program,
        this script actually computes the empirical interaction given the
        force field parameters.

        It loops through the snapshots and atom pairs, and computes pairwise
        contributions to an energy term according to hard-coded functional forms.

        One potential issue is that we go through all atom pairs instead of
        looking only at atom pairs between different fragments.  This means that
        even for two infinitely separated fragments it will predict a finite
        CP correction.  While it might be okay to apply such a potential in practice,
        there will be some issues for the fitting.  Thus, we assume the last snapshot
        to be CP-free and subtract that value of the potential back out.

        Note that forces and parametric derivatives are not implemented.

        @param[in] mvals Mathematical parameter values
        @param[in] AGrad Switch to turn on analytic gradient (not implemented)
        @param[in] AHess Switch to turn on analytic Hessian (not implemented)
        @return Answer Contribution to the objective function
        
        """

        # Create the force field physical values from the mathematical values
        pvals = self.FF.create_pvals(mvals)
        cpmm = []
        logger.info("CPMM: %s   \r" % self.name)
        # Loop through the snapshots
        for s in range(self.ns):
            xyz = self.xyzs[s] # Harvest the xyz. :)
            cpmm_i = 0.0
            # Loop through atom pairs
            for i in range(self.na):
                for j in range(i+1,self.na):
                    # Compute the distance between 2 atoms
                    # and the names of the elements involved
                    dx = np.linalg.norm(xyz[i,:]-xyz[j,:])
                    ai = self.elem[i]
                    aj = self.elem[j]
                    aiaj = ai < aj and "%s%s" % (ai,aj) or "%s%s" % (aj,ai)
                    pidlist = ['CPGAUSSA','CPGAUSSB','CPGAUSSC']
                    if all([k+aiaj in self.FF.map for k in pidlist]):
                        # Look up the parameter values.  This might be something like 'CPGAUSSBLiCl'.
                        A, B, C = [self.FF.pvals[self.FF.map[k+aiaj]] for k in pidlist]
                        # Actually compute the interaction.
                        cpmm_i += A * np.exp (-B * (dx - C)**2)
                    # This is repeated for different interaction types...
                    pidlist = ['CPGEXPA', 'CPGEXPB', 'CPGEXPG', 'CPGEXPX']
                    if all([k+aiaj in self.FF.map for k in pidlist]):
                        A, B, G, X0 = [self.FF.pvals[self.FF.map[k+aiaj]] for k in pidlist]
                        a  = 2*A*(X0-B)
                        b  = A*(X0**2-B**2)+G
                        if dx < X0:
                            cpmm_i += np.exp(-a*dx+b)
                        else:
                            cpmm_i += np.exp(-A*(dx-B)**2+G)
                    pidlist = ['CPEXPGA1','CPEXPGB','CPEXPGX0','CPEXPGA2']
                    if all([k+aiaj in self.FF.map for k in pidlist]):
                        A1, B, X0, A2 = [self.FF.pvals[self.FF.map[k+aiaj]] for k in pidlist]
                        B2 = 2 * A2 * X0 - A1
                        G  = B - A2 * X0**2
                        if dx < X0:
                            cpmm_i += np.exp(-A1*dx+B)
                        else:
                            cpmm_i += np.exp(-A2*dx**2 + B2*dx + G)
            cpmm.append(cpmm_i)
        cpmm = np.array(cpmm)
        cpmm -= cpmm[-1] # This prevents craziness from happening
        # Write the results to a file for plotting
        with wopen('results') as f: f.writelines(["% .4f % .4f\n" % (cpmm[i],self.cpqm[i]) for i in range(len(cpmm))])
        # Compute the difference between MM and QM counterpoise corrections
        dcp  = cpmm - self.cpqm
        # Build the final answer and return it
        Answer = {'X': np.dot(dcp,dcp),
                  'G': np.zeros(self.FF.np),
                  'H': np.zeros((self.FF.np,self.FF.np))
                  }
        return Answer
