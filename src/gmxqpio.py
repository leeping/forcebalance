""" @package gmxio GROMACS input/output.

@todo Even more stuff from forcefield.py needs to go into here.

@author Lee-Ping Wang
@date 12/2011
"""

import os
from re import match, sub
from nifty import isint, isfloat, _exec, warn_press_key, printcool_dictionary
import numpy as Np
from molecule import Molecule
from copy import deepcopy
import itertools
from fitsim import FittingSimulation
from collections import OrderedDict
from finite_difference import *

# Prerequisite: Water monomer geometry with:
# - Oxygen atom at the origin
# - Molecule in the xz-plane with C2 axis along z
# - Hydrogen molecules have positive z
def get_monomer_properties(print_stuff=0):
    # Multiply a quantity in nm to convert to a0
    nm_to_a0 = 1./0.05291772108
    # Multiply a quantity in e*a0 to convert to Debye
    ea0_to_debye = 0.393430307
    os.system("rm -rf *.log \#*")
    _exec(["./grompp"], print_command=False)
    _exec(["./mdrun"], outfnm="mdrun.txt", print_command=False)
    x = []
    q = []
    for line in open("confout.gro").readlines():
        sline = line.split()
        if len(sline) == 9 and isfloat(sline[3]) and isfloat(sline[4]) and isfloat(sline[5]):
            x.append([float(i) for i in sline[3:6]])
    for line in open("charges.log").readlines():
        sline = line.split()
        if 'AtomNr' in line:
            q.append(float(sline[5]))
    mode = 0
    a = []
    for line in open("mdrun.txt").readlines():
        if mode == 1:
            sline = line.split()
            if len(sline) == 3:
                if isfloat(sline[0]) and isfloat(sline[1]) and isfloat(sline[2]):
                    a.append([float(i) for i in sline])
                elif any(["nan" in s for s in sline[:3]]):
                    a.append([1e10,1e10,1e10])
        if "Computing the polarizability tensor" in line:
            mode = 1
    x = Np.array(x)
    q = Np.array(q)
    a = Np.array(a)
    Dip = Np.zeros(3,dtype=float)
    QuadXX = 0.0
    QuadYY = 0.0
    QuadZZ = 0.0
    OctXXZ = 0.0
    OctYYZ = 0.0
    OctZZZ = 0.0
    for i in range(q.shape[0]):
        Dip += x[i]*q[i]*nm_to_a0/ea0_to_debye
        xx = x[i,0]*x[i,0]
        yy = x[i,1]*x[i,1]
        zz = x[i,2]*x[i,2]
        z  = x[i,2]
        r2 = Np.dot(x[i,:],x[i,:])
        QuadXX += 0.5*q[i]*(2*xx - yy - zz) * 10 * nm_to_a0 / ea0_to_debye
        QuadYY += 0.5*q[i]*(2*yy - xx - zz) * 10 * nm_to_a0 / ea0_to_debye
        QuadZZ += 0.5*q[i]*(2*zz - xx - yy) * 10 * nm_to_a0 / ea0_to_debye
        OctXXZ += 0.5*q[i]*z*(5*xx-r2) * 100 * nm_to_a0 / ea0_to_debye
        OctYYZ += 0.5*q[i]*z*(5*yy-r2) * 100 * nm_to_a0 / ea0_to_debye
        OctZZZ += 0.5*q[i]*z*(5*zz-3*r2) * 100 * nm_to_a0 / ea0_to_debye
    DipZ = Dip[2]
    AlphaXX = a[0,0]
    AlphaYY = a[1,1]
    AlphaZZ = a[2,2]
    # Quantities taken from Niu (2001) and Berne (1994)
    DipZ0 = 1.855
    QuadXX0 =  2.51
    QuadYY0 = -2.63
    QuadZZ0 =  0.11
    Quad0   = Np.sqrt((QuadXX0**2 + QuadYY0**2 + QuadZZ0**2)/3)
    OctXXZ0 =  2.58
    OctYYZ0 = -1.24
    OctZZZ0 = -1.35
    Oct0   = Np.sqrt((OctXXZ0**2 + OctYYZ0**2 + OctZZZ0**2)/3)
    AlphaXX0 = 10.32
    AlphaYY0 =  9.56
    AlphaZZ0 =  9.91
    Alpha0   = Np.sqrt((AlphaXX0**2 + AlphaYY0**2 + AlphaZZ0**2)/3)
    Err_DipZ = ((DipZ-DipZ0)/DipZ0)**2
    Err_QuadXX = ((QuadXX-QuadXX0)/Quad0)**2
    Err_QuadYY = ((QuadYY-QuadYY0)/Quad0)**2
    Err_QuadZZ = ((QuadZZ-QuadZZ0)/Quad0)**2
    Err_OctXXZ = ((OctXXZ-OctXXZ0)/Oct0)**2
    Err_OctYYZ = ((OctYYZ-OctYYZ0)/Oct0)**2
    Err_OctZZZ = ((OctZZZ-OctZZZ0)/Oct0)**2
    Err_AlphaXX = ((AlphaXX-AlphaXX0)/Alpha0)**2
    Err_AlphaYY = ((AlphaYY-AlphaYY0)/Alpha0)**2
    Err_AlphaZZ = ((AlphaZZ-AlphaZZ0)/Alpha0)**2
    Objective   = Err_DipZ + (Err_QuadXX + Err_QuadYY + Err_QuadZZ)/3 + (Err_AlphaXX + Err_AlphaYY + Err_AlphaZZ)/3
    if print_stuff:
        #print "\rvalues (errors): mu_z = % .3f (%.3f) q_xx = % .3f (%.3f) q_yy = % .3f (%.3f) q_zz = % .3f (%.3f) o_xxz = % .3f (%.3f) o_yyz = % .3f (%.3f) o_zzz = % .3f (%.3f) a_xx = % .3f (%.3f) a_yy = % .3f (%.3f) a_zz = % .3f (%.3f)" % (DipZ,Err_DipZ,QuadXX,Err_QuadXX,QuadYY,Err_QuadYY,QuadZZ,Err_QuadZZ,OctXXZ,Err_OctXXZ,OctYYZ,Err_OctYYZ,OctZZZ,Err_OctZZZ,AlphaXX,Err_AlphaXX,AlphaYY,Err_AlphaYY,AlphaZZ,Err_AlphaZZ)
        print "\rvalues (errors): mu_z = % .3f (%.3f) q = % .3f % .3f % .3f (% .3f % .3f % .3f) o = % .3f % .3f % .3f (% .3f % .3f % .3f) a = %.3f %.3f %.3f (%.3f %.3f %.3f) x2 = % .4f" % (DipZ,Err_DipZ,QuadXX,QuadYY,QuadZZ,Err_QuadXX,Err_QuadYY,Err_QuadZZ,OctXXZ,OctYYZ,OctZZZ,Err_OctXXZ,Err_OctYYZ,Err_OctZZZ,AlphaXX,AlphaYY,AlphaZZ,Err_AlphaXX,Err_AlphaYY,Err_AlphaZZ,Objective)
    #Objective   = Err_DipZ + (Err_QuadXX + Err_QuadYY + Err_QuadZZ)/3 + (Err_OctXXZ + Err_OctYYZ + Err_OctZZZ)/3 + (Err_AlphaXX + Err_AlphaYY + Err_AlphaZZ)/3
    Properties = OrderedDict()
    Properties['DipZ'] = DipZ
    Properties['QuadXX'] = QuadXX
    Properties['QuadYY'] = QuadYY
    Properties['QuadZZ'] = QuadZZ
    Properties['OctXXZ'] = OctXXZ
    Properties['OctYYZ'] = OctYYZ
    Properties['OctZZZ'] = OctZZZ
    Properties['AlphaXX'] = AlphaXX
    Properties['AlphaYY'] = AlphaYY
    Properties['AlphaZZ'] = AlphaZZ
    return Properties

class Monomer_QTPIE(FittingSimulation):
    """ Subclass of FittingSimulation for monomer properties of QTPIE (implemented within gromacs WCV branch)."""

    def __init__(self,options,sim_opts,forcefield):
        super(Monomer_QTPIE,self).__init__(options,sim_opts,forcefield)
        self.prepare_temp_directory(options, sim_opts)

        # # Quantities taken from Niu (2001) and Berne (1994)
        # DipZ0 = 1.855
        # QuadXX0 =  2.51
        # QuadYY0 = -2.63
        # QuadZZ0 =  0.11
        # Quad0   = Np.sqrt((QuadXX0**2 + QuadYY0**2 + QuadZZ0**2)/3)
        # OctXXZ0 =  2.58
        # OctYYZ0 = -1.24
        # OctZZZ0 = -1.35
        # Oct0   = Np.sqrt((OctXXZ0**2 + OctYYZ0**2 + OctZZZ0**2)/3)
        # AlphaXX0 = 10.32
        # AlphaYY0 =  9.56
        # AlphaZZ0 =  9.91
        # Alpha0   = Np.sqrt((AlphaXX0**2 + AlphaYY0**2 + AlphaZZ0**2)/3)

        self.ref_moments = OrderedDict()
        self.ref_moments['DipZ'] = 1.855
        self.ref_moments['QuadXX'] =  2.51
        self.ref_moments['QuadYY'] = -2.63
        self.ref_moments['QuadZZ'] =  0.11
        self.ref_moments['OctXXZ'] =  2.58
        self.ref_moments['OctYYZ'] = -1.24
        self.ref_moments['OctZZZ'] = -1.35
        self.ref_moments['AlphaXX'] = 10.32
        self.ref_moments['AlphaYY'] =  9.56
        self.ref_moments['AlphaZZ'] =  9.91
        Quad0   = Np.sqrt((self.ref_moments['QuadXX']**2 + self.ref_moments['QuadYY']**2 + self.ref_moments['QuadZZ']**2)/3)
        Oct0   = Np.sqrt((self.ref_moments['OctXXZ']**2 + self.ref_moments['OctYYZ']**2 + self.ref_moments['OctZZZ']**2)/3)
        Alpha0   = Np.sqrt((self.ref_moments['AlphaXX']**2 + self.ref_moments['AlphaYY']**2 + self.ref_moments['AlphaZZ']**2)/3)
        self.weights = OrderedDict()
        self.weights['DipZ'] = 1.0 / 1.855
        self.weights['QuadXX'] = 1.0 / Quad0**2 / 3
        self.weights['QuadYY'] = 1.0 / Quad0**2 / 3
        self.weights['QuadZZ'] = 1.0 / Quad0**2 / 3
        self.weights['OctXXZ'] = 1.0 / Oct0**2 / 3
        self.weights['OctYYZ'] = 1.0 / Oct0**2 / 3
        self.weights['OctZZZ'] = 1.0 / Oct0**2 / 3
        self.weights['AlphaXX'] = 1.0 / Alpha0**2 / 3
        self.weights['AlphaYY'] = 1.0 / Alpha0**2 / 3
        self.weights['AlphaZZ'] = 1.0 / Alpha0**2 / 3
        
    def indicate(self):
        """ Print qualitative indicator. """
        print "\rSim: %-15s" % self.name

        ref_momvals = self.unpack_moments(self.ref_moments)
        calc_momvals = self.unpack_moments(self.calc_moments)
        PrintDict = OrderedDict()
        i = 0
        for Ord in self.ref_moments:
            if abs(self.calc_moments[Ord]) > 1e-6 or abs(self.ref_moments[Ord]) > 1e-6:
                PrintDict["%s" % (Ord)] = "% 9.3f % 9.3f % 9.3f % 12.5f" % (self.calc_moments[Ord],
                                                                                     self.ref_moments[Ord],
                                                                                     self.calc_moments[Ord]-self.ref_moments[Ord],
                                                                                     (ref_momvals[i] - calc_momvals[i])**2)

            i += 1
                
        printcool_dictionary(PrintDict,title="Moments and/or Polarizabilities, Objective = % .5e\n %-20s %9s %9s %9s %11s" % 
                             (self.objective, "Component", "Calc.", "Ref.", "Delta", "Term"))

        return
        
    def prepare_temp_directory(self, options, sim_opts):
        os.environ["GMX_NO_SOLV_OPT"] = "TRUE"
        os.environ["GMX_NO_ALLVSALL"] = "TRUE"
        abstempdir = os.path.join(self.root,self.tempdir)
        if options['gmxpath'] == None or options['gmxsuffix'] == None:
            warn_press_key('Please set the options gmxpath and gmxsuffix in the input file!')
        if not os.path.exists(os.path.join(options['gmxpath'],"mdrun"+options['gmxsuffix'])):
            warn_press_key('The mdrun executable pointed to by %s doesn\'t exist! (Check gmxpath and gmxsuffix)' % os.path.join(options['gmxpath'],"mdrun"+options['gmxsuffix']))
        # Link the necessary programs into the temporary directory
        os.symlink(os.path.join(options['gmxpath'],"mdrun"+options['gmxsuffix']),os.path.join(abstempdir,"mdrun"))
        os.symlink(os.path.join(options['gmxpath'],"grompp"+options['gmxsuffix']),os.path.join(abstempdir,"grompp"))
        # Link the run files
        os.symlink(os.path.join(self.root,self.simdir,"conf.gro"),os.path.join(abstempdir,"conf.gro"))
        os.symlink(os.path.join(self.root,self.simdir,"settings","grompp.mdp"),os.path.join(abstempdir,"grompp.mdp"))
        os.symlink(os.path.join(self.root,self.simdir,"settings","topol.top"),os.path.join(abstempdir,"topol.top"))

    def unpack_moments(self, moment_dict):
        answer = Np.array([moment_dict[i]*self.weights[i] for i in moment_dict])
        return answer

    def get(self, mvals, AGrad=False, AHess=False):
        # Some of this code is being repeated here and there.
        # Maybe the start of 'get', where the force field is created, can be put into 
        # some kind of common block.
        # Create the new force field!!
        Answer = {'X':0.0, 'G':Np.zeros(self.FF.np, dtype=float), 'H':Np.zeros((self.FF.np, self.FF.np), dtype=float)}
        def get_momvals(mvals_):
            self.FF.make(mvals_)
            moments = get_monomer_properties()
            # Unpack from dictionary.
            return self.unpack_moments(moments)

        np = len(mvals)
        G = Np.zeros(np,dtype=float)
        H = Np.zeros((np,np),dtype=float)
        pvals = self.FF.make(mvals,self.usepvals)

        calc_moments = get_monomer_properties()
        
        ref_momvals = self.unpack_moments(self.ref_moments)
        calc_momvals = self.unpack_moments(calc_moments)

        D = calc_momvals - ref_momvals
        dV = Np.zeros((self.FF.np,len(calc_momvals)),dtype=float)

        if AGrad or AHess:
            for p in range(self.FF.np):
                dV[p,:], _ = f12d3p(fdwrap(get_momvals, mvals, p), h = self.h, f0 = calc_momvals)
                
        Answer['X'] = dot(D,D)
        for p in range(self.FF.np):
            Answer['G'][p] = 2*dot(D, dV[p,:])
            for q in range(self.FF.np):
                Answer['H'][p,q] = 2*dot(dV[p,:], dV[q,:])

        if not in_fd():
            self.FF.make(mvals)
            self.calc_moments = calc_moments
            self.objective = Answer['X']

        return Answer

