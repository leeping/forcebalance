""" @package forcebalance.tinkerio TINKER input/output.

This serves as a good template for writing future force matching I/O
modules for other programs because it's so simple.

@author Lee-Ping Wang
@date 01/2012
"""

import os, shutil
from re import match, sub
from forcebalance.nifty import *
from forcebalance.nifty import _exec
import numpy as np
import networkx as nx
from copy import deepcopy
from forcebalance import BaseReader
from subprocess import Popen, PIPE
from forcebalance.engine import Engine
from forcebalance.abinitio import AbInitio
from forcebalance.vibration import Vibration
from forcebalance.moments import Moments
from forcebalance.liquid import Liquid
from forcebalance.molecule import Molecule
from forcebalance.binding import BindingEnergy
from forcebalance.interaction import Interaction
from forcebalance.finite_difference import in_fd
from collections import OrderedDict
from forcebalance.optimizer import GoodStep
from forcebalance.unit import *

from forcebalance.output import getLogger
logger = getLogger(__name__)

pdict = {'VDW'          : {'Atom':[1], 2:'S',3:'T',4:'D'}, # Van der Waals distance, well depth, distance from bonded neighbor?
         'BOND'         : {'Atom':[1,2], 3:'K',4:'B'},     # Bond force constant and equilibrium distance (Angstrom)
         'ANGLE'        : {'Atom':[1,2,3], 4:'K',5:'B'},   # Angle force constant and equilibrium angle
         'UREYBRAD'     : {'Atom':[1,2,3], 4:'K',5:'B'},   # Urey-Bradley force constant and equilibrium distance (Angstrom)
         'MCHARGE'       : {'Atom':[1,2,3], 4:''},          # Atomic charge
         'DIPOLE'       : {0:'X',1:'Y',2:'Z'},             # Dipole moment in local frame
         'QUADX'        : {0:'X'},                         # Quadrupole moment, X component
         'QUADY'        : {0:'X',1:'Y'},                   # Quadrupole moment, Y component
         'QUADZ'        : {0:'X',1:'Y',2:'Z'},             # Quadrupole moment, Y component
         'POLARIZE'     : {'Atom':[1], 2:'A',3:'T'},       # Atomic dipole polarizability
         'BOND-CUBIC'   : {'Atom':[], 0:''},    # Below are global parameters.
         'BOND-QUARTIC' : {'Atom':[], 0:''},
         'ANGLE-CUBIC'  : {'Atom':[], 0:''},
         'ANGLE-QUARTIC': {'Atom':[], 0:''},
         'ANGLE-PENTIC' : {'Atom':[], 0:''},
         'ANGLE-SEXTIC' : {'Atom':[], 0:''},
         'DIELECTRIC'   : {'Atom':[], 0:''},
         'POLAR-SOR'    : {'Atom':[], 0:''}
                                                # Ignored for now: stretch/bend coupling, out-of-plane bending,
                                                # torsional parameters, pi-torsion, torsion-torsion
         }



class Tinker_Reader(BaseReader):
    """Finite state machine for parsing TINKER force field files.

    This class is instantiated when we begin to read in a file.  The
    feed(line) method updates the state of the machine, informing it
    of the current interaction type.  Using this information we can
    look up the interaction type and parameter type for building the
    parameter ID.
    
    """
    
    def __init__(self,fnm):
        super(Tinker_Reader,self).__init__(fnm)
        ## The parameter dictionary (defined in this file)
        self.pdict  = pdict
        ## The atom numbers in the interaction (stored in the TINKER parser)
        self.atom   = []

    def feed(self, line):
        """ Given a line, determine the interaction type and the atoms involved (the suffix).

        TINKER generally has stuff like this:

        @verbatim
        bond-cubic              -2.55
        bond-quartic            3.793125

        vdw           1               3.4050     0.1100
        vdw           2               2.6550     0.0135      0.910 # PRM 4

        multipole     2    1    2               0.25983
                                               -0.03859    0.00000   -0.05818
                                               -0.03673
                                                0.00000   -0.10739
                                               -0.00203    0.00000    0.14412
        @endverbatim

        The '#PRM 4' has no effect on TINKER but it indicates that we
        are tuning the fourth field on the line (the 0.910 value).

        @todo Put the rescaling factors for TINKER parameters in here.
        Currently we're using the initial value to determine the
        rescaling factor which is not very good.

        Every parameter line is prefaced by the interaction type
        except for 'multipole' which is on multiple lines.  Because
        the lines that come after 'multipole' are predictable, we just
        determine the current line using the previous line.

        Random note: Unit of force is kcal / mole / angstrom squared.
        
        """
        s          = line.split()
        self.ln   += 1
        # No sense in doing anything for an empty line or a comment line.
        if len(s) == 0 or match('^#',line): return None, None
        # From the line, figure out the interaction type.  If this line doesn't correspond
        # to an interaction, then do nothing.
        if s[0].upper() in pdict:
            self.itype = s[0].upper()
        # This is kind of a silly hack that allows us to take care of the 'multipole' keyword,
        # because of the syntax that the TINKER .prm file uses.
        elif s[0].upper() == 'MULTIPOLE':
            self.itype = 'MCHARGE'
        elif self.itype == 'MCHARGE':
            self.itype = 'DIPOLE'
        elif self.itype == 'DIPOLE':
            self.itype = 'QUADX'
        elif self.itype == 'QUADX':
            self.itype = 'QUADY'
        elif self.itype == 'QUADY':
            self.itype = 'QUADZ'
        else:
            self.itype = None

        if self.itype in pdict:
            if 'Atom' in pdict[self.itype]:
                # List the atoms in the interaction.
                self.atom = [s[i] for i in pdict[self.itype]['Atom']]
            # The suffix of the parameter ID is built from the atom    #
            # types/classes involved in the interaction.
            self.suffix = '.'.join(self.atom)

def write_key_with_prm(src, dest, prmfnm=None, ffobj=None):
    """ Copies a TINKER .key file but changes the parameter keyword as
    necessary to reflect the ForceBalance settings. """

    if src == dest:
        raise Exception("This function shouldn't be used to modify a file in-place.")

    if prmfnm == None and ffobj == None:
        raise Exception('write_key_with_prm should be called with either a ForceField object or a parameter file name')
    elif prmfnm == None:
        if hasattr(ffobj, 'tinkerprm'):
            prmfnm = ffobj.tinkerprm
        else:
            raise AttributeError('The TINKER parameter file name must be specified in the ForceBalance input file')
    elif prmfnm != None and ffobj != None:
        raise Exception('write_key_with_prm should be called with either a ForceField object or a parameter file name, but not both')
        
    # Account for both cases where the file name may end with .prm
    prms = [prmfnm]
    if prms[0].endswith('.prm'):
        prms.append(prmfnm[:-4])

    # This is a flag which tells us whether the "parameters" line has appeared
    prmflag = False
    outlines = []
    for line in open(src):
        if len(line.split()) > 1 and line.split()[0].lower() == 'parameters':
            prmflag = True
            # This is the case where "parameters" correctly corresponds to optimize.in
            if line.split()[1] in prms: pass
            else:
                logger.info(line + '\n')
                warn_press_key("The above line was found in %s, but we expected something like %s" % (src,prmfnm))
        outlines.append(line)
    if not prmflag:
        logger.info("Adding parameter file %s to key file\n" % prmfnm)
        outlines.insert(0,"parameters %s\n" % prmfnm)
    with open(dest,'w') as f: f.writelines(outlines)

def modify_key(src, in_dict):
    """ Performs in-place modification of a TINKER .key file. 

    The input dictionary contains key:value pairs such as
    "polarization direct".  If the key exists in the TINKER file, then
    that line is modified such that it contains the value in the
    dictionary.  Note that this "key" is not to be confused with the
    .key extension in the TINKER file that we're modifying.

    Sometimes keys like 'archive' do not have a value, in which case
    the dictionary should contain a None value or a blank space.

    If the key doesn't exist in the TINKER file, then the key:value pair
    will be printed at the end.

    @param[in] src Name of the TINKER file to be modified.
    @param[in] in_dict Dictionary containing key-value pairs used to modify the TINKER file.

    """

    if os.path.isfile(src) and not os.path.islink(src):
        fin = open(src).readlines()
    else:
        raise Exception("This function shouldn't be used to follow symbolic links, because I don't want to modify files in the target directory")
    odict = OrderedDict([(key.lower(), val) for key, val in in_dict.items()])
    flags = OrderedDict([(key, False) for key in odict.keys()])
    outlines = []
    for line in open(src).readlines():
        s = line.split()
        if len(s) == 0:
            outlines.append(line)
            continue
        key = s[0].lower()
        # Modify the line in-place if the key already exists.
        if key in odict:
            val = odict[key]
            if val != None:
                outlines.append("%s %s\n" % (key, val))
            else:
                outlines.append("%s\n" % (key))
            flags[key] = True
        else:
            outlines.append(line)
    for key, val in odict.items():
        if not flags[key]:
            if val != None:
                outlines.append("%s %s\n" % (key, val))
            else:
                outlines.append("%s\n" % (key))
    with open(src,'w') as f: f.writelines(outlines)

class Liquid_TINKER(Liquid):
    def __init__(self,options,tgt_opts,forcefield):
        super(Liquid_TINKER,self).__init__(options,tgt_opts,forcefield)
        self.DynDict = OrderedDict()
        self.DynDict_New = OrderedDict()
        if self.do_self_pol:
            warn_press_key("Self-polarization correction not implemented yet when using TINKER")

    def prepare_temp_directory(self,options,tgt_opts):
        """ Prepare the temporary directory by copying in important files. """
        abstempdir = os.path.join(self.root,self.tempdir)
        LinkFile(os.path.join(options['tinkerpath'],"dynamic"),os.path.join(abstempdir,"dynamic"))
        LinkFile(os.path.join(options['tinkerpath'],"analyze"),os.path.join(abstempdir,"analyze"))
        LinkFile(os.path.join(options['tinkerpath'],"minimize"),os.path.join(abstempdir,"minimize"))
        LinkFile(os.path.join(self.root,self.tgtdir,"liquid.xyz"),os.path.join(abstempdir,"liquid.xyz"))
        write_key_with_prm(os.path.join(self.root,self.tgtdir,"liquid.key"),os.path.join(abstempdir,"liquid.key"),ffobj=self.FF)
        modify_key(os.path.join(abstempdir,"liquid.key"),{'archive':None,'save-box':None})
        # LinkFile(os.path.join(self.root,self.tgtdir,"liquid.key"),os.path.join(abstempdir,"liquid.key"))
        LinkFile(os.path.join(self.root,self.tgtdir,"mono.xyz"),os.path.join(abstempdir,"mono.xyz"))
        write_key_with_prm(os.path.join(self.root,self.tgtdir,"mono.key"),os.path.join(abstempdir,"mono.key"),ffobj=self.FF)
        modify_key(os.path.join(abstempdir,"mono.key"),{'archive':None})
        # LinkFile(os.path.join(self.root,self.tgtdir,"mono.key"),os.path.join(abstempdir,"mono.key"))
        LinkFile(os.path.join(os.path.split(__file__)[0],"data","npt_tinker.py"),os.path.join(abstempdir,"npt_tinker.py"))
        # LinkFile(os.path.join(self.root,self.tgtdir,"npt_tinker.py"),os.path.join(abstempdir,"npt_tinker.py"))

    def npt_simulation(self, temperature, pressure, simnum):
        """ Submit a NPT simulation to the Work Queue. """
        wq = getWorkQueue()
        if not (os.path.exists('npt_result.p') or os.path.exists('npt_result.p.bz2')):
            link_dir_contents(os.path.join(self.root,self.rundir),os.getcwd())
            if wq == None:
                logger.info("Running condensed phase simulation locally.\n")
                logger.info("You may tail -f %s/npt_tinker.out in another terminal window\n" % os.getcwd())
                if GoodStep() and (temperature, pressure) in self.DynDict_New:
                    self.DynDict[(temperature, pressure)] = self.DynDict_New[(temperature, pressure)]
                if (temperature, pressure) in self.DynDict:
                    dynsrc = self.DynDict[(temperature, pressure)]
                    dyndest = os.path.join(os.getcwd(), 'liquid.dyn')
                    logger.info("Copying .dyn file: %s to %s\n" % (dynsrc, dyndest))
                    shutil.copy2(dynsrc,dyndest)
                cmdstr = 'python npt_tinker.py liquid.xyz %i %.3f %.3f %.3f %.3f %s --liquid_equ_steps %i &> npt_tinker.out' % \
                    (self.liquid_prod_steps, self.liquid_timestep, self.liquid_interval, temperature, pressure, self.liquid_equ_steps,
                     " --minimize_energy" if self.minimize_energy else "", 
                     )
                _exec(cmdstr)
                self.DynDict_New[(temperature, pressure)] = os.path.join(os.getcwd(),'liquid.dyn')
            else:
                # This part of the code has never been used before
                # Still need to figure out where to specify TINKER location on each cluster
                queue_up(wq,
                         command = 'python npt_tinker.py liquid.xyz %.3f %.3f &> npt_tinker.out' % (temperature, pressure),
                         input_files = ['liquid.xyz','liquid.key','mono.xyz','mono.key','forcebalance.p','npt_tinker.py'],
                         output_files = ['npt_result.p.bz2', 'npt_tinker.py'] + self.FF.fnms,
                         tgt=self)

class TINKER(Engine):
    """ Derived from Engine object for carrying out general purpose TINKER calculations. """
    def __init__(self, name="tinker", **kwargs):
        kwargs = {i:j for i,j in kwargs.items() if j != None} 
        super(TINKER,self).__init__(name=name, **kwargs)

        ## The directory containing TINKER executables (e.g. dynamic)
        if 'tinkerpath' in kwargs:
            self.tinkerpath = kwargs['tinkerpath']
            if not os.path.exists(os.path.join(self.tinkerpath,"dynamic")):
                warn_press_key("The 'dynamic' executable indicated by %s doesn't exist! (Check tinkerpath)" \
                                   % os.path.join(self.tinkerpath,"dynamic"))
        else:
            warn_once("The 'tinkerpath' option was not specified; using default.")
            if which('mdrun') == '':
                warn_press_key("Please add TINKER executables to the PATH or specify tinkerpath.")
            self.tinkerpath = which('dynamic')
        cwd = os.getcwd()
        os.chdir(self.srcdir)
        
        ## Autodetect TINKER .key file and determine coordinates.
        self.key = onefile('key', kwargs['tinker_key'] if 'tinker_key' in kwargs else None)
        if 'mol' in kwargs:
            self.mol = kwargs['mol']
        elif 'coords' in kwargs:
            if kwargs['coords'].endswith('.xyz'):
                self.mol = Molecule(kwargs['coords'], ftype="tinker")
            else:
                self.mol = Molecule(kwargs['coords'])
        else:
            arcfile = onefile('arc')
            self.mol = Molecule(arcfile)
        os.chdir(cwd)
        self.postinit()

    def calltinker(self, command, stdin=None, print_to_screen=False, print_command=False, **kwargs):

        """ Call TINKER; prepend the tinkerpath to calling the TINKER program. """

        csplit = command.split()
        prog = os.path.join(self.tinkerpath, csplit[0])
        csplit[0] = prog
        o = _exec(' '.join(csplit), stdin=stdin, print_to_screen=print_to_screen, print_command=print_command, **kwargs)
        for line in o[-10:]:
            # Catch exceptions since TINKER does not have exit status.
            if "TINKER is Unable to Continue" in line:
                for l in o:
                    logger.info("%s\n" % l)
                warn_press_key("TINKER may have crashed! (See above output)")
                break
        return o

    def prepare(self):

        """ Prepare the calculation.  Write coordinates to the temp-directory.  Read the topology. """

        ## First move into the temp directory if specified by the input arguments.
        cwd = os.getcwd()
        if hasattr(self,'target'):
            dnm = os.path.join(self.root, self.target.tempdir)
        else:
            warn_once("Running in current directory (%s)." % os.getcwd())
            dnm = os.getcwd()
        os.chdir(dnm)

        self.rigid = False
        
        ## Write the appropriate coordinate and key files.
        if hasattr(self,'target'):
            # Create the force field in this directory if the force field object is provided.  
            # This is because the .key file could be a force field file! :)
            FF = self.target.FF
            FF.make(np.zeros(FF.np, dtype=float))
            if FF.rigid_water:
                self.rigid = True
            # if hasattr(self.target,'shots'):
            #     self.mol.write(os.path.join(dnm, "%s-all.arc" % self.name), select=range(self.target.shots))
            # else:
            #     self.mol.write(os.path.join(dnm, "%s-all.arc" % self.name))
            write_key_with_prm(os.path.join(self.srcdir, self.key), os.path.join(dnm, "%s.key" % self.name), ffobj=FF)
        else:
            # self.mol.write(os.path.join(dnm, "%s-all.arc" % self.name))
            LinkFile(os.path.join(self.srcdir, self.key), os.path.join(dnm, "%s.key" % self.name), nosrcok=True)
        self.mol[0].write(os.path.join(dnm, "%s.xyz" % self.name), ftype="tinker")

        ## If the coordinates do not come with TINKER suffixes then throw an error.
        self.mol.require('tinkersuf')

        ## Call analyze to read information needed to build the atom lists.
        o = self.calltinker("analyze %s.xyz P,C" % (self.name))

        ## Parse the output of analyze.
        mode = 0
        self.AtomMask = []
        self.AtomLists = defaultdict(list)
        ptype_dict = {'atom': 'A', 'vsite': 'D'}
        G = nx.Graph()
        for line in o:
            s = line.split()
            if len(s) == 0: continue
            if "Atom Type Definition Parameters" in line:
                mode = 1
            if mode == 1:
                if isint(s[0]): mode = 2
            if mode == 2:
                if isint(s[0]):
                    mass = float(s[5])
                    self.AtomLists['Mass'].append(mass)
                    if mass < 1.0:
                        # Particles with mass less than one count as virtual sites.
                        self.AtomLists['ParticleType'].append('D')
                    else:
                        self.AtomLists['ParticleType'].append('A')
                    self.AtomMask.append(mass >= 1.0)
                else:
                    mode = 0
            if "List of 1-2 Connected Atomic Interactions" in line:
                mode = 3
            if mode == 3:
                if isint(s[0]): mode = 4
            if mode == 4:
                if isint(s[0]):
                    a = int(s[0])
                    b = int(s[1])
                    G.add_node(a)
                    G.add_node(b)
                    G.add_edge(a, b)
                else: mode = 0
        # Use networkx to figure out a list of molecule numbers.
        gs = np.array(nx.connected_component_subgraphs(G))
        tmols = gs[np.argsort(np.array([min(g.nodes()) for g in gs]))]
        self.AtomLists['MoleculeNumber'] = [[i+1 in m.nodes() for m in tmols].index(1) for i in range(self.mol.na)]
        os.chdir(cwd)
        if hasattr(self,'target'):
            self.target.AtomLists = self.AtomLists
            self.target.AtomMask = self.AtomMask

    def optimize(self, crit=1e-4):
        if os.path.exists('%s.xyz_2' % self.name):
            raise RuntimeError("Presence of %s.xyz_2 will ruin the workflow!" % self.name)
        if self.rigid:
            o = self.calltinker("optrigid %s.xyz %f" % (self.name, crit))
        else:
            o = self.calltinker("optimize %s.xyz %f" % (self.name, crit))
        cnvgd = 0
        for line in o:
            if "Normal Termination" in line:
                cnvgd = 1
        if not cnvgd:
            logger.info(str(o) + '\n')
            logger.info("The system %s did not converge in the geometry optimization - printout is above.\n" % sysname)
        return o

    def energy_force_one(self, shot):

        """ Computes the energy and force using TINKER for one snapshot. """

        self.mol.write("%s.arc" % self.name,select=[shot])
        # This line actually runs TINKER
        o = self.calltinker("testgrad %s.arc y n n" % (self.name))
        # Read data from stdout and stderr, and convert it to GROMACS
        # units for consistency with existing code.
        E = []
        F = []
        for line in o:
            s = line.split()
            if "Total Potential Energy" in line:
                E = [float(s[4]) * 4.184]
            elif len(s) == 6 and all([s[0] == 'Anlyt',isint(s[1]),isfloat(s[2]),isfloat(s[3]),isfloat(s[4]),isfloat(s[5])]):
                F += [-1 * float(i) * 4.184 * 10 for i in s[2:5]]
        M = np.array(E + F)
        return M

    def energy(self):

        """ Computes the energy using TINKER over a trajectory. """

        self.mol.write("%s.arc" % self.name)
        # This line actually runs TINKER
        o = self.calltinker("analyze %s.arc e" % (self.name))
        # Read data from stdout and stderr, and convert it to GROMACS units.
        E = []
        for line in o:
            s = line.split()
            if "Total Potential Energy" in line:
                E.append(float(s[4]) * 4.184)
        return np.array(E)

    def energy_force(self, force=True):

        """ Computes the energy and force using TINKER over a trajectory. """

        if hasattr(self,'target') and hasattr(self.target,'force'):
            force = self.target.force
        if not force:
            return self.energy().reshape(-1,1)
        M = []
        warn_once("Using testgrad to loop over %i energy/force calculations; will be slow." % len(self.mol))
        for i in range(len(self.mol)):
            logger.info("\r%i/%i\r" % (i+1, len(self.mol)))
            M.append(self.energy_force_one(i))
        return np.array(M)

    def normal_modes(self, optimize=True):
        # This line actually runs TINKER
        if optimize:
            self.optimize(crit=1e-6)
            o = self.calltinker("vibrate %s.xyz_2 a" % (self.name))
        else:
            warn_press_key("Asking for normal modes without geometry optimization?")
            o = self.calltinker("vibrate %s.xyz a" % (self.name))
        # Read the TINKER output.  The vibrational frequencies are ordered.
        # The six modes with frequencies closest to zero are ignored
        readev = False
        calc_eigvals = []
        calc_eigvecs = []
        for line in o:
            s = line.split()
            if "Vibrational Normal Mode" in line:
                freq = float(s[-2])
                readev = False
                calc_eigvals.append(freq)
                calc_eigvecs.append([])
            elif "Atom" in line and "Delta X" in line:
                readev = True
            elif readev and len(s) == 4 and all([isint(s[0]), isfloat(s[1]), isfloat(s[2]), isfloat(s[3])]):
                calc_eigvecs[-1].append([float(i) for i in s[1:]])
        calc_eigvals = np.array(calc_eigvals)
        calc_eigvecs = np.array(calc_eigvecs)
        # Sort by frequency absolute value and discard the six that are closest to zero
        calc_eigvecs = calc_eigvecs[np.argsort(np.abs(calc_eigvals))][6:]
        calc_eigvals = calc_eigvals[np.argsort(np.abs(calc_eigvals))][6:]
        # Sort again by frequency
        calc_eigvecs = calc_eigvecs[np.argsort(calc_eigvals)]
        calc_eigvals = calc_eigvals[np.argsort(calc_eigvals)]
        os.system("rm -rf *.xyz_* *.[0-9][0-9][0-9]")
        return calc_eigvals, calc_eigvecs

    def multipole_moments(self, optimize=True, polarizability=True):
        # This line actually runs TINKER
        if optimize:
            self.optimize(crit=1e-6)
            o = self.calltinker("analyze %s.xyz_2 M" % (self.name))
        else:
            o = self.calltinker("analyze %s.xyz M" % (self.name))
        # Read the TINKER output.
        qn = -1
        ln = 0
        for line in o:
            s = line.split()
            if "Dipole X,Y,Z-Components" in line:
                dipole_dict = OrderedDict(zip(['x','y','z'], [float(i) for i in s[-3:]]))
            elif "Quadrupole Moment Tensor" in line:
                qn = ln
                quadrupole_dict = OrderedDict([('xx',float(s[-3]))])
            elif qn > 0 and ln == qn + 1:
                quadrupole_dict['xy'] = float(s[-3])
                quadrupole_dict['yy'] = float(s[-2])
            elif qn > 0 and ln == qn + 2:
                quadrupole_dict['xz'] = float(s[-3])
                quadrupole_dict['yz'] = float(s[-2])
                quadrupole_dict['zz'] = float(s[-1])
            ln += 1

        calc_moments = OrderedDict([('dipole', dipole_dict), ('quadrupole', quadrupole_dict)])

        if polarizability:
            if optimize:
                o = self.calltinker("polarize %s.xyz_2" % (self.name))
            else:
                o = self.calltinker("polarize %s.xyz" % (self.name))
            # Read the TINKER output.
            pn = -1
            ln = 0
            polarizability_dict = OrderedDict()
            for line in o:
                s = line.split()
                if "Molecular Polarizability Tensor" in line:
                    pn = ln
                elif pn > 0 and ln == pn + 2:
                    polarizability_dict['xx'] = float(s[-3])
                    polarizability_dict['yx'] = float(s[-2])
                    polarizability_dict['zx'] = float(s[-1])
                elif pn > 0 and ln == pn + 3:
                    polarizability_dict['xy'] = float(s[-3])
                    polarizability_dict['yy'] = float(s[-2])
                    polarizability_dict['zy'] = float(s[-1])
                elif pn > 0 and ln == pn + 4:
                    polarizability_dict['xz'] = float(s[-3])
                    polarizability_dict['yz'] = float(s[-2])
                    polarizability_dict['zz'] = float(s[-1])
                ln += 1
            calc_moments['polarizability'] = polarizability_dict
        os.system("rm -rf *.xyz_* *.[0-9][0-9][0-9]")
        return calc_moments

    def energy_rmsd(self, optimize=True):

        """ Optionally minimize the energy, calculate energy and RMSD from starting structure. """

        rmsd = 0.0
        # This line actually runs TINKER
        # xyzfnm = sysname+".xyz"
        if optimize:
            self.optimize()
            o = self.calltinker("analyze %s.xyz_2 E" % self.name)
            oo = self.calltinker("superpose %s.xyz %s.xyz_2 1 y u n 0" % (self.name, self.name))
            for line in oo:
                if "Root Mean Square Distance" in line:
                    rmsd = float(line.split()[-1])
            os.system("rm %s.xyz_2" % self.name)
        else:
            o = self.calltinker("analyze %s.xyz E" % self.name)
        # Read the TINKER output. 
        for line in o:
            if "Total Potential Energy" in line:
                return float(line.split()[-2].replace('D','e')) * kilocalories_per_mole, rmsd * angstrom
        warn_press_key("Total potential energy wasn't encountered for system %s!" % sysname)

    def interaction_energy(self, fraga, fragb):
        
        """ Calculate the interaction energy for two fragments. """

        self.A = TINKER(name="A", mol=self.mol.atom_select(fraga), tinker_key="%s.key" % self.name)
        self.B = TINKER(name="B", mol=self.mol.atom_select(fragb), tinker_key="%s.key" % self.name)
        # Interaction energy needs to be in kcal/mol.
        return (self.energy() - self.A.energy() - self.B.energy()) / 4.184

class AbInitio_TINKER(AbInitio):

    """Subclass of Target for force and energy matching using TINKER. """

    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'coords',default="all.arc")
        self.set_option(tgt_opts,'tinker_key',default="shot.key")
        ## Initialize base class.
        super(AbInitio_TINKER,self).__init__(options,tgt_opts,forcefield)
        ## Build keyword dictionaries to pass to engine.
        engine_args = deepcopy(self.__dict__)
        engine_args.update(options)
        ## Create engine object.
        self.engine = TINKER(target=self, **engine_args)
        ## all_at_once is not implemented.
        if self.force and self.all_at_once:
            warn_press_key("Force matching is turned on but TINKER can only do trajectory loops for energy-only jobs.")
            self.all_at_once = False

    def read_topology(self):
        self.topology_flag = True

    def energy_force_driver(self, shot):
        return self.engine.energy_force_one(shot)

    def energy_force_driver_all(self):
        if self.force:
            raise Exception('Trying to call unimplemented functionality.')
        return self.engine.energy_force()

class Vibration_TINKER(Vibration):

    """Subclass of Target for vibrational frequency matching
    using TINKER.  Provides optimized geometry, vibrational frequencies (in cm-1),
    and eigenvectors."""

    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'coords','coords',default="input.xyz")
        self.set_option(tgt_opts,'tinker_key',default="input.key")
        ## Initialize base class.
        super(Vibration_TINKER,self).__init__(options,tgt_opts,forcefield)
        ## Build keyword dictionaries to pass to engine.
        engine_args = deepcopy(self.__dict__)
        engine_args.update(options)
        ## Create engine object.
        self.engine = TINKER(target=self, **engine_args)
        if self.FF.rigid_water:
            raise Exception('This class cannot be used with rigid water molecules.')

    def vibration_driver(self):
        return self.engine.normal_modes()

class Moments_TINKER(Moments):

    """Subclass of Target for multipole moment matching
    using TINKER."""

    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'coords','coords',default="input.xyz")
        self.set_option(tgt_opts,'tinker_key',default="input.key")
        ## Initialize base class.
        super(Moments_TINKER,self).__init__(options,tgt_opts,forcefield)
        ## Build keyword dictionaries to pass to engine.
        engine_args = deepcopy(self.__dict__)
        engine_args.update(options)
        ## Create engine object.
        self.engine = TINKER(target=self, **engine_args)

    def moments_driver(self):
        return self.engine.multipole_moments(polarizability='polarizability' in self.ref_moments)

class BindingEnergy_TINKER(BindingEnergy):

    """Subclass of BindingEnergy for binding energy matching
    using TINKER.  """

    def __init__(self,options,tgt_opts,forcefield):
        ## Initialize base class.
        super(BindingEnergy_TINKER,self).__init__(options,tgt_opts,forcefield)
        ## Build keyword dictionaries to pass to engine.
        engine_args = deepcopy(self.__dict__)
        engine_args.update(options)
        ## Create engine objects.
        self.engines = OrderedDict()
        for sysname,sysopt in self.sys_opts.items():
            M = Molecule(os.path.join(self.root, self.tgtdir, sysopt['geometry']),ftype="tinker")
            if 'select' in sysopt:
                atomselect = np.array(uncommadash(sysopt['select']))
                M = M.atom_select(atomselect)
            if self.FF.rigid_water: M.rigid_water()
            self.engines[sysname] = TINKER(target=self, mol=M, name=sysname, tinker_key=os.path.join(sysopt['keyfile']))

    def system_driver(self, sysname):
        opts = self.sys_opts[sysname]
        return self.engines[sysname].energy_rmsd(optimize = (opts['optimize'] if 'optimize' in opts else False))
    
class Interaction_TINKER(Interaction):

    """Subclass of Target for interaction matching using TINKER. """

    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'coords',default="all.arc")
        self.set_option(tgt_opts,'tinker_key',default="shot.key")
        ## Initialize base class.
        super(Interaction_TINKER,self).__init__(options,tgt_opts,forcefield)
        ## Build keyword dictionaries to pass to engine.
        engine_args = deepcopy(self.__dict__)
        engine_args.update(options)
        ## Create engine object.
        self.engine = TINKER(target=self, **engine_args)

    def interaction_driver_all(self,dielectric=False):
        # Compute the energies for the dimer
        return self.engine.interaction_energy(self.select1, self.select2)
