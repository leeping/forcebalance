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

# All TINKER force field parameter types, which should eventually go into pdict
# at some point (for full compatibility).
allp = ['atom', 'vdw', 'vdw14', 'vdwpr', 'hbond', 'bond', 'bond5', 'bond4',
        'bond3', 'electneg', 'angle', 'angle5', 'angle4', 'angle3', 'anglef',
        'strbnd', 'ureybrad', 'angang', 'opbend', 'opdist', 'improper', 'imptors',
        'torsion', 'torsion5', 'torsion4', 'pitors', 'strtors', 'tortors', 'charge',
        'dipole', 'dipole5', 'dipole4', 'dipole3', 'multipole', 'polarize', 'piatom',
        'pibond', 'pibond5', 'pibond4', 'metal', 'biotype', 'mmffvdw', 'mmffbond',
        'mmffbonder', 'mmffangle', 'mmffstrbnd', 'mmffopbend', 'mmfftorsion', 'mmffbci',
        'mmffpbci', 'mmffequiv', 'mmffdefstbn', 'mmffcovrad', 'mmffprop', 'mmffarom']

from forcebalance.output import getLogger
logger = getLogger(__name__)

pdict = {'VDW'          : {'Atom':[1], 2:'S',3:'T',4:'D'}, # Van der Waals distance, well depth, distance from bonded neighbor?
         'BOND'         : {'Atom':[1,2], 3:'K',4:'B'},     # Bond force constant and equilibrium distance (Angstrom)
         'ANGLE'        : {'Atom':[1,2,3], 4:'K',5:'B'},   # Angle force constant and equilibrium angle
         'UREYBRAD'     : {'Atom':[1,2,3], 4:'K',5:'B'},   # Urey-Bradley force constant and equilibrium distance (Angstrom)
         'MCHARGE'      : {'Atom':[1,2,3], 4:''},          # Atomic charge
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

def write_key(fout, options, fin=None, defaults={}, verbose=False, prmfnm=None):
    """
    Create or edit a TINKER .key file.
    @param[in] fout Output file name, can be the same as input file name.
    @param[in] options Dictionary containing .key options. Existing options are replaced, new options are added at the end.
    Passing None causes options to be deleted.  To pass an option without an argument, use ''.
    @param[in] fin Input file name.
    @param[in] defaults Default options to add to the mdp only if they don't already exist.
    @param[in] verbose Print out all modifications to the file.
    @param[in] prmfnm TINKER parameter file name.
    """
    # Make sure that the keys are lowercase, and the values are all strings.
    options = OrderedDict([(key.lower(), str(val)) for key, val in options.items()])
    if 'parameters' in options and prmfnm != None:
        raise RuntimeError("Please pass prmfnm or 'parameters':'filename.prm' in options but not both.")
    elif 'parameters' in options:
        prmfnm = options['parameters']
    
    if prmfnm != None:
        # Account for both cases where the file name may end with .prm
        prms = [prmfnm]
        if prms[0].endswith('.prm'):
            prms.append(prmfnm[:-4])
    else:
        prms = []

    # Options that cause the program to crash if they are overwritten
    clashes = []
    # List of lines in the output file.
    out = []
    # List of options in the output file.
    haveopts = []
    skip = 0
    prmflag = 0
    if fin != None and os.path.isfile(fin):
        for line0 in open(fin).readlines():
            line1   = line0.replace('\n','').expandtabs()
            line    = line0.strip().expandtabs()
            s       = line.split()
            if skip:
                out.append(line1)
                skip -= 1
                continue
            # Skip over these three cases:
            # 1) Empty lines get appended to the output and skipped.
            if len(line) == 0 or set(line).issubset([' ']):
                out.append('')
                continue
            # 2) Lines that start with comments are skipped as well.
            if line.startswith("#"):
                out.append(line1)
                continue
            # 3) Lines that correspond to force field parameters are skipped
            if s[0].lower() in allp:
                out.append(line1)
                # 4) For AMOEBA multipole parameters, skip four additional lines
                if s[0].lower() == "multipole":
                    skip = 4
                continue
            # Now split by the comment character.
            s = line.split('#',1)
            data = s[0]
            comms = s[1] if len(s) > 1 else None
            # Now split off the key and value fields at the space.
            keyf, valf = data.split(' ',1)
            key = keyf.strip().lower()
            haveopts.append(key)
            if key == 'parameters':
                val0 = valf.strip()
                # This is the case where "parameters" correctly corresponds to optimize.in
                prmflag = 1
                if prmfnm == None or val0 in prms:
                    out.append(line1)
                    continue
                else:
                    logger.info(line + '\n')
                    warn_press_key("The above line was found in %s, but we expected something like 'parameters %s'; replacing." % (line,prmfnm))
                    options['parameters'] = prmfnm
            if key in options:
                # This line replaces the line in the .key file with the value provided in the dictionary.
                val = options[key]
                val0 = valf.strip()
                if key in clashes and val != val0:
                    raise RuntimeError("write_key tried to set %s = %s but its original value was %s = %s" % (key, val, key, val0))
                # Passing None as the value causes the option to be deleted
                if val == None: continue
                if len(val) < len(valf):
                    valf = ' ' + val + ' '*(len(valf) - len(val)-1)
                else:
                    valf = ' ' + val + ' '
                lout = [keyf, ' ', valf]
                if comms != None:
                    lout += ['#',comms]
                out.append(''.join(lout))
            else:
                out.append(line1)
    # Options that don't already exist are written at the bottom.
    for key, val in options.items():
        key = key.lower()
        if key not in haveopts:
            haveopts.append(key)
            out.append("%-20s %s" % (key, val))
    # Fill in default options.
    for key, val in defaults.items():
        key = key.lower()
        options[key] = val
        if key not in haveopts:
            haveopts.append(key)
            out.append("%-20s %s" % (key, val))
    # If parameters are not specified, they are printed at the top.
    if not prmflag and prmfnm != None:
        out.insert(0,"parameters %s" % prmfnm)
    # Finally write the key file.
    file_out = wopen(fout) 
    for line in out:
        print >> file_out, line
    if verbose:
        printcool_dictionary(options, title="%s -> %s with options:" % (fin, fout))
    file_out.close()

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
        write_key(os.path.join(abstempdir,"liquid.key"), {'archive':None,'save-box':None}, 
                  os.path.join(self.root,self.tgtdir,"liquid.key"), {}, prmfnm=self.FF.tinkerprm)
        # LinkFile(os.path.join(self.root,self.tgtdir,"liquid.key"),os.path.join(abstempdir,"liquid.key"))
        LinkFile(os.path.join(self.root,self.tgtdir,"mono.xyz"),os.path.join(abstempdir,"mono.xyz"))
        write_key(os.path.join(abstempdir,"mono.key"), {'archive':None,'save-box':None}, 
                  os.path.join(self.root,self.tgtdir,"mono.key"), {}, prmfnm=self.FF.tinkerprm)
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
        ## Keyword args that aren't in this list are filtered out.
        self.valkwd = ['tinker_key', 'tinkerpath']
        super(TINKER,self).__init__(name=name, **kwargs)

    def setopts(self, **kwargs):
        
        """ Called by __init__ ; Set TINKER-specific options. """

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

    def readsrc(self, **kwargs):

        """ Called by __init__ ; read files from the source directory. """

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
            if not arcfile: raise RuntimeError('Cannot determine which .arc file to use')
            self.mol = Molecule(arcfile)

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

    def prepare(self, **kwargs):

        """ Called by __init__ ; prepare the temp directory and figure out the topology. """

        self.rigid = False
        ## Write the appropriate coordinate and key files.
        if hasattr(self,'target'):
            # Create the force field in this directory if the force field object is provided.  
            # This is because the .key file could be a force field file!
            FF = self.target.FF
            FF.make(np.zeros(FF.np))
            if FF.rigid_water:
                self.rigid = True
            write_key("%s.key" % self.name, {'digits':'10'}, os.path.join(self.srcdir, self.key), {}, FF.tinkerprm)
        elif self.key:
            LinkFile(os.path.join(self.srcdir, self.key), "%s.key" % self.name, nosrcok=True)
        else:
            raise RuntimeError('Need a .key file to continue')
        self.mol[0].write(os.path.join("%s.xyz" % self.name), ftype="tinker")

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
        gs = nx.connected_component_subgraphs(G)
        tmols = [gs[i] for i in np.argsort(np.array([min(g.nodes()) for g in gs]))]
        self.AtomLists['MoleculeNumber'] = [[i+1 in m.nodes() for m in tmols].index(1) for i in range(self.mol.na)]
        # Delete force field files.
        if hasattr(self,'target'):
            for f in FF.fnms:
                os.unlink(f)

    def optimize(self, shot=0, method="newton", crit=1e-4):

        """ Optimize the geometry and align the optimized geometry to the starting geometry. """

        if os.path.exists('%s.xyz_2' % self.name):
            os.unlink('%s.xyz_2' % self.name)

        self.mol[shot].write('%s.xyz' % self.name, ftype="tinker")

        if method == "newton":
            if self.rigid: optprog = "optrigid"
            else: optprog = "optimize"
        elif method == "bfgs":
            if self.rigid: optprog = "minrigid"
            else: optprog = "minimize"

        o = self.calltinker("%s %s.xyz %f" % (optprog, self.name, crit))
        # Silently align the optimized geometry.
        M12 = Molecule("%s.xyz" % self.name, ftype="tinker") + Molecule("%s.xyz_2" % self.name, ftype="tinker")
        M12.align(center=False)
        M12[1].write("%s.xyz_2" % self.name, ftype="tinker")
        rmsd = M12.ref_rmsd(0)[1]
        cnvgd = 0
        mode = 0
        for line in o:
            s = line.split()
            if len(s) == 0: continue
            if "Optimally Conditioned Variable Metric Optimization" in line: mode = 1
            if "Limited Memory BFGS Quasi-Newton Optimization" in line: mode = 1
            if mode == 1 and isint(s[0]): mode = 2
            if mode == 2:
                if isint(s[0]): E = float(s[1])
                else: mode = 0
            if "Normal Termination" in line:
                cnvgd = 1
        if not cnvgd:
            for line in o:
                logger.info(str(line) + '\n')
            logger.info("The minimization did not converge in the geometry optimization - printout is above.\n")
        return E, rmsd

    def energy_force_one(self, shot):

        """ Computes the energy and force using TINKER for one snapshot. """

        self.mol.write("%s.xyz" % self.name,select=[shot], ftype="tinker")
        # This line actually runs TINKER
        o = self.calltinker("testgrad %s.xyz y n n" % (self.name))
        # Read data from stdout and stderr, and convert it to GROMACS
        # units for consistency with existing code.
        E = []
        F = []
        i = 0
        for line in o:
            s = line.split()
            if "Total Potential Energy" in line:
                E = [float(s[4]) * 4.184]
            elif len(s) == 6 and all([s[0] == 'Anlyt',isint(s[1]),isfloat(s[2]),isfloat(s[3]),isfloat(s[4]),isfloat(s[5])]):
                if self.AtomMask[i]:
                    F += [-1 * float(j) * 4.184 * 10 for j in s[2:5]]
                i += 1
        M = np.array(E + F)
        return M

    def energy(self):

        """ Computes the energy using TINKER over a trajectory. """

        self.mol.write("%s.xyz" % self.name, ftype="tinker")
        # This line actually runs TINKER
        o = self.calltinker("analyze %s.xyz e" % (self.name))
        # Read data from stdout and stderr, and convert it to GROMACS units.
        Borked = 0
        E = []
        for line in o:
            s = line.split()
            if "Total Potential Energy" in line:
                E.append(float(s[4]) * 4.184)
        return np.array(E)

    def energy_force(self):

        """ Computes the energy and force using TINKER over a trajectory. """

        M = []
        warn_once("Using testgrad to loop over %i energy/force calculations; will be slow." % len(self.mol))
        for i in range(len(self.mol)):
            logger.info("\r%i/%i\r" % (i+1, len(self.mol)))
            M.append(self.energy_force_one(i))
        return np.array(M)

    def normal_modes(self, shot=0, optimize=True):
        # This line actually runs TINKER
        if optimize:
            self.optimize(shot, crit=1e-6)
            o = self.calltinker("vibrate %s.xyz_2 a" % (self.name))
        else:
            warn_once("Asking for normal modes without geometry optimization?")
            self.mol[shot].write('%s.xyz' % self.name, ftype="tinker")
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

    def multipole_moments(self, shot=0, optimize=True, polarizability=False):

        """ Return the multipole moments of the 1st snapshot in Debye and Buckingham units. """
        
        # This line actually runs TINKER
        if optimize:
            self.optimize(shot, crit=1e-6)
            o = self.calltinker("analyze %s.xyz_2 M" % (self.name))
        else:
            self.mol[shot].write('%s.xyz' % self.name, ftype="tinker")
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
                if "Total Polarizability Tensor" in line:
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

    def energy_rmsd(self, shot=0, optimize=True):

        """ Calculate energy of the selected structure (optionally minimize and return the minimized energy and RMSD). In kcal/mol. """

        rmsd = 0.0
        # This line actually runs TINKER
        # xyzfnm = sysname+".xyz"
        if optimize:
            E_, rmsd = self.optimize(shot)
            o = self.calltinker("analyze %s.xyz_2 E" % self.name)
            #----
            # Two equivalent ways to get the RMSD, here for reference.
            #----
            # M1 = Molecule("%s.xyz" % self.name, ftype="tinker")
            # M2 = Molecule("%s.xyz_2" % self.name, ftype="tinker")
            # M1 += M2
            # rmsd = M1.ref_rmsd(0)[1]
            #----
            # oo = self.calltinker("superpose %s.xyz %s.xyz_2 1 y u n 0" % (self.name, self.name))
            # for line in oo:
            #     if "Root Mean Square Distance" in line:
            #         rmsd = float(line.split()[-1])
            #----
            os.system("rm %s.xyz_2" % self.name)
        else:
            o = self.calltinker("analyze %s.xyz E" % self.name)
        # Read the TINKER output. 
        E = None
        for line in o:
            if "Total Potential Energy" in line:
                E = float(line.split()[-2].replace('D','e'))
        if E == None:
            raise RuntimeError("Total potential energy wasn't encountered when calling analyze!")
        if optimize and abs(E-E_) > 0.1:
            warn_press_key("Energy from optimize and analyze aren't the same (%.3f vs. %.3f)" % (E, E_))
        return E, rmsd

    def interaction_energy(self, fraga, fragb):
        
        """ Calculate the interaction energy for two fragments. """

        if not hasattr(self,'A'):
            self.A = TINKER(name="A", mol=self.mol.atom_select(fraga), tinker_key="%s.key" % self.name, tinkerpath=self.tinkerpath)
        if not hasattr(self,'B'):
            self.B = TINKER(name="B", mol=self.mol.atom_select(fragb), tinker_key="%s.key" % self.name, tinkerpath=self.tinkerpath)

        # Interaction energy needs to be in kcal/mol.
        return (self.energy() - self.A.energy() - self.B.energy()) / 4.184

    def molecular_dynamics(self, nsteps, timestep, temperature=None, pressure=None, nequil=0, nsave=1000, minimize=True, anisotropic=False, verbose=False, **kwargs):
        
        """
        Method for running a molecular dynamics simulation.  

        Required arguments:
        nsteps      = (int)   Number of total time steps
        timestep    = (float) Time step in FEMTOSECONDS
        temperature = (float) Temperature control (Kelvin)
        pressure    = (float) Pressure control (atmospheres)
        nequil      = (int)   Number of additional time steps at the beginning for equilibration
        nsave       = (int)   Step interval for saving and printing data
        minimize    = (bool)  Perform an energy minimization prior to dynamics

        Returns simulation data:
        Rhos        = (array)     Density in kilogram m^-3
        Potentials  = (array)     Potential energies
        Kinetics    = (array)     Kinetic energies
        Volumes     = (array)     Box volumes
        Dips        = (3xN array) Dipole moments
        EComps      = (dict)      Energy components
        """

        if minimize:
            if verbose: logger.info("Minimizing the energy...")
            self.optimize(crit=1)
            os.system("mv %s.xyz_2 %s.xyz" % (self.name, self.name))
            if verbose: logger.info("Done\n")

        # Run equilibration.
        if nequil > 0:
            if verbose: logger.info("Running equilibration...\n")
            if self.pbc and pressure:
                self.calltinker("dynamic %s %i %f %f 4 %f %f" % (xin, nsteps, timestep, float(nsave*timestep)/1000, 
                                                                 temperature, pressure), print_to_screen=verbose)
            else:
                self.calltinker("dynamic %s %i %f %f 2 %f" % (xin, nstep, timestep, float(nsave*timestep)/1000,
                                                                  temperature), print_to_screen=verbose)
            os.system("rm -f %s.arc %s.box" % (basename, basename))

        # Run production.
        if verbose: logger.info("Running production...\n")
        if self.pbc and pressure:
            odyn = self.calltinker("dynamic %s %i %f %f 4 %f %f" % (xin, nstep*npr, tstep, float(nstep*tstep/1000), 
                                                                    temperature, pressure), print_to_screen=verbose)
        else:
            odyn = self.calltinker("dynamic %s %i %f %f 2 %f" % (xin, nstep*npr, tstep, float(nstep*tstep/1000), 
                                                          temperature), print_to_screen=verbose)

        # Gather information.
        os.system("mv %s.arc %s-md.arc" % (self.name, self.name))
        self.mdtraj = "%s-md.arc" % self.name
        edyn = []
        kdyn = []
        for line in odyn:
            if 'Current Potential' in line:
                edyn.append(float(line.split()[2]))
            if 'Current Kinetic' in line:
                kdyn.append(float(line.split()[2]))

        # Potential and kinetic energies converted to kJ/mol.
        edyn = np.array(edyn) * 4.184
        kdyn = np.array(kdyn) * 4.184
    
        if verbose: logger.info("Post-processing to get the dipole moments\n")
        cmdstr = "analyze %s" % xain
        oanl = self.calltinker("analyze %s.arc" % self.name, stdin="G,E", verbose=False)

        # Read potential energy and dipole from file.
        eanl = []
        dip = []
        mass = 0.0
        for line in oanl:
            if 'Total System Mass' in line:
                mass = float(line.split()[-1])
            if 'Total Potential Energy : ' in line:
                eanl.append(float(line.split()[4]))
            if 'Dipole X,Y,Z-Components :' in line:
                dip.append([float(line.split()[i]) for i in range(-3,0)])

        # Energies in kilojoules per mole
        eanl = np.array(eanl) * 4.184
        # Dipole moments in debye
        dip = np.array(dip)
        # Volume of simulation boxes in cubic nanometers
        # Conversion factor derived from the following:
        # In [22]: 1.0 * gram / mole / (1.0 * nanometer)**3 / AVOGADRO_CONSTANT_NA / (kilogram/meter**3)
        # Out[22]: 1.6605387831627252
        conv = 1.6605387831627252
        if self.pbc:
            box = [[float(i) for i in line.split()[1:4]] for line in open(xyz[:-3]+"box").readlines()]
            vol = np.array([i[0]*i[1]*i[2] for i in box]) / 1000
            rho = conv * mass / vol
        else:
            vol = None
            rho = None

        ecomp = None

        return rho, edyn, kdyn, vol, dip, ecomp

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
        del engine_args['name']
        ## Create engine object.
        self.engine = TINKER(target=self, **engine_args)
        self.AtomLists = self.engine.AtomLists
        self.AtomMask = self.engine.AtomMask
        ## all_at_once is not implemented.
        if self.force and self.all_at_once:
            warn_press_key("Force matching is turned on but TINKER can only do trajectory loops for energy-only jobs.")
            self.all_at_once = False

    def read_topology(self):
        self.topology_flag = True

    def energy_force(self):
        return self.engine.energy_force()

class Vibration_TINKER(Vibration):

    """Subclass of Target for vibrational frequency matching
    using TINKER.  Provides optimized geometry, vibrational frequencies (in cm-1),
    and eigenvectors."""

    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'coords',default="input.xyz")
        self.set_option(tgt_opts,'tinker_key',default="input.key")
        ## Initialize base class.
        super(Vibration_TINKER,self).__init__(options,tgt_opts,forcefield)
        ## Build keyword dictionaries to pass to engine.
        engine_args = deepcopy(self.__dict__)
        engine_args.update(options)
        del engine_args['name']
        ## Create engine object.
        self.engine = TINKER(target=self, **engine_args)

class Moments_TINKER(Moments):

    """Subclass of Target for multipole moment matching
    using TINKER."""

    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'coords',default="input.xyz")
        self.set_option(tgt_opts,'tinker_key',default="input.key")
        ## Initialize base class.
        super(Moments_TINKER,self).__init__(options,tgt_opts,forcefield)
        ## Build keyword dictionaries to pass to engine.
        engine_args = deepcopy(self.__dict__)
        engine_args.update(options)
        del engine_args['name']
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
        del engine_args['name']
        ## Create engine objects.
        self.engines = OrderedDict()
        for sysname,sysopt in self.sys_opts.items():
            M = Molecule(os.path.join(self.root, self.tgtdir, sysopt['geometry']),ftype="tinker")
            if 'select' in sysopt:
                atomselect = np.array(uncommadash(sysopt['select']))
                M = M.atom_select(atomselect)
            if self.FF.rigid_water: M.rigid_water()
            self.engines[sysname] = TINKER(target=self, mol=M, name=sysname, tinker_key=os.path.join(sysopt['keyfile']), tinkerpath=options['tinkerpath'])

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
        del engine_args['name']
        ## Create engine object.
        self.engine = TINKER(target=self, **engine_args)

    def interaction_driver_all(self,dielectric=False):
        # Compute the energies for the dimer
        return self.engine.interaction_energy(self.select1, self.select2)
