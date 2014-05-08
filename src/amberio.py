""" @package forcebalance.amberio AMBER force field input/output.

This serves as a good template for writing future force matching I/O
modules for other programs because it's so simple.

@author Lee-Ping Wang
@date 01/2012
"""

import os
from re import match, sub, split, findall
from forcebalance.nifty import isint, isfloat, _exec, LinkFile
import numpy as np
from forcebalance import BaseReader
from forcebalance.abinitio import AbInitio

from forcebalance.output import getLogger
logger = getLogger(__name__)

mol2_pdict = {'COUL':{'Atom':[1], 8:''}}

frcmod_pdict = {'BONDS': {'Atom':[0], 1:'K', 2:'B'},
                'ANGLES':{'Atom':[0], 1:'K', 2:'B'},
                'PDIHS1':{'Atom':[0], 2:'K', 3:'B'},
                'PDIHS2':{'Atom':[0], 2:'K', 3:'B'},
                'PDIHS3':{'Atom':[0], 2:'K', 3:'B'},
                'PDIHS4':{'Atom':[0], 2:'K', 3:'B'},
                'PDIHS5':{'Atom':[0], 2:'K', 3:'B'},
                'PDIHS6':{'Atom':[0], 2:'K', 3:'B'},
                'IDIHS' :{'Atom':[0], 1:'K', 3:'B'},
                'VDW':{'Atom':[0], 1:'S', 2:'T'}
                }

def is_mol2_atom(line):
    s = line.split()
    if len(s) < 9:
        return False
    return all([isint(s[0]), isfloat(s[2]), isfloat(s[3]), isfloat(s[4]), isfloat(s[8])])

class Mol2_Reader(BaseReader):
    """Finite state machine for parsing Mol2 force field file. (just for parameterizing the charges)"""
    
    def __init__(self,fnm):
        # Initialize the superclass. :)
        super(Mol2_Reader,self).__init__(fnm)
        ## The parameter dictionary (defined in this file)
        self.pdict  = mol2_pdict
        ## The atom numbers in the interaction (stored in the parser)
        self.atom   = []
        ## The mol2 file provides a list of atom names
        self.atomnames = []
        ## The section that we're in
        self.section = None
        # The name of the molecule
        self.mol = None

    def feed(self, line):
        s          = line.split()
        self.ln   += 1
        # In mol2 files, the only defined interaction type is the Coulomb interaction.
        if line.strip().lower() == '@<tripos>atom':
            self.itype = 'COUL'
            self.section = 'Atom'
        elif line.strip().lower() == '@<tripos>bond':
            self.itype = 'None'
            self.section = 'Bond'
        elif line.strip().lower() == '@<tripos>substructure':
            self.itype = 'None'
            self.section = 'Substructure'
        elif line.strip().lower() == '@<tripos>molecule':
            self.itype = 'None'
            self.section = 'Molecule'
        elif self.section == 'Molecule' and self.mol == None:
            self.mol = '_'.join(s)
        elif not is_mol2_atom(line):
            self.itype = 'None'

        if is_mol2_atom(line) and self.itype == 'COUL':
            #self.atomnames.append(s[self.pdict[self.itype]['Atom'][0]])
            #self.adict.setdefault(self.mol,[]).append(s[self.pdict[self.itype]['Atom'][0]])
            self.atomnames.append(s[0])
            self.adict.setdefault(self.mol,[]).append(s[0])

        if self.itype in self.pdict:
            if 'Atom' in self.pdict[self.itype] and match(' *[0-9]', line):
                # List the atoms in the interaction.
                #self.atom = [s[i] for i in self.pdict[self.itype]['Atom']]
                self.atom = [s[0]]
                # The suffix of the parameter ID is built from the atom    #
                # types/classes involved in the interaction.
                self.suffix = ':' + '-'.join([self.mol,''.join(self.atom)])
            #self.suffix = '.'.join(self.atom)
                self.molatom = (self.mol, self.atom if type(self.atom) is list else [self.atom])

class FrcMod_Reader(BaseReader):
    """Finite state machine for parsing FrcMod force field file."""
    
    def __init__(self,fnm):
        # Initialize the superclass. :)
        super(FrcMod_Reader,self).__init__(fnm)
        ## The parameter dictionary (defined in this file)
        self.pdict  = frcmod_pdict
        ## The atom numbers in the interaction (stored in the parser)
        self.atom   = []
        ## Whether we're inside the dihedral section
        self.dihe  = False
        ## The frcmod file never has any atoms in it
        self.adict = {None:None}
        
    def Split(self, line):
        return split(' +(?!-(?![0-9.]))', line.strip().replace('\n',''))

    def Whites(self, line):
        return findall(' +(?!-(?![0-9.]))', line.replace('\n',''))

    def feed(self, line):
        s          = self.Split(line)
        self.ln   += 1

        if len(line.strip()) == 0: 
            return
        if match('^dihe', line.strip().lower()):
            self.dihe = True
            return
        elif match('^mass$', line.strip().lower()):
            self.dihe  = False
            self.itype = 'MASS'
            return
        elif match('^bond$', line.strip().lower()):
            self.dihe  = False
            self.itype = 'BONDS'
            return
        elif match('^angle$', line.strip().lower()):
            self.dihe  = False
            self.itype = 'ANGLES'
            return
        elif match('^improper$', line.strip().lower()):
            self.dihe  = False
            self.itype = 'IDIHS'
            return
        elif match('^nonbon$', line.strip().lower()):
            self.dihe  = False
            self.itype = 'VDW'
            return
        elif len(s) == 0:
            self.dihe  = False
            return

        if self.dihe:
            try:
                self.itype = 'PDIHS%i' % int(np.abs(float(s[4])))
            except:
                self.itype = 'None'

        if self.itype in self.pdict:
            if 'Atom' in self.pdict[self.itype]:
                # List the atoms in the interaction.
                self.atom = [s[i].replace(" -","-") for i in self.pdict[self.itype]['Atom']]

            # The suffix of the parameter ID is built from the atom    #
            # types/classes involved in the interaction.
            self.suffix = ''.join(self.atom)

class AMBER(Engine):

    """ Engine for carrying out general purpose AMBER calculations. """

    def __init__(self, name="amber", **kwargs):
        ## Keyword args that aren't in this list are filtered out.
        self.valkwd = ['amberhome']
        super(AMBER,self).__init__(name=name, **kwargs)

    def setopts(self, **kwargs):
        
        """ Called by __init__ ; Set AMBER-specific options. """

        ## The directory containing TINKER executables (e.g. dynamic)
        if 'amberhome' in kwargs:
            self.amberhome = kwargs['amberhome']
            if not os.path.exists(os.path.join(self.amberhome,"sander")):
                warn_press_key("The 'sander' executable indicated by %s doesn't exist! (Check amberhome)" \
                                   % os.path.join(self.amberhome,"sander"))
        else:
            warn_once("The 'amberhome' option was not specified; using default.")
            if which('sander') == '':
                warn_press_key("Please add AMBER executables to the PATH or specify amberhome.")
            self.tinkerpath = which('sander')

    def readsrc(self, **kwargs):

        """ Called by __init__ ; read files from the source directory. """

        self.key = onefile('key', kwargs['tinker_key'] if 'tinker_key' in kwargs else None)
        self.prm = onefile('prm', kwargs['tinker_prm'] if 'tinker_prm' in kwargs else None)

        if 'mol' in kwargs:
            self.mol = kwargs['mol']
        elif 'coords' in kwargs:
            self.mol = Molecule(kwargs['coords'])
        else:
            pdbfile = onefile('pdb')
            if not arcfile: 
                logger.error('Cannot determine which .pdb file to use\n')
                raise RuntimeError
            self.mol = Molecule(pdbfile)

    def calltinker(self, command, stdin=None, print_to_screen=False, print_command=False, **kwargs):

        """ Call TINKER; prepend the tinkerpath to calling the TINKER program. """

        csplit = command.split()
        # Sometimes the engine changes dirs and the key goes missing, so we link it.
        if "%s.key" % self.name in csplit and not os.path.exists("%s.key" % self.name):
            LinkFile(self.abskey, "%s.key" % self.name)
        prog = os.path.join(self.tinkerpath, csplit[0])
        csplit[0] = prog
        o = _exec(' '.join(csplit), stdin=stdin, print_to_screen=print_to_screen, print_command=print_command, rbytes=1024, **kwargs)
        # Determine the TINKER version number.
        for line in o[:10]:
            if "Version" in line:
                vw = line.split()[2]
                if len(vw.split('.')) <= 2:
                    vn = float(vw)
                else:
                    vn = float(vw.split('.')[:2])
                vn_need = 6.3
                try:
                    if vn < vn_need:
                        if self.warn_vn: 
                            warn_press_key("ForceBalance requires TINKER %.1f - unexpected behavior with older versions!" % vn_need)
                        self.warn_vn = True
                except:
                    logger.error("Unable to determine TINKER version number!\n")
                    raise RuntimeError
        for line in o[-10:]:
            # Catch exceptions since TINKER does not have exit status.
            if "TINKER is Unable to Continue" in line:
                for l in o:
                    logger.error("%s\n" % l)
                time.sleep(1)
                logger.error("TINKER may have crashed! (See above output)\nThe command was: %s\nThe directory was: %s\n" % (' '.join(csplit), os.getcwd()))
                raise RuntimeError
                break
        for line in o:
            if 'D+' in line:
                logger.info(line+'\n')
                warn_press_key("TINKER returned a very large floating point number! (See above line; will give error on parse)")
        return o

    def prepare(self, pbc=False, **kwargs):

        """ Called by __init__ ; prepare the temp directory and figure out the topology. """

        # Call TINKER but do nothing to figure out the version number.
        o = self.calltinker("dynamic", persist=1, print_error=False)

        self.rigid = False

        ## Attempt to set some TINKER options.
        tk_chk = []
        tk_opts = OrderedDict([("digits", "10"), ("archive", "")])
        tk_defs = OrderedDict()
        
        prmtmp = False

        if hasattr(self,'FF'):
            if not os.path.exists(self.FF.tinkerprm):
                # If the parameter files don't already exist, create them for the purpose of
                # preparing the engine, but then delete them afterward.
                prmtmp = True
                self.FF.make(np.zeros(self.FF.np))
            if self.FF.rigid_water:
                tk_opts["rattle"] = "water"
                self.rigid = True
            if self.FF.amoeba_pol == 'mutual':
                tk_opts['polarization'] = 'mutual'
                if self.FF.amoeba_eps != None:
                    tk_opts['polar-eps'] = str(self.FF.amoeba_eps)
                else:
                    tk_defs['polar-eps'] = '1e-6'
            elif self.FF.amoeba_pol == 'direct':
                tk_opts['polarization'] = 'direct'
            else:
                warn_press_key("Using TINKER without explicitly specifying AMOEBA settings. Are you sure?")
            self.prm = self.FF.tinkerprm
            prmfnm = self.FF.tinkerprm
        elif self.prm:
            prmfnm = self.prm
        else:
            prmfnm = None

        # Periodic boundary conditions may come from the TINKER .key file.
        keypbc = False
        minbox = 1e10
        if self.key:
            for line in open(os.path.join(self.srcdir, self.key)).readlines():
                s = line.split()
                if len(s) > 0 and s[0].lower() == 'a-axis':
                    keypbc = True
                    minbox = float(s[1])
                if len(s) > 0 and s[0].lower() == 'b-axis' and float(s[1]) < minbox:
                    minbox = float(s[1])
                if len(s) > 0 and s[0].lower() == 'c-axis' and float(s[1]) < minbox:
                    minbox = float(s[1])
            if keypbc and (not pbc):
                warn_once("Deleting PBC options from the .key file.")
                tk_opts['a-axis'] = None
                tk_opts['b-axis'] = None
                tk_opts['c-axis'] = None
                tk_opts['alpha'] = None
                tk_opts['beta'] = None
                tk_opts['gamma'] = None
        if pbc:
            if (not keypbc) and 'boxes' not in self.mol.Data:
                logger.error("Periodic boundary conditions require either (1) a-axis to be in the .key file or (b) boxes to be in the coordinate file.\n")
                raise RuntimeError
        self.pbc = pbc
        if pbc:
            tk_opts['ewald'] = ''
            if minbox <= 10:
                warn_press_key("Periodic box is set to less than 10 Angstroms across")
            # TINKER likes to use up to 7.0 Angstrom for PME cutoffs
            rpme = 0.05*(float(int(minbox - 1))) if minbox <= 15 else 7.0
            tk_defs['ewald-cutoff'] = "%f" % rpme
            # TINKER likes to use up to 9.0 Angstrom for vdW cutoffs
            rvdw = 0.05*(float(int(minbox - 1))) if minbox <= 19 else 9.0
            tk_defs['vdw-cutoff'] = "%f" % rvdw
            if (minbox*0.5 - rpme) > 2.5 and (minbox*0.5 - rvdw) > 2.5:
                tk_defs['neighbor-list'] = ''
            elif (minbox*0.5 - rpme) > 2.5:
                tk_defs['mpole-list'] = ''
        else:
            tk_opts['ewald'] = None
            tk_opts['ewald-cutoff'] = None
            tk_opts['vdw-cutoff'] = None
            # This seems to have no effect on the kinetic energy.
            # tk_opts['remove-inertia'] = '0'

        write_key("%s.key" % self.name, tk_opts, os.path.join(self.srcdir, self.key) if self.key else None, tk_defs, verbose=False, prmfnm=prmfnm)
        self.abskey = os.path.abspath("%s.key")

        self.mol[0].write(os.path.join("%s.xyz" % self.name), ftype="tinker")

        ## If the coordinates do not come with TINKER suffixes then throw an error.
        self.mol.require('tinkersuf')

        ## Call analyze to read information needed to build the atom lists.
        o = self.calltinker("analyze %s.xyz P,C" % (self.name), stdin="ALL")

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
        if len(G.nodes()) > 0:
            # The following code only works in TINKER 6.2
            gs = nx.connected_component_subgraphs(G)
            tmols = [gs[i] for i in np.argsort(np.array([min(g.nodes()) for g in gs]))]
            self.AtomLists['MoleculeNumber'] = [[i+1 in m.nodes() for m in tmols].index(1) for i in range(self.mol.na)]
        else:
            grouped = [i.L() for i in self.mol.molecules]
            self.AtomLists['MoleculeNumber'] = [[i in g for g in grouped].index(1) for i in range(self.mol.na)]
        if prmtmp:
            for f in self.FF.fnms: 
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
        if not self.pbc:
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

    def evaluate_(self, xyzin, force=False, dipole=False):

        """ 
        Utility function for computing energy, and (optionally) forces and dipoles using TINKER. 
        
        Inputs:
        xyzin: TINKER .xyz file name.
        force: Switch for calculating the force.
        dipole: Switch for calculating the dipole.

        Outputs:
        Result: Dictionary containing energies, forces and/or dipoles.
        """

        Result = OrderedDict()
        # If we want the dipoles (or just energies), analyze is the way to go.
        if dipole or (not force):
            oanl = self.calltinker("analyze %s -k %s" % (xyzin, self.name), stdin="G,E,M", print_to_screen=False)
            # Read potential energy and dipole from file.
            eanl = []
            dip = []
            for line in oanl:
                s = line.split()
                if 'Total Potential Energy : ' in line:
                    eanl.append(float(s[4]) * 4.184)
                if dipole:
                    if 'Dipole X,Y,Z-Components :' in line:
                        dip.append([float(s[i]) for i in range(-3,0)])
            Result["Energy"] = np.array(eanl)
            Result["Dipole"] = np.array(dip)
        # If we want forces, then we need to call testgrad.
        if force:
            E = []
            F = []
            Fi = []
            o = self.calltinker("testgrad %s -k %s y n n" % (xyzin, self.name))
            i = 0
            ReadFrc = 0
            for line in o:
                s = line.split()
                if "Total Potential Energy" in line:
                    E.append(float(s[4]) * 4.184)
                if "Cartesian Gradient Breakdown over Individual Atoms" in line:
                    ReadFrc = 1
                if ReadFrc and len(s) == 6 and all([s[0] == 'Anlyt',isint(s[1]),isfloat(s[2]),isfloat(s[3]),isfloat(s[4]),isfloat(s[5])]):
                    ReadFrc = 2
                    if self.AtomMask[i]:
                        Fi += [-1 * float(j) * 4.184 * 10 for j in s[2:5]]
                    i += 1
                if ReadFrc == 2 and len(s) < 6:
                    ReadFrc = 0
                    F.append(Fi)
                    Fi = []
                    i = 0
            Result["Energy"] = np.array(E)
            Result["Force"] = np.array(F)
        return Result

    def energy_force_one(self, shot):

        """ Computes the energy and force using TINKER for one snapshot. """

        self.mol[shot].write("%s.xyz" % self.name, ftype="tinker")
        Result = self.evaluate_("%s.xyz" % self.name, force=True)
        return np.hstack((Result["Energy"].reshape(-1,1), Result["Force"]))

    def energy(self):

        """ Computes the energy using TINKER over a trajectory. """

        if hasattr(self, 'mdtraj') : 
            x = self.mdtraj
        else:
            x = "%s.xyz" % self.name
            self.mol.write(x, ftype="tinker")
        return self.evaluate_(x)["Energy"]

    def energy_force(self):

        """ Computes the energy and force using TINKER over a trajectory. """

        if hasattr(self, 'mdtraj') : 
            x = self.mdtraj
        else:
            x = "%s.xyz" % self.name
            self.mol.write(x, ftype="tinker")
        Result = self.evaluate_(x, force=True)
        return np.hstack((Result["Energy"].reshape(-1,1), Result["Force"]))

    def energy_dipole(self):

        """ Computes the energy and dipole using TINKER over a trajectory. """

        if hasattr(self, 'mdtraj') : 
            x = self.mdtraj
        else:
            x = "%s.xyz" % self.name
            self.mol.write(x, ftype="tinker")
        Result = self.evaluate_(x, dipole=True)
        return np.hstack((Result["Energy"].reshape(-1,1), Result["Dipole"]))

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
            logger.error("Total potential energy wasn't encountered when calling analyze!\n")
            raise RuntimeError
        if optimize and abs(E-E_) > 0.1:
            warn_press_key("Energy from optimize and analyze aren't the same (%.3f vs. %.3f)" % (E, E_))
        return E, rmsd

    def interaction_energy(self, fraga, fragb):
        
        """ Calculate the interaction energy for two fragments. """

        self.A = TINKER(name="A", mol=self.mol.atom_select(fraga), tinker_key="%s.key" % self.name, tinkerpath=self.tinkerpath)
        self.B = TINKER(name="B", mol=self.mol.atom_select(fragb), tinker_key="%s.key" % self.name, tinkerpath=self.tinkerpath)

        # Interaction energy needs to be in kcal/mol.
        return (self.energy() - self.A.energy() - self.B.energy()) / 4.184

    def molecular_dynamics(self, nsteps, timestep, temperature=None, pressure=None, nequil=0, nsave=1000, minimize=True, anisotropic=False, threads=1, verbose=False, **kwargs):
        
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
        threads     = (int)   Specify how many OpenMP threads to use

        Returns simulation data:
        Rhos        = (array)     Density in kilogram m^-3
        Potentials  = (array)     Potential energies
        Kinetics    = (array)     Kinetic energies
        Volumes     = (array)     Box volumes
        Dips        = (3xN array) Dipole moments
        EComps      = (dict)      Energy components
        """

        md_defs = OrderedDict()
        md_opts = OrderedDict()
        # Print out averages only at the end.
        md_opts["printout"] = nsave
        md_opts["openmp-threads"] = threads
        # Langevin dynamics for temperature control.
        if temperature != None:
            md_defs["integrator"] = "stochastic"
        else:
            md_defs["integrator"] = "beeman"
            md_opts["thermostat"] = None
        # Periodic boundary conditions.
        if self.pbc:
            md_opts["vdw-correction"] = ''
            if temperature != None and pressure != None: 
                md_defs["integrator"] = "beeman"
                md_defs["thermostat"] = "bussi"
                md_defs["barostat"] = "montecarlo"
                if anisotropic:
                    md_opts["aniso-pressure"] = ''
            elif pressure != None:
                warn_once("Pressure is ignored because temperature is turned off.")
        else:
            if pressure != None:
                warn_once("Pressure is ignored because pbc is set to False.")
            # Use stochastic dynamics for the gas phase molecule.
            # If we use the regular integrators it may miss
            # six degrees of freedom in calculating the kinetic energy.
            md_opts["barostat"] = None

        eq_opts = deepcopy(md_opts)
        if self.pbc and temperature != None and pressure != None: 
            eq_opts["integrator"] = "beeman"
            eq_opts["thermostat"] = "bussi"
            eq_opts["barostat"] = "berendsen"

        if minimize:
            if verbose: logger.info("Minimizing the energy...")
            self.optimize(method="bfgs", crit=1)
            os.system("mv %s.xyz_2 %s.xyz" % (self.name, self.name))
            if verbose: logger.info("Done\n")

        # Run equilibration.
        if nequil > 0:
            write_key("%s-eq.key" % self.name, eq_opts, "%s.key" % self.name, md_defs)
            if verbose: printcool("Running equilibration dynamics", color=0)
            if self.pbc and pressure != None:
                self.calltinker("dynamic %s -k %s-eq %i %f %f 4 %f %f" % (self.name, self.name, nequil, timestep, float(nsave*timestep)/1000, 
                                                                          temperature, pressure), print_to_screen=verbose)
            else:
                self.calltinker("dynamic %s -k %s-eq %i %f %f 2 %f" % (self.name, self.name, nequil, timestep, float(nsave*timestep)/1000,
                                                                       temperature), print_to_screen=verbose)
            os.system("rm -f %s.arc" % (self.name))

        # Run production.
        if verbose: printcool("Running production dynamics", color=0)
        write_key("%s-md.key" % self.name, md_opts, "%s.key" % self.name, md_defs)
        if self.pbc and pressure != None:
            odyn = self.calltinker("dynamic %s -k %s-md %i %f %f 4 %f %f" % (self.name, self.name, nsteps, timestep, float(nsave*timestep/1000), 
                                                                             temperature, pressure), print_to_screen=verbose)
        else:
            odyn = self.calltinker("dynamic %s -k %s-md %i %f %f 2 %f" % (self.name, self.name, nsteps, timestep, float(nsave*timestep/1000), 
                                                                          temperature), print_to_screen=verbose)
            
        # Gather information.
        os.system("mv %s.arc %s-md.arc" % (self.name, self.name))
        self.mdtraj = "%s-md.arc" % self.name
        edyn = []
        kdyn = []
        temps = []
        for line in odyn:
            s = line.split()
            if 'Current Potential' in line:
                edyn.append(float(s[2]))
            if 'Current Kinetic' in line:
                kdyn.append(float(s[2]))
            if len(s) > 0 and s[0] == 'Temperature' and s[2] == 'Kelvin':
                temps.append(float(s[1]))

        # Potential and kinetic energies converted to kJ/mol.
        edyn = np.array(edyn) * 4.184
        kdyn = np.array(kdyn) * 4.184
        temps = np.array(temps)
    
        if verbose: logger.info("Post-processing to get the dipole moments\n")
        oanl = self.calltinker("analyze %s-md.arc" % self.name, stdin="G,E,M", print_to_screen=False)

        # Read potential energy and dipole from file.
        eanl = []
        dip = []
        mass = 0.0
        ecomp = OrderedDict()
        havekeys = set()
        first_shot = True
        for ln, line in enumerate(oanl):
            strip = line.strip()
            s = line.split()
            if 'Total System Mass' in line:
                mass = float(s[-1])
            if 'Total Potential Energy : ' in line:
                eanl.append(float(s[4]))
            if 'Dipole X,Y,Z-Components :' in line:
                dip.append([float(s[i]) for i in range(-3,0)])
            if first_shot:
                for key in eckeys:
                    if strip.startswith(key):
                        if key in ecomp:
                            ecomp[key].append(float(s[-2])*4.184)
                        else:
                            ecomp[key] = [float(s[-2])*4.184]
                        if key in havekeys:
                            first_shot = False
                        havekeys.add(key)
            else:
                for key in havekeys:
                    if strip.startswith(key):
                        if key in ecomp:
                            ecomp[key].append(float(s[-2])*4.184)
                        else:
                            ecomp[key] = [float(s[-2])*4.184]
        for key in ecomp:
            ecomp[key] = np.array(ecomp[key])
        ecomp["Potential Energy"] = edyn
        ecomp["Kinetic Energy"] = kdyn
        ecomp["Temperature"] = temps
        ecomp["Total Energy"] = edyn+kdyn

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
            vol = np.array([BuildLatticeFromLengthsAngles(*[float(j) for j in line.split()]).V \
                                for line in open("%s-md.arc" % self.name).readlines() \
                                if (len(line.split()) == 6 and isfloat(line.split()[1]) \
                                        and all([isfloat(i) for i in line.split()[:6]]))]) / 1000
            rho = conv * mass / vol
        else:
            vol = None
            rho = None
        prop_return = OrderedDict()
        prop_return.update({'Rhos': rho, 'Potentials': edyn, 'Kinetics': kdyn, 'Volumes': vol, 'Dips': dip, 'Ecomps': ecomp})
        return prop_return


class AbInitio_AMBER(AbInitio):

    """Subclass of Target for force and energy matching
    using AMBER.  Implements the prepare and energy_force_driver
    methods.  The get method is in the base class.  """

    def __init__(self,options,tgt_opts,forcefield):
        ## Name of the trajectory, we need this BEFORE initializing the SuperClass
        self.coords = "all.gro"
        super(AbInitio_AMBER,self).__init__(options,tgt_opts,forcefield)
        ## all_at_once is not implemented.
        self.all_at_once = True

    def prepare_temp_directory(self, options, tgt_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        LinkFile(os.path.join(self.root,self.tgtdir,"force.mdin"),os.path.join(abstempdir,"force.mdin"))
        LinkFile(os.path.join(self.root,self.tgtdir,"stage.leap"),os.path.join(abstempdir,"stage.leap"))
        # I also need to write the trajectory
        if 'boxes' in self.mol.Data.keys():
            del self.mol.Data['boxes']
        self.mol.write(os.path.join(abstempdir,"all.mdcrd"))

    def energy_force_driver_all_external_(self):
        ## Create the run input files (inpcrd, prmtop) from the force field file.  
        ## Note that the frcmod and mol2 files are required.
        ## This is like 'grompp' in GROMACS.
        _exec("tleap -f stage.leap", print_to_screen=False, print_command=False)
        ## This line actually runs AMBER.
        _exec("sander -i force.mdin -o force.mdout -p prmtop -c inpcrd -y all.mdcrd -O", print_to_screen=False, print_command=False)
        ## Simple parser for 
        ParseMode = 0
        Energies = []
        Forces = []
        Force = []
        for line in open('forcedump.dat'):
            line = line.strip()
            sline = line.split()
            if ParseMode == 1:
                if len(sline) == 1 and isfloat(sline[0]):
                    Energies.append(float(sline[0]) * 4.184)
                    ParseMode = 0
            if ParseMode == 2:
                if len(sline) == 3 and all(isfloat(sline[i]) for i in range(3)):
                    Force += [float(sline[i]) * 4.184 * 10 for i in range(3)]
                if len(Force) == 3*self.qmatoms:
                    Forces.append(np.array(Force))
                    Force = []
                    ParseMode = 0
            if line == '0 START of Energies':
                ParseMode = 1
            elif line == '1 Total Force':
                ParseMode = 2

        Energies = np.array(Energies[1:])
        Forces = np.array(Forces[1:])
        
        M = np.hstack((Energies.reshape(-1,1), Forces))

        return M

    def energy_force_driver_all(self):
        return self.energy_force_driver_all_external_()
