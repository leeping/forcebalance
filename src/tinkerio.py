""" @package forcebalance.tinkerio TINKER input/output.

This serves as a good template for writing future force matching I/O
modules for other programs because it's so simple.

@author Lee-Ping Wang
@date 01/2012
"""

import os, shutil
from re import match, sub
from forcebalance.nifty import *
import numpy as Np
from forcebalance.basereader import BaseReader
from subprocess import Popen, PIPE
from forcebalance.abinitio import AbInitio
from vibration import Vibration
from forcebalance.moments import Moments
from forcebalance.liquid import Liquid
from forcebalance.molecule import Molecule
from forcebalance.binding import BindingEnergy
from forcebalance.interaction import Interaction
from forcebalance.finite_difference import in_fd
from collections import OrderedDict
from forcebalance.optimizer import GoodStep
try:
    from simtk.unit import *
except: pass

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
        vdw           2               2.6550     0.0135      0.910 # PARM 4

        multipole     2    1    2               0.25983
                                               -0.03859    0.00000   -0.05818
                                               -0.03673
                                                0.00000   -0.10739
                                               -0.00203    0.00000    0.14412
        @endverbatim

        The '#PARM 4' has no effect on TINKER but it indicates that we
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
                print line
                warn_press_key("The above line was found in %s, but we expected something like %s" % (src,prmfnm))
        outlines.append(line)
    if not prmflag:
        print "Adding parameter file %s to key file" % prmfnm
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

    def npt_simulation(self, temperature, pressure):
        """ Submit a NPT simulation to the Work Queue. """
        wq = getWorkQueue()
        if not (os.path.exists('npt_result.p') or os.path.exists('npt_result.p.bz2')):
            link_dir_contents(os.path.join(self.root,self.rundir),os.getcwd())
            if wq == None:
                print "Running condensed phase simulation locally."
                print "You may tail -f %s/npt_tinker.out in another terminal window" % os.getcwd()
                if GoodStep() and (temperature, pressure) in self.DynDict_New:
                    self.DynDict[(temperature, pressure)] = self.DynDict_New[(temperature, pressure)]
                if (temperature, pressure) in self.DynDict:
                    dynsrc = self.DynDict[(temperature, pressure)]
                    dyndest = os.path.join(os.getcwd(), 'liquid.dyn')
                    print "Copying .dyn file: %s to %s" % (dynsrc, dyndest)
                    shutil.copy2(dynsrc,dyndest)
                cmdstr = 'python npt_tinker.py liquid.xyz %i %.3f %.3f %.3f %.3f --liquid_equ_steps %i &> npt_tinker.out' % (self.liquid_prod_steps, self.liquid_timestep, self.liquid_interval, temperature, pressure, self.liquid_equ_steps)
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
                

class AbInitio_TINKER(AbInitio):

    """Subclass of Target for force and energy matching
    using TINKER.  Implements the prepare and energy_force_driver
    methods.  """

    def __init__(self,options,tgt_opts,forcefield):
        ## Name of the trajectory
        self.trajfnm = "all.arc"
        super(AbInitio_TINKER,self).__init__(options,tgt_opts,forcefield)
        ## all_at_once is not implemented.
        if self.force and self.all_at_once:
            warn_press_key("Currently, TINKER can only do trajectory loops for energy-only jobs.")
            self.all_at_once = False

    def prepare_temp_directory(self, options, tgt_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        # Link the necessary programs into the temporary directory
        LinkFile(os.path.join(options['tinkerpath'],"testgrad"),os.path.join(abstempdir,"testgrad"))
        LinkFile(os.path.join(options['tinkerpath'],"analyze"),os.path.join(abstempdir,"analyze"))
        write_key_with_prm(os.path.join(self.root,self.tgtdir,"shot.key"),os.path.join(abstempdir,"shot.key"),ffobj=self.FF)
        # LinkFile(os.path.join(self.root,self.tgtdir,"shot.key"),os.path.join(abstempdir,"shot.key"))

    def energy_force_driver(self, shot):
        self.traj.write("shot.arc",select=[shot])
        # This line actually runs TINKER
        o, e = Popen(["./testgrad","shot.arc","y","n"],stdout=PIPE,stderr=PIPE).communicate()
        # Read data from stdout and stderr, and convert it to GROMACS
        # units for consistency with existing code.
        F = []
        for line in o.split('\n'):
            s = line.split()
            if "Total Potential Energy" in line:
                E = [float(s[4]) * 4.184]
            elif len(s) == 6 and all([s[0] == 'Anlyt',isint(s[1]),isfloat(s[2]),isfloat(s[3]),isfloat(s[4]),isfloat(s[5])]):
                F += [-1 * float(i) * 4.184 * 10 for i in s[2:5]]
        M = Np.array(E + F)
        return M

    def energy_driver_all(self):
        self.traj.write("shot.arc")
        # This line actually runs TINKER
        o, e = Popen(["./analyze","shot.arc","e"],stdout=PIPE,stderr=PIPE).communicate()
        # Read data from stdout and stderr, and convert it to GROMACS units.
        E = []
        for line in o.split('\n'):
            s = line.split()
            if "Total Potential Energy" in line:
                E.append([float(s[4]) * 4.184])
        M = Np.array(E)
        return M

    def energy_force_driver_all(self):
        if self.force:
            raise Exception('Trying to call unimplemented functionality.')
        return self.energy_driver_all()

class Vibration_TINKER(Vibration):

    """Subclass of Target for vibrational frequency matching
    using TINKER.  Provides optimized geometry, vibrational frequencies (in cm-1),
    and eigenvectors."""

    def __init__(self,options,tgt_opts,forcefield):
        super(Vibration_TINKER,self).__init__(options,tgt_opts,forcefield)
        if self.FF.rigid_water:
            raise Exception('This class cannot be used with rigid water molecules.')

    def prepare_temp_directory(self, options, tgt_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        # Link the necessary programs into the temporary directory
        LinkFile(os.path.join(options['tinkerpath'],"vibrate"),os.path.join(abstempdir,"vibrate"))
        LinkFile(os.path.join(options['tinkerpath'],"optimize"),os.path.join(abstempdir,"optimize"))
        # Link the run parameter file
        write_key_with_prm(os.path.join(self.root,self.tgtdir,"input.key"),os.path.join(abstempdir,"input.key"),ffobj=self.FF)
        # LinkFile(os.path.join(self.root,self.tgtdir,"input.key"),os.path.join(abstempdir,"input.key"))
        LinkFile(os.path.join(self.root,self.tgtdir,"input.xyz"),os.path.join(abstempdir,"input.xyz"))

    def vibration_driver(self):
        # This line actually runs TINKER
        o, e = Popen(["./optimize","input.xyz","1.0e-6"],stdout=PIPE,stderr=PIPE).communicate()
        o, e = Popen(["./vibrate","input.xyz_2","a"],stdout=PIPE,stderr=PIPE).communicate()
        # Read the TINKER output.  The vibrational frequencies are ordered.
        # The first six modes are ignored
        moden = -6
        readev = False
        calc_eigvals = []
        calc_eigvecs = []
        for line in o.split('\n'):
            s = line.split()
            if "Vibrational Normal Mode" in line:
                moden += 1
                freq = float(s[-2])
                readev = False
                if moden > 0:
                    calc_eigvals.append(freq)
                    calc_eigvecs.append([])
            elif "Atom" in line and "Delta X" in line:
                readev = True
            elif moden > 0 and readev and len(s) == 4 and all([isint(s[0]), isfloat(s[1]), isfloat(s[2]), isfloat(s[3])]):
                calc_eigvecs[-1].append([float(i) for i in s[1:]])
        calc_eigvals = Np.array(calc_eigvals)
        calc_eigvecs = Np.array(calc_eigvecs)
        os.system("rm -rf *_* *[0-9][0-9][0-9]*")

        return calc_eigvals, calc_eigvecs

class Moments_TINKER(Moments):

    """Subclass of Target for multipole moment matching
    using TINKER."""

    def __init__(self,options,tgt_opts,forcefield):
        super(Moments_TINKER,self).__init__(options,tgt_opts,forcefield)
        if self.FF.rigid_water:
            raise Exception('This class cannot be used with rigid water molecules.')

    def prepare_temp_directory(self, options, tgt_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        # Link the necessary programs into the temporary directory
        LinkFile(os.path.join(options['tinkerpath'],"analyze"),os.path.join(abstempdir,"analyze"))
        LinkFile(os.path.join(options['tinkerpath'],"polarize"),os.path.join(abstempdir,"polarize"))
        LinkFile(os.path.join(options['tinkerpath'],"optimize"),os.path.join(abstempdir,"optimize"))
        # Link the run parameter file
        write_key_with_prm(os.path.join(self.root,self.tgtdir,"input.key"),os.path.join(abstempdir,"input.key"),ffobj=self.FF)
        # LinkFile(os.path.join(self.root,self.tgtdir,"input.key"),os.path.join(abstempdir,"input.key"))
        LinkFile(os.path.join(self.root,self.tgtdir,"input.xyz"),os.path.join(abstempdir,"input.xyz"))

    def moments_driver(self):
        # This line actually runs TINKER
        if self.optimize_geometry:
            o, e = Popen(["./optimize","input.xyz","1.0e-6"],stdout=PIPE,stderr=PIPE).communicate()
            o, e = Popen(["./analyze","input.xyz_2","M"],stdout=PIPE,stderr=PIPE).communicate()
        else:
            o, e = Popen(["./analyze","input.xyz","M"],stdout=PIPE,stderr=PIPE).communicate()
        # Read the TINKER output.
        qn = -1
        ln = 0
        for line in o.split('\n'):
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

        if 'polarizability' in self.ref_moments:
            if self.optimize_geometry:
                o, e = Popen(["./polarize","input.xyz_2"],stdout=PIPE,stderr=PIPE).communicate()
            else:
                o, e = Popen(["./polarize","input.xyz"],stdout=PIPE,stderr=PIPE).communicate()
            # Read the TINKER output.
            pn = -1
            ln = 0
            polarizability_dict = OrderedDict()
            for line in o.split('\n'):
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

        os.system("rm -rf *_* *[0-9][0-9][0-9]*")
        return calc_moments

class BindingEnergy_TINKER(BindingEnergy):

    """Subclass of BindingEnergy for binding energy matching
    using TINKER.  """

    def __init__(self,options,tgt_opts,forcefield):
        super(BindingEnergy_TINKER,self).__init__(options,tgt_opts,forcefield)
        self.prepare_temp_directory(options, tgt_opts)

    def prepare_temp_directory(self, options, tgt_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        if self.FF.rigid_water:
            self.optprog = "optrigid"
            #LinkFile(os.path.join(self.root,self.tgtdir,"rigid.key"),os.path.join(abstempdir,"rigid.key"))
        else:
            self.optprog = "optimize"
        # Link the necessary programs into the temporary directory
        LinkFile(os.path.join(options['tinkerpath'],"analyze"),os.path.join(abstempdir,"analyze"))
        LinkFile(os.path.join(options['tinkerpath'],self.optprog),os.path.join(abstempdir,self.optprog))
        LinkFile(os.path.join(options['tinkerpath'],"superpose"),os.path.join(abstempdir,"superpose"))
        # Link the run parameter file
        # The master file might be unneeded??
        for sysname,sysopt in self.sys_opts.items():
            if self.FF.rigid_water:
                # Make every water molecule rigid
                # WARNING: Hard coded values here!
                M = Molecule(os.path.join(self.root, self.tgtdir, sysopt['geometry']),ftype="tinker")
                for a in range(0, len(M.xyzs[0]), 3):
                    flex = M.xyzs[0]
                    wat = flex[a:a+3]
                    com = wat.mean(0)
                    wat -= com
                    o  = wat[0]
                    h1 = wat[1]
                    h2 = wat[2]
                    r1 = h1 - o
                    r2 = h2 - o
                    r1 /= Np.linalg.norm(r1)
                    r2 /= Np.linalg.norm(r2)
                    # Obtain unit vectors.
                    ex = r1 + r2
                    ey = r1 - r2
                    ex /= Np.linalg.norm(ex)
                    ey /= Np.linalg.norm(ey)
                    Bond = 0.9572
                    Ang = Np.pi * 104.52 / 2 / 180
                    cosx = Np.cos(Ang)
                    cosy = Np.sin(Ang)
                    h1 = o + Bond*ex*cosx + Bond*ey*cosy
                    h2 = o + Bond*ex*cosx - Bond*ey*cosy
                    rig = Np.array([o, h1, h2]) + com
                    M.xyzs[0][a:a+3] = rig
                M.write(os.path.join(abstempdir,sysopt['geometry']),ftype="tinker")
            else:
                M = Molecule(os.path.join(self.root, self.tgtdir, sysopt['geometry']),ftype="tinker")
                if 'select' in sysopt:
                    atomselect = Np.array(uncommadash(sysopt['select']))
                    #atomselect = eval("Np.arange(M.na)"+sysopt['select'])
                    M = M.atom_select(atomselect)
                M.write(os.path.join(abstempdir,sysname+".xyz"),ftype="tinker")
                write_key_with_prm(os.path.join(self.root,self.tgtdir,sysopt['keyfile']),os.path.join(abstempdir,sysname+".key"),ffobj=self.FF)
                # LinkFile(os.path.join(self.root,self.tgtdir,sysopt['keyfile']),os.path.join(abstempdir,sysname+".key"))

    def system_driver(self,sysname):
        sysopt = self.sys_opts[sysname]
        rmsd = 0.0
        # This line actually runs TINKER
        xyzfnm = sysname+".xyz"
        if 'optimize' in sysopt and sysopt['optimize'] == True:
            if self.FF.rigid_water:
                #os.system("cp rigid.key %s" % os.path.splitext(xyzfnm)[0] + ".key")
                o, e = Popen(["./%s" % self.optprog,xyzfnm,"1e-4"],stdout=PIPE,stderr=PIPE).communicate()
            else:
                o, e = Popen(["./%s" % self.optprog,xyzfnm,"1e-4"],stdout=PIPE,stderr=PIPE).communicate()
            cnvgd = 0
            for line in o.split('\n'):
                if "Normal Termination" in line:
                    cnvgd = 1
            if not cnvgd:
                print o
                print "The system %s did not converge in the geometry optimization - printout is above." % sysname
                #warn_press_key("The system %s did not converge in the geometry optimization" % sysname)
            o, e = Popen(["./analyze",xyzfnm+'_2',"E"],stdout=PIPE,stderr=PIPE).communicate()
            if self.FF.rigid_water:
                oo, ee = Popen(['./superpose', xyzfnm, xyzfnm+'_2', '1', 'y', 'u', 'n', '0'], stdout=PIPE, stderr=PIPE).communicate()
            else:
                oo, ee = Popen(['./superpose', xyzfnm, xyzfnm+'_2', '1', 'y', 'u', 'n', '0'], stdout=PIPE, stderr=PIPE).communicate()
            for line in oo.split('\n'):
                if "Root Mean Square Distance" in line:
                    rmsd = float(line.split()[-1])
            os.system("rm %s" % xyzfnm+'_2')
        else:
            o, e = Popen(["./analyze",xyzfnm,"E"],stdout=PIPE,stderr=PIPE).communicate()
        # Read the TINKER output. 
        for line in o.split('\n'):
            if "Total Potential Energy" in line:
                return float(line.split()[-2].replace('D','e')) * kilocalories_per_mole, rmsd * angstrom
        warn_press_key("Total potential energy wasn't encountered for system %s!" % sysname)
    
class Interaction_TINKER(Interaction):

    """Subclass of Target for interaction matching using TINKER. """

    def __init__(self,options,tgt_opts,forcefield):
        ## Name of the trajectory
        self.trajfnm = "all.arc"
        super(Interaction_TINKER,self).__init__(options,tgt_opts,forcefield)

    def prepare_temp_directory(self, options, tgt_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        # Link the necessary programs into the temporary directory
        LinkFile(os.path.join(options['tinkerpath'],"analyze"),os.path.join(abstempdir,"analyze"))
        # Link the run parameter file
        write_key_with_prm(os.path.join(self.root,self.tgtdir,"shot.key"),os.path.join(abstempdir,"shot.key"),ffobj=self.FF)
        # LinkFile(os.path.join(self.root,self.tgtdir,"shot.key"),os.path.join(abstempdir,"shot.key"))

    def energy_driver_all(self,select=None):
        if select == None:
            self.traj.write("shot.arc")
        else:
            traj = self.traj.atom_select(select)
            traj.write("shot.arc")
        # This line actually runs TINKER
        o, e = Popen(["./analyze","shot.arc","e"],stdout=PIPE,stderr=PIPE).communicate()
        # Read data from stdout and stderr, and convert it to GROMACS units.
        E = []
        for line in o.split('\n'):
            s = line.split()
            if "Total Potential Energy" in line:
                E.append(float(s[4]) * 4.184)
        M = Np.array(E)
        return M

    def interaction_driver_all(self,dielectric=False):
        # Compute the energies for the dimer
        D = self.energy_driver_all()
        A = self.energy_driver_all(select=self.select1)
        B = self.energy_driver_all(select=self.select2)
        return D - A - B

