""" @package forcebalance.opt_geo_target Optimized Geometry fitting module.

@author Yudong Qiu, Lee-Ping Wang
@date 03/2019
"""
from __future__ import division
import os
import shutil
import numpy as np
import re
import subprocess
from collections import OrderedDict
from forcebalance.nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, printcool_dictionary, bohr2ang, warn_press_key
from forcebalance.target import Target
from forcebalance.molecule import Molecule
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from forcebalance.output import getLogger
logger = getLogger(__name__)

def periodic_diff(a, b, v_periodic=np.pi*2):
    ''' convenient function for computing the minimum difference in periodic coordinates
    Parameters
    ----------
    a: np.ndarray or float
        Reference values in a numpy array
    b: np.ndarray or float
        Target values in a numpy arrary
    v_periodic: float > 0
        Value of the periodic boundary

    Returns
    -------
    diff: np.ndarray
        The array of same shape containing the difference between a and b
        All return values are in range [-v_periodic/2, v_periodic/2),
        "( )" means exclusive, "[ ]" means inclusive

    Examples
    -------
    periodic_diff(0.0, 2.1, 2.0) => -0.1
    periodic_diff(0.0, 1.9, 2.0) => 0.1
    periodic_diff(0.0, 1.0, 2.0) => -1.0
    periodic_diff(1.0, 0.0, 2.0) => -1.0
    periodic_diff(1.0, 0.1, 2.0) => -0.9
    periodic_diff(1.0, 10.1, 2.0) => 0.9
    periodic_diff(1.0, 9.9, 2.0) => -0.9
    '''
    assert v_periodic > 0
    h = 0.5 * v_periodic
    return (a - b + h) % v_periodic - h



class OptGeoTarget(Target):
    """ Subclass of Target for fitting MM optimized geometries to QM optimized geometries. """
    def __init__(self,options,tgt_opts,forcefield):
        super(OptGeoTarget,self).__init__(options,tgt_opts,forcefield)
        self.set_option(None, None, 'optgeo_options', os.path.join(self.tgtdir,tgt_opts['optgeo_options_txt']))
        self.sys_opts = self.parse_optgeo_options(self.optgeo_options)
        ## Build keyword dictionaries to pass to engine.
        engine_args = OrderedDict(list(self.OptionDict.items()) + list(options.items()))
        del engine_args['name']
        ## Create engine objects.
        self.engines = OrderedDict()
        for sysname, sysopt in self.sys_opts.items():
            if self.engine_.__name__ == 'OpenMM':
                # path to pdb file
                pdbpath = os.path.join(self.root, self.tgtdir, sysopt['topology'])
                # use the PDB file with topology
                M = Molecule(pdbpath)
                # replace geometry with values from xyz file for higher presision
                M0 = Molecule(os.path.join(self.root, self.tgtdir, sysopt['geometry']))
                M.xyzs = M0.xyzs
                # here mol=M is given for the purpose of using the topology from the input pdb file
                # if we don't do this, pdb=top.pdb option will only copy some basic information but not the topology into OpenMM.mol (openmmio.py line 615)
                self.engines[sysname] = self.engine_(target=self, mol=M, name=sysname, pdb=pdbpath, **engine_args)
            elif self.engine_.__name__ == 'SMIRNOFF':
                # SMIRNOFF is a subclass of OpenMM engine but it requires the mol2 input
                # note: OpenMM.mol is a Molecule class instance;  mol2 is a file format.
                # path to .pdb file
                pdbpath = os.path.join(self.root, self.tgtdir, sysopt['topology'])
                # a list of paths to .mol2 files
                mol2path = [os.path.join(self.root, self.tgtdir, f) for f in sysopt['mol2']]
                # use the PDB file with topology
                M = Molecule(os.path.join(self.root, self.tgtdir, sysopt['topology']))
                # replace geometry with values from xyz file for higher presision
                M0 = Molecule(os.path.join(self.root, self.tgtdir, sysopt['geometry']))
                M.xyzs = M0.xyzs
                # here mol=M is given for the purpose of using the topology from the input pdb file
                # if we don't do this, pdb=top.pdb option will only copy some basic information but not the topology into OpenMM.mol (openmmio.py line 615)
                self.engines[sysname] = self.engine_(target=self, mol=M, name=sysname, pdb=pdbpath, mol2=mol2path, **engine_args)
        ## Create internal coordinates
        self._build_internal_coordinates()

    def parse_optgeo_options(self, filename):
        """ Parse an optgeo_options.txt file into specific OptGeoTarget Target Options"""
        logger.info("Reading interactions from file: %s\n" % filename)
        global_opts = OrderedDict()
        sys_opts = OrderedDict()
        section = None
        section_opts = OrderedDict()
        with open(filename) as f:
            for ln, line in enumerate(f, 1):
                # Anything after "#" is a comment
                line = line.split("#", maxsplit=1)[0].strip()
                if not line: continue
                ls = line.split()
                key = ls[0].lower()
                if key[0] == "$":
                    # section sign $
                    if key == '$end':
                        if section is None:
                            warn_press_key("Line %i: Encountered $end before any section." % ln)
                        elif section == 'global':
                            # global options read finish
                            global_opts = section_opts
                        elif section == 'system':
                            # check if system section contains name
                            if 'name' not in section_opts:
                                warn_press_key("Line %i: You need to specify a name for the system section ending." % ln)
                            elif section_opts['name'] in sys_opts:
                                warn_press_key("Line %i: A system named %s already exists in Systems" % (ln, section_opts['name']))
                            else:
                                sys_opts[section_opts['name']] = section_opts
                        section = None
                        section_opts = OrderedDict()
                    else:
                        if section is not None:
                            warn_press_key("Line %i: Encountered section start %s before previous section $end." % (ln, key))
                        if key == '$global':
                            section = 'global'
                        elif key == '$system':
                            section = 'system'
                        else:
                            warn_press_key("Line %i: Encountered unsupported section name %s " % (ln, key))
                else:
                    # put normal key-value options into section_opts
                    if key in ['name', 'geometry', 'topology']:
                        if len(ls) != 2:
                            warn_press_key("Line %i: one value expected for key %s" % (ln, key))
                        if section == 'global':
                            warn_press_key("Line %i: key %s should not appear in $global section" % (ln, key))
                        section_opts[key] = ls[1]
                    elif key in ['bond_denom', 'angle_denom', 'dihedral_denom', 'improper_denom']:
                        if len(ls) != 2:
                            warn_press_key("Line %i: one value expected for key %s" % (ln, key))
                        section_opts[key] = float(ls[1])
                    elif key == 'mol2':
                        # special parsing for mol2 option for SMIRNOFF engine
                        # the value is a list of filenames
                        section_opts[key] = ls[1:]
            # apply a few default global options
            global_opts.setdefault('bond_denom', 0.1)
            global_opts.setdefault('angle_denom', 0.1)
            global_opts.setdefault('dihedral_denom', 0.1)
            global_opts.setdefault('improper_denom', 0.1)
            # copy global options into each system
            for sys_name, sys_opt_dict in sys_opts.items():
                for k,v in global_opts.items():
                    # do not overwrite system options
                    sys_opt_dict.setdefault(k, v)
                for k in ['name', 'geometry', 'topology']:
                    if k not in sys_opt_dict:
                        warn_press_key("key %s missing in system section named %s" %(k, sys_name))
            return sys_opts

    def _build_internal_coordinates(self):
        "Build internal coordinates system with geometric.internal.PrimitiveInternalCoordinates"
        # geometric module is imported to build internal coordinates
        # importing here will avoid import error for calculations not using this target
        from geometric.internal import PrimitiveInternalCoordinates, Distance, Angle, Dihedral, OutOfPlane
        self.internal_coordinates = OrderedDict()
        for sysname, sysopt in self.sys_opts.items():
            geofile = os.path.join(self.root, self.tgtdir, sysopt['geometry'])
            topfile = os.path.join(self.root, self.tgtdir, sysopt['topology'])
            logger.info("Building internal coordinates from file: %s\n" % topfile)
            m0 = Molecule(geofile)
            m = Molecule(topfile)
            p_IC = PrimitiveInternalCoordinates(m)
            # here we explicitly pick the bonds, angles and dihedrals to evaluate
            ic_bonds, ic_angles, ic_dihedrals, ic_impropers = [], [], [], []
            for ic in p_IC.Internals:
                if isinstance(ic, Distance):
                    ic_bonds.append(ic)
                elif isinstance(ic, Angle):
                    ic_angles.append(ic)
                elif isinstance(ic, Dihedral):
                    ic_dihedrals.append(ic)
                elif isinstance(ic, OutOfPlane):
                    ic_impropers.append(ic)
            # compute and store reference values
            pos_ref = m0.xyzs[0]
            vref_bonds = np.array([ic.value(pos_ref) for ic in ic_bonds])
            vref_angles = np.array([ic.value(pos_ref) for ic in ic_angles])
            vref_dihedrals = np.array([ic.value(pos_ref) for ic in ic_dihedrals])
            vref_impropers = np.array([ic.value(pos_ref) for ic in ic_impropers])
            self.internal_coordinates[sysname] = {
                'ic_bonds': ic_bonds,
                'ic_angles': ic_angles,
                'ic_dihedrals': ic_dihedrals,
                'ic_impropers': ic_impropers,
                'vref_bonds': vref_bonds,
                'vref_angles': vref_angles,
                'vref_dihedrals': vref_dihedrals,
                'vref_impropers': vref_impropers,
            }

    def system_driver(self, sysname):
        """ Run calculation for one system, return internal coordinate values after optimization """
        engine = self.engines[sysname]
        ic_dict = self.internal_coordinates[sysname]
        if engine.__class__.__name__ in ('OpenMM', 'SMIRNOFF'):
            # OpenMM.optimize() by default resets geometry to initial geometry before optimization
            engine.optimize()
            pos = engine.getContextPosition()
        else:
            raise NotImplementedError("system_driver() not implemented for %s" % engine.__name__)
        v_ic = {
            'bonds': np.array([ic.value(pos) for ic in ic_dict['ic_bonds']]),
            'angles': np.array([ic.value(pos) for ic in ic_dict['ic_angles']]),
            'dihedrals': np.array([ic.value(pos) for ic in ic_dict['ic_dihedrals']]),
            'impropers': np.array([ic.value(pos) for ic in ic_dict['ic_impropers']]),
        }
        return v_ic

    def indicate(self):
        title_str = "Optimized Geometries, Objective = % .5e" % self.objective
        #QYD: This title is carefully placed to align correctly
        column_head_str1 =  " %-20s %13s     %13s     %15s   %15s   %17s " % ("System", "Bonds", "Angles", "Dihedrals", "Impropers", "Term.")
        column_head_str2 =  " %-20s %9s %7s %9s %7s %9s %7s %9s %7s %17s " % ('', 'RMSD', 'denom', 'RMSD', 'denom', 'RMSD', 'denom', 'RMSD', 'denom', '')
        printcool_dictionary(self.PrintDict,title=title_str + '\n' + column_head_str1 + '\n' + column_head_str2, center=[True,False,False])

    def get(self, mvals, AGrad=False, AHess=False):
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}
        self.PrintDict = OrderedDict()
        def compute(mvals):
            ''' Compute total objective value for each system '''
            self.FF.make(mvals)
            v_obj_list = []
            for sysname, sysopt in self.sys_opts.items():
                # read denominators from system options
                bond_denom = sysopt['bond_denom']
                angle_denom = sysopt['angle_denom']
                dihedral_denom = sysopt['dihedral_denom']
                improper_denom = sysopt['improper_denom']
                # inverse demon to be scaling factors, 0 for denom 0
                scale_bond = 1.0 / bond_denom if bond_denom != 0 else 0.0
                scale_angle = 1.0 / angle_denom if angle_denom != 0 else 0.0
                scale_dihedral = 1.0 / dihedral_denom if dihedral_denom != 0 else 0.0
                scale_improper = 1.0 / improper_denom if improper_denom != 0 else 0.0
                # calculate new internal coordinates f
                v_ic = self.system_driver(sysname)
                # objective contribution from bonds
                vref_bonds = self.internal_coordinates[sysname]['vref_bonds']
                vtar_bonds = v_ic['bonds']
                rmsd_bond = np.sqrt(np.sum((vref_bonds - vtar_bonds)**2)/len(vref_bonds))
                # objective contribution from angles
                vref_angles = self.internal_coordinates[sysname]['vref_angles']
                vtar_angles = v_ic['angles']
                rmsd_angle = np.sqrt(np.sum(periodic_diff(vref_angles, vtar_angles)**2)/len(vref_angles))
                # objective contribution from dihedrals
                vref_dihedrals = self.internal_coordinates[sysname]['vref_dihedrals']
                vtar_dihedrals = v_ic['dihedrals']
                rmsd_dihedral = np.sqrt(np.sum(periodic_diff(vref_dihedrals, vtar_dihedrals)**2)/len(vref_dihedrals))
                # objective contribution from improper dihedrals
                vref_impropers = self.internal_coordinates[sysname]['vref_impropers']
                vtar_impropers = v_ic['impropers']
                rmsd_improper = np.sqrt(np.sum(periodic_diff(vref_impropers, vtar_impropers)**2)/len(vref_impropers))
                # add total objective value to result list
                obj_total = rmsd_bond * scale_bond + rmsd_angle * scale_angle + \
                    rmsd_dihedral * scale_dihedral + rmsd_improper * scale_improper
                v_obj_list.append(obj_total)
                # save print string
                if not in_fd():
                    self.PrintDict[sysname] = "% 9.3f % 7.2f % 9.3f % 7.2f % 9.3f % 7.2f % 9.3f % 7.2f %17.3f" % (rmsd_bond, \
                        bond_denom, rmsd_angle, angle_denom, rmsd_dihedral, dihedral_denom, rmsd_improper, improper_denom, obj_total)
            return np.array(v_obj_list, dtype=float)
        V = compute(mvals)
        dV = np.zeros((self.FF.np,len(V)))
        if AGrad or AHess:
            for p in self.pgrad:
                dV[p,:], _ = f12d3p(fdwrap(compute, mvals, p), h = self.h, f0 = V)
        Answer['X'] = np.dot(V,V)
        for p in self.pgrad:
            Answer['G'][p] = 2*np.dot(V, dV[p,:])
            for q in self.pgrad:
                Answer['H'][p,q] = 2*np.dot(dV[p,:], dV[q,:])
        if not in_fd():
            self.objective = Answer['X']
            self.FF.make(mvals)
        return Answer
