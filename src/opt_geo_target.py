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
                # use the PDB file with topology
                # we explicitly do this because Openmm(pdb=file) does not copy topology into self.mol (openmmio.py line 615)
                M = Molecule(os.path.join(self.root, self.tgtdir, sysopt['topology']))
                # replace geometry with values from xyz file for higher presision
                M0 = Molecule(os.path.join(self.root, self.tgtdir, sysopt['geometry']))
                M.xyzs = M0.xyzs
                self.engines[sysname] = self.engine_(target=self, mol=M, name=sysname, pdb=os.path.join(self.root, self.tgtdir, sysopt['topology']), **engine_args)
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
                    elif key in ['bond_denom', 'angle_denom', 'dihedral_denom']:
                        if len(ls) != 2:
                            warn_press_key("Line %i: one value expected for key %s" % (ln, key))
                        section_opts[key] = float(ls[1])
            # apply a few default global options
            global_opts.setdefault('bond_denom', 0.01)
            global_opts.setdefault('angle_denom', 1.0)
            global_opts.setdefault('dihedral_denom', 1.0)
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
        from geometric.internal import PrimitiveInternalCoordinates, Distance, Angle, Dihedral
        self.internal_coordinates = OrderedDict()
        for sysname, sysopt in self.sys_opts:
            geofile = sysopt['geometry']
            topfile = sysopt['topology']
            logger.info("Building internal coordinates from file: %s\n" % topfile)
            m0 = Molecule(geofile)
            m = Molecule(topfile)
            p_IC = PrimitiveInternalCoordinates(m)
            # here we explicitly pick the bonds, angles and dihedrals to evaluate
            ic_bonds, ic_angles, ic_dihedrals = [], [], []
            for ic in p_IC.Internals:
                if isinstance(ic, Distance):
                    ic_bonds.append(ic)
                elif isinstance(ic, Angle):
                    ic_angles.append(ic)
                elif isinstance(ic, Dihedral):
                    ic_dihedrals.append(ic)
            # compute and store reference values
            pos_ref = m0.xyzs[0]
            vref_bonds = np.array([ic.value(pos_ref) for ic in ic_bonds])
            vref_angles = np.array([ic.value(pos_ref) for ic in ic_angles])
            vref_dihedrals = np.array([ic.value(pos_ref) for ic in ic_dihedrals])
            self.internal_coordinates[sysname] = {
                'ic_bonds': ic_bonds,
                'ic_angles': ic_angles,
                'ic_dihedrals': ic_dihedrals,
                'vref_bonds': vref_bonds,
                'vref_angles': vref_angles,
                'vref_dihedrals': vref_dihedrals,
            }

    def system_driver(self, sysname):
        """ Run calculation for one system, return internal coordinate values after optimization """
        engine = self.engines[sysname]
        ic_dict = self.internal_coordinates[sysname]
        v_ic = {}
        if engine.__name__ == 'OpenMM':
            # OpenMM.optimize() by default resets geometry to initial geometry before optimization
            engine.optimize()
            pos = engine.getContextPosition()
            v_ic['bonds'] = np.array([ic.value(pos) for ic in ic_dict['ic_bonds']])
            v_ic['angles'] = np.array([ic.value(pos) for ic in ic_dict['ic_angles']])
            v_ic['dihedrals'] = np.array([ic.value(pos) for ic in ic_dict['dihedrals']])
        return v_ic

    def indicate(self):
        title_str = "Optimized Geometries, Objective = % .5e" % self.objective
        column_head_str =  " %-15s %11s %11s %11s %11s %11s %11s %11s" % ('System', 'RMSD_bond',
            'denom_bond', 'RMSD_angle', 'denom_angle', 'RMSD_dihedral', 'denom_dihedral', 'Term.')
        printcool_dictionary(self.PrintDict,title=title_str+'\n'+column_head_str)

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
                # calculate new internal coordinates f
                v_ic = self.system_driver(sysname)
                # objective contribution from bonds
                vref_bonds = self.internal_coordinates[sysname]['vref_bonds']
                vtar_bonds = v_ic['bonds']
                rmsd_bond = np.sqrt(np.sum((vref_bonds - vtar_bonds)**2))
                # objective contribution from angles
                vref_angles = self.internal_coordinates[sysname]['vref_angles']
                vtar_angles = v_ic['angles']
                rmsd_angle = np.sqrt(np.sum((vref_angles - vtar_angles)**2))
                # objective contribution from dihedrals
                vref_dihedrals = self.internal_coordinates[sysname]['vref_dihedrals']
                vtar_dihedrals = v_ic['dihedrals']
                rmsd_dihedral = np.sqrt(np.sum((vref_dihedrals - vtar_dihedrals)**2))
                # add total objective value to result list
                obj_total = rmsd_bond / bond_denom + rmsd_angle / angle_denom + rmsd_dihedral / dihedral_denom
                v_obj_list.append(obj_total)
                # save print string
                if not in_fd():
                    self.PrintDict[sysname] = "% 7.3f % 5.2f % 7.3f % 5.2f % 7.3f % 5.2f %9.3f" % (rmsd_bond,
                        bond_denom, rmsd_angle, angle_denom, rmsd_dihedral, dihedral_denom, obj_total)
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