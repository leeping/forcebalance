#!/usr/bin/env python
import os, sys, shutil
from forcebalance.molecule import *
from forcebalance.gmxio import *
from forcebalance.openmmio import *
from forcebalance.nifty import _exec, queue_up_src_dest, wq_wait, printcool
import mdtraj as md
import numpy as np
import random as rd
from scipy.cluster.hierarchy import linkage, fcluster
import work_queue
from collections import OrderedDict
from matplotlib import pyplot as plt
import argparse

def makeWorkQueue(wq_port):
    """
    Make a Work Queue object.

    Parameter
    ---------
    wq_port: Integer 
        integer for the WQ port

    Returns 
    ---------
    wq : Work Queue object
        object for starting multiple workers
    """

    work_queue.set_debug_flag('all')
    wq = work_queue.WorkQueue(port=wq_port, exclusive=False, shutdown=False)
    wq.specify_keepalive_interval(8640000)
    wq.specify_name('reopt')
    print('Work Queue listening on {0}'.format(wq.port))
    return wq

def getClusterIndices(index_file, res_list):
    """
    A file with indices for the RMSD can be provided. This function
    looks through this file and returns the indices for each residue.

    Parameters
    ---------
    index_file : string
        Filename of the index file containing the atom indices
    res_list : list
        List containing residue names

    Returns 
    ---------
    indices : dictionary
        Dictionary containing the atom indices for each residue
    """

    indices = {}
    for line in open(index_file):
        if line.split('\n')[0].strip() in res_list:
            res = line.split('\n')[0].strip()
            indices[res] = []
        if line.strip().split('\n')[0].isdigit():
            indices[res].append(int(line.strip().split('\n')[0]))
    return indices

def isfloat(num):
    """
    Check whether input is a float or not

    Parameters
    ----------
    num : Object
        Can in principle be any object, as it is being tested for it being a float or not

    Returns
    -------
    bool
        Returns True or False depending upon whether num is a float or not
    """

    try:
        float(num)
        return True
    except ValueError:
        return False

def getChargesMult(target_list, charges_input, mult_input):
    """
    Get the charge and multiplicty for each residue for supplying to the QM program
    If the user has supplied nonzero charges or multiplicites, then get them from
    their input list. Otherwise, assign the charge and multiplicty to be zero.

    Parameters
    ----------
    target_list : list
        List of target names
    charges_input : list or NoneType
        List of residues followed by charge if input to the program, or None otherwise
    mult_input : list or NoneType
        List of multiplicities followed by charge if input to the program, or None otherwise

    Returns
    -------
    charges : dictionary
        Dictionary of residues and their charge
    mult : dictionary
        Dictionary of residues and their charge
    """

    charges = {}
    mult = {}
    for key in target_list:
        if '-' in key: res = key.split('-')[0]
        elif '_' in key: res = key.split('_')[0]
        else: res = key
        if not any(res in i for i in charges):
            if charges_input is not None:
                if res in charges_input:
                    ind = charges_input.index(res)
                    charges[res] = charges_input[ind+1]
            else: charges[res] = 0
        if not any(res in i for i in mult):
            if mult_input is not None:
                if res in mult_input:
                    ind = mult_input.index(res)
                    mult[res] = mult_input[ind+1]
            else: mult[res] = 1
    return charges, mult

class MMTask:
    """
    Template class for initializing the MM engines
    
    Parameters
    ----------
    md_engine_opts : dictionary
        Contains options and values for running the MM code
    wq : Object
        WorkQueue object
    coord_set : dictionary
        Contains a Molecule object for each residue
    top_set : dictionary
        Contains the topology file for each residue
    ffname : string
        Location of the actual force field
    remote_md : string
        Remote script used for running the MM optimizations with Work Queue
    """

    def __init__(self, md_engine_opts, wq, coord_set, top_set, ffname, remote_md):
        self.md_engine_opts = md_engine_opts
        self.wq = wq
        self.coord_set = coord_set
        self.top_set = top_set
        self.ffname = ffname
        self.remote_md = remote_md

    def cluster(self, fnm):
        """
        Cluster together the different output trajectories
        
        Parameter
        ---------
        fnm : string
            filename of the minimized coordinate files
        
        Returns
        -------
        traj : MDTraj Trajectory object
            returns a Trajectory object for use in the clustering part of the code
        M : Molecule object
            A Molecule object is returned for other uses besides the clustering
        """

        ext = fnm.split('.')[1]
        _exec('find . -name {0} | sort | xargs cat > mmopt.{1}'.format(fnm, ext))
        traj = md.load('mmopt.{0}'.format(ext))
        M = Molecule('mmopt.{0}'.format(ext))
        return traj, M

class GMXTask(MMTask):
    """
    GROMACS specific MM routines
    """
    def __init__(self, md_engine_opts, wq, coord_set, top_set, ffname, remote_md):
        """
        Parameters
        ----------
        md_engine_opts : dictionary
            Contains options and values for running the MM code
        wq : Object
            WorkQueue object
        coord_set : dictionary
            Contains a Molecule object for each residue
        top_set : dictionary
            Contains the topology file for each residue
        ffname : string
            Location of the actual force field
        remote_md : string
            Remote script used for running the MM optimizations with Work Queue
        """

        super().__init__(md_engine_opts, wq, coord_set, top_set, ffname, remote_md)

    def min_grids(self, res, cwd, min_file):
        """
        Perform the first round of MM optimizations from the initial FB target structures
        
        Parameters
        ----------
        res : string 
            Name of the specific residue
        cwd : string
            current directory path
        min_file : string
            Name of the output file containing the minimized coordinates
        """

        with open('topol.top', 'w') as f:
            f.write('#include "{0}"\n'.format(self.ffname.split('/')[-1]))
            for line in open("../../{0}".format(self.top_set[res])):
                f.write(line)
        self.md_engine_opts['mol'] = self.coord_set[res]
        self.md_engine_opts['gmx_top'] = 'topol.top'
        if not os.path.exists(self.ffname.split('/')[-1]): os.symlink('{0}/{1}'.format(cwd,self.ffname), './{0}'.format(self.ffname.split('/')[-1]))
        engine = GMX(**self.md_engine_opts)
        os.chdir('../')
        result_dir = os.getcwd()
        for i in range(len(self.coord_set[res])):
            print("\rRunning Frame {0}/{1}\r".format(i+1, len(self.coord_set[res])))
            idir = os.path.join(res, "{0:04d}".format(i))
            if not os.path.exists(idir): os.makedirs(idir)
            os.chdir(idir)
            if not os.path.exists(min_file):
                self.coord_set[res][i].write("conf.gro")
                if not os.path.exists(self.ffname.split('/')[-1]): os.symlink('{0}/{1}'.format(cwd,self.ffname), './{0}'.format(self.ffname.split('/')[-1]))
                if not os.path.exists('gmx.top'): os.symlink('../gmx.top', './gmx.top')
                if not os.path.exists('gmx.mdp'): os.symlink('../gmx.mdp', './gmx.mdp')
                if self.remote_md is None:
                    engine.optimize(i, **self.md_engine_opts)
                else:
                    if self.wq is None:
                        raise Exception("Supply a work queue port for the code if you want to use the remote script functionality.")
                    if not os.path.exists("min.mdp"):
                        min_opts = {"integrator" : "l-bfgs", "emtol" : 1e-4, "nstxout" : 0, "nstfout" : 0, "nsteps" : 10000, "nstenergy" : 1}
                        edit_mdp(fin="gmx.mdp", fout="min.mdp", options=min_opts)
                    queue_up_src_dest(self.wq, command="sh {0} &> min.log".format(self.remote_md), tag=idir,
                    input_files = ([(os.path.join(os.getcwd(), i), i) for i in [self.ffname.split('/')[-1], 'conf.gro', 'topol.top', 'min.mdp']] +
                                   [(os.path.join(cwd, self.remote_md), self.remote_md)]),
                    output_files = [(os.path.join(os.getcwd(), 'min.log'), 'min.log'),
                                   (os.path.join(os.getcwd(), min_file), min_file)])

            os.chdir(result_dir)

    def cluster(self, fnm):
        """
        Assemble the files containing the optimized coordinates into one file
        This calls a subroutine in the base class.

        Parameter
        ---------
        fnm : string
            Contains the file name that is being collected

        Returns
        -------
        traj : MDTraj Trajectory object
            returns a Trajectory object for use in the clustering part of the code
        M : Molecule object
            A Molecule object is returned for other uses besides the clustering
        """

        return super().cluster(fnm)

    def cluster_min_MM(self, pdbfnm, cwd, cmdn, min_file, res):
        """
        Perform MM optimizations on the selected clusters

        Parameters
        ----------
        pdbfnm : string
            Name of pdb file being input
        cwd : string
            current directory path
        cdnm : string
            current job being run
        min_file : string
            Name of the output file containing the minimized coordinates
        res : string
            Name of the residue that is currently being optimized
        """

        if not os.path.exists(min_file):
            coord = Molecule(os.path.join('../../Clusters', pdbfnm))
            coord[0].write("conf.gro")
            if not os.path.exists(self.ffname.split('/')[-1]): os.symlink('{0}/{1}'.format(cwd,self.ffname), './{0}'.format(self.ffname.split('/')[-1]))
            if not os.path.exists('topol.top'): os.symlink('../../topol.top', './topol.top')
            self.md_engine_opts['mol'] = coord[0]
            self.md_engine_opts['gmx_top'] = 'topol.top'
            if self.remote_md is None:
                engine = GMX(**self.md_engine_opts)
                engine.optimize(**self.md_engine_opts)
            else:
                if self.wq is None:
                    raise RuntimeError("Supply a work queue port for the code if you want to use the remote script functionality.")
                if not os.path.exists("min.mdp"):
                    min_opts = {"integrator" : "l-bfgs", "emtol" : 1e-4, "nstxout" : 0, "nstfout" : 0, "nsteps" : 10000, "nstenergy" : 1}
                    edit_mdp(fin="gmx.mdp", fout="min.mdp", options=min_opts)
                queue_up_src_dest(self.wq, command="sh {0} &> min.log".format(self.remote_md), tag=cdnm,
                input_files = ([(os.path.join(os.getcwd(), i), i) for i in [self.ffname.split('/')[-1], 'conf.gro', 'topol.top', 'min.mdp']] +
                [(os.path.join(cwd, self.remote_md), self.remote_md)]),
                output_files = [(os.path.join(os.getcwd(), 'min.log'), 'min.log'),
                (os.path.join(os.getcwd(), min_file), min_file)])

    def cluster_singlepoint_MM(self, min_file, res):
        """
        Get the singlepoint energies for the structure

        Parameters
        ----------
        min_file : string
            Name of the file containing hte coordinates
        res:
            Name of the current residue

        Returns
        -------
        energy : float
            MM energy 
        """

        self.md_engine_opts['mol'] = Molecule(min_file)
        self.md_engine_opts['gmx_top'] = 'topol.top'
        md = GMX(**self.md_engine_opts)
        energy = md.energy_one(shot=0)
        return energy

class OpenMMTask(MMTask):
    """
    OpenMM specific MM routines
    """
    def __init__(self, md_engine_opts, wq, coord_set, top_set, ffname, remote_md):
        """
        Parameters
        ----------
        md_engine_opts : dictionary
            Contains options and values for running the MM code
        wq : Object
            WorkQueue object
        coord_set : dictionary
            Contains a Molecule object for each residue
        top_set : dictionary
            Contains the topology file for each residue
        ffname : string
            Location of the actual force field
        remote_md : string
            Remote script used for running the MM optimizations with Work Queue
        """

        super().__init__(md_engine_opts, wq, coord_set, top_set, ffname, remote_md)

    def min_grids(self, res, cwd, min_file):
        """
        Perform the first round of MM optimizations from the initial FB target structures

        Parameters
        ----------
        res : string
            Name of the specific residue
        cwd : string
            current directory path
        min_file : string
            Name of the output file containing the minimized coordinates
        """

        os.chdir('../')
        result_dir = os.getcwd()
        if not os.path.exists(self.ffname.split('/')[-1]): os.symlink('{0}/{1}'.format(cwd,self.ffname), './{0}'.format(self.ffname.split('/')[-1]))
        if not os.path.exists(self.top_set[res].split('/')[-1]): os.symlink('{0}/{1}'.format(cwd,self.top_set[res]), self.top_set[res].split('/')[-1])
        self.md_engine_opts['mol'] = self.coord_set[res]
        self.md_engine_opts['pdb'] = self.top_set[res].split('/')[-1]
        self.md_engine_opts['ffxml'] = self.ffname.split('/')[-1]
        self.md_engine_opts['precision'] = 'double'
        self.md_engine_opts['platform'] = 'Reference'
        engine = OpenMM(**self.md_engine_opts)
        for i in range(len(self.coord_set[res])):
            print("\rRunning Frame {0}/{1}\r".format(i+1, len(self.coord_set[res])))
            idir = os.path.join(res, "{0:04d}".format(i))
            if not os.path.exists(idir): os.makedirs(idir)
            os.chdir(idir)
            if not os.path.exists(min_file):
                self.coord_set[res][i].write("conf.pdb")
                if not os.path.exists(self.ffname.split('/')[-1]): os.symlink('{0}/{1}'.format(cwd,self.ffname), './{0}'.format(self.ffname.split('/')[-1]))
                if not os.path.exists(self.top_set[res].split('/')[-1]): os.symlink('{0}/{1}'.format(cwd,self.top_set[res]), self.top_set[res].split('/')[-1])
                engine.optimize(i)
                top = engine.pdb.topology
                pos = engine.getContextPosition(removeVirtual=True)
                PDBFile.writeFile(top, pos, open(min_file, 'w'))
            os.chdir(result_dir)

    def cluster(self, fnm):
        """
        Assemble the files containing the optimized coordinates into one file
        This calls a subroutine in the base class.

        Parameter
        ---------
        fnm : string
            Contains the file name that is being collected

        Returns
        -------
        traj : MDTraj Trajectory object
            returns a Trajectory object for use in the clustering part of the code
        M : Molecule object
            A Molecule object is returned for other uses besides the clustering
        """

        return super().cluster(fnm)

    def cluster_min_MM(self, pdbfnm, cwd, cdnm, min_file, res):
        """
        Perform MM optimizations on the selected clusters

        Parameters
        ----------
        pdbfnm : string
            Name of pdb file being input
        cwd : string
            current directory path
        cdnm : string
            current job being run
        min_file : string
            Name of the output file containing the minimized coordinates
        res : string
            Name of the residue that is currently being optimized
        """

        if not os.path.exists(min_file):
            coord = Molecule(os.path.join('../../Clusters', pdbfnm))
            coord[0].write("conf.pdb")
            if not os.path.exists(self.ffname.split('/')[-1]): os.symlink('{0}/{1}'.format(cwd,self.ffname), './{0}'.format(self.ffname.split('/')[-1]))
            if not os.path.exists(self.top_set[res].split('/')[-1]): os.symlink('{0}/{1}'.format(cwd,self.top_set[res]), self.top_set[res].split('/')[-1])
            self.md_engine_opts['mol'] = coord
            self.md_engine_opts['pdb'] = self.top_set[res].split('/')[-1]
            self.md_engine_opts['ffxml'] = self.ffname.split('/')[-1]
            self.md_engine_opts['precision'] = 'double'
            self.md_engine_opts['platform'] = 'Reference'
            engine = OpenMM(**self.md_engine_opts)
            engine.optimize(0)
            top = engine.pdb.topology
            pos = engine.getContextPosition(removeVirtual=True)
            PDBFile.writeFile(top, pos, open(min_file, 'w'))

    def cluster_singlepoint_MM(self, min_file, res):
        """
        Get the singlepoint energies for the structure

        Parameters
        ----------
        min_file : string
            Name of the file containing hte coordinates
        res:
            Name of the current residue

        Returns
        -------
        energy : float
            MM energy
        """

        self.md_engine_opts['mol'] = Molecule(min_file)
        self.md_engine_opts['pdb'] = self.top_set[res].split('/')[-1]
        self.md_engine_opts['ffxml'] = self.ffname.split('/')[-1]
        self.md_engine_opts['precision'] = 'double'
        self.md_engine_opts['platform'] = 'Reference'
        md = OpenMM(**self.md_engine_opts)
        md.update_simulation()
        energy = md.energy_one(0)
        return energy

class QMTask:
    """
    Template class for initializing the QM engines

    Parameters
    ----------
    fbdir : string
        Location of the ForceBalance directory
    qm_method : string
        Name of the QM calculation method
    basis : string
        Name of the basis set
    cbs : bool
        Whether to run cbs calculation or not
    grad : bool
        Whether to run gradient caculations
    mem : integer
        Number of gb of memory to use
    """

    def __init__(self, fbdir, qm_method, basis, cbs, grad, nt, mem):
        self.fbdir = fbdir
        self.qm_method = qm_method
        self.basis = basis
        self.cbs = cbs
        self.grad = grad
        self.nt = nt
        self.mem = mem

class Psi4Task(QMTask):
    """
    Launches tasks for Psi4

    Parameters
    ----------
    fbdir : string
        Location of the ForceBalance directory
    qm_method : string
        Name of the QM calculation method
    basis : string
        Name of the basis set
    cbs : bool
        Whether to run cbs calculation or not
    grad : bool
        Whether to run gradient caculations
    mem : integer
        Number of gb of memory to use
    """

    def __init__(self, fbdir, qm_method, basis, cbs, grad, nt, mem):
        super().__init__(fbdir, qm_method, basis, cbs, grad, nt, mem)
        loc = shutil.which("psi4")
        if loc is None:
            raise Exception("The psi4 executable has not been found. Make sure you have it installed in your path.")

    def writeEnergy(self, mol, fnm, charge, mult):
        """
        Determines whether a CBS energy calculation is being run. If not,
        write a standard Psi4 output file

        Parameters
        ----------
        mol : Molecule object
            The Molecule object for the residue being calculated
        fnm : string
            The file name to be written out
        charge : integer
            Charge for the residue
        mult : integer
            Multiplicity for the residue
        """

        if self.cbs is True:
            self.writeCBSenergy(mol, fnm, charge, mult)
        else:
            out = open(fnm, 'w')
            out.write("memory {0} gb\n\n".format(self.mem))
            out.write("molecule {\n")
            out.write("{0} {1}\n".format(charge, mult))
            for e, c in zip(mol.elem, mol.xyzs[0]):
                out.write("  {0}   {1:9.6f}   {2:9.6f}   {3:9.6f}\n".format(e, c[0], c[1], c[2]))
            out.write("units angstrom\n")
            out.write("no_reorient\n")
            out.write("}\n\n")
            out.write("set globals {\n")
            out.write("basis       {0}\n".format(self.basis))
            out.write("guess       sad\n")
            out.write("scf_type    df\n")
            out.write("puream      true\n")
            out.write("print       1\n")
            out.write("}\n\n")
            out.write("energy('{0}')".format(self.qm_method))
            out.close()

    def writeCBSenergy(self, mol, fnm, charge, mult):
        """
        Writes out a Psi4 input file for a CBS calculation using Helgaker's two-point 
        extrapolation.

        Parameters
        ----------
        mol : Molecule object
            The Molecule object for the residue being calculated
        fnm : string
            The file name to be written out
        charge : integer
            Charge for the residue
        mult : integer
            Multiplicity for the residue
        """

        out = open(fnm, 'w')
        out.write("memory {0} gb\n\n".format(self.mem))
        out.write("molecule {\n")
        out.write("{0} {1}\n".format(charge, mult))
        for e, c in zip(mol.elem, mol.xyzs[0]):
            out.write("  {0}   {1:9.6f}   {2:9.6f}   {3:9.6f}\n".format(e, c[0], c[1], c[2]))
        out.write("units angstrom\n")
        out.write("no_reorient\n")
        out.write("}\n\n")
        out.write("set globals {\n")
        out.write("basis       {0}\n".format(self.basis))
        out.write("guess       sad\n")
        out.write("scf_type    df\n")
        out.write("puream      true\n")
        out.write("print       1\n")
        out.write("}\n\n")
        out.write("energy(cbs, corl_wfn='{0}', corl_basis='aug-cc-pv[tq]z', corl_scheme=corl_xtpl_helgaker_2)".format(self.qm_method))
        out.close()

    def custom_energy(self, mol, fnm, charge, mult, custom_eng):
        """
        I'm testing a current custom energy function where users can supply a Psi4 input file.
        However, I can't guarantee that it will always work, so it's not called by any of the
        code currently.
        """

        eng_lines = []
        for line in open(custom_eng):
            eng_lines.append(line)
        out = open(fnm, 'w')
        out.write("memory {0} gb\n\n".format(self.mem))
        out.write("molecule {\n")
        out.write("{0} {1}\n".format(charge, mult))
        for e, c in zip(mol.elem, mol.xyzs[0]):
            out.write("  {0}   {1:9.6f}   {2:9.6f}   {3:9.6f}\n".format(e, c[0], c[1], c[2]))
        out.write("units angstrom\n")
        out.write("no_reorient\n")
        out.write("}\n\n")
        out.write("set globals {\n")
        out.write("basis       {0}\n".format(self.basis))
        out.write("guess       sad\n")
        out.write("scf_type    df\n")
        out.write("puream      true\n")
        out.write("print       1\n")
        out.write("}\n\n")
        for line in eng_lines:
            out.write("{0}".format(line))
        out.close()

    def writeGrad(self, mol, fnm, charge, mult):
        """
        Writes an input file for a gradient calculation.

        Parameters
        ----------
        mol : Molecule object
            The Molecule object for the residue being calculated
        fnm : string
            The file name to be written out
        charge : integer
            Charge for the residue
        mult : integer
            Multiplicity for the residue
        """

        out = open(fnm, 'w')
        out.write("memory {0} gb\n\n".format(self.mem))
        out.write("molecule {\n")
        out.write("{0} {1}\n".format(charge, mult))
        for e, c in zip(mol.elem, mol.xyzs[0]):
            out.write("  {0}   {1:9.6f}   {2:9.6f}   {3:9.6f}\n".format(e, c[0], c[1], c[2]))
        out.write("units angstrom\n")
        out.write("no_reorient\n")
        out.write("}\n\n")
        out.write("set globals {\n")
        out.write("basis       {0}\n".format(self.basis))
        out.write("guess       sad\n")
        out.write("scf_type    df\n")
        out.write("puream      true\n")
        out.write("print       1\n")
        out.write("}\n\n")
        out.write("gradient('{0}')".format(self.qm_method))
        out.close()

    def readEnergy(self, fnm):
        """
        Finds the energy in a Psi4 input file and directs to other
        methods for the specific type of calculation.

        Parameters
        ----------
        fnm : string
            Filename for Psi4 output file
        """

        found_coords = False
        energy = None
        coords = []
        elem = []
        for line in open(fnm):
            ls = line.split()
            if "molecule {" in line:
                found_coords = True
            if found_coords is True:
                if len(ls)==4:
                    elem.append(ls[0])
                    coords.append([float(s) for s in ls[1:4]])
            if "}" in line:
                found_coords = False
            #This is the standard default case with this program
            if "energy('mp2')" in line or "energy('MP2')" in line:
                energy = self.readMP2energy(fnm)
                break
            #This should match any CBS energy calls
            elif "CBS" in line:
                energy = self.readCBSenergy(fnm)
                break
            #Hopefully this gets any other energy calls that could be used
            elif "Total Energy" in line or "total energy" in line or "Total energy" in line:
                energy = self.read_std_energy(fnm)
                break
        if len(coords)==0:
            raise Exception("The coordinates in {0}/grad.out have length zero".format(os.getcwd()))
        if energy is None:
            raise Exception("Oops, the energy output file {0}/energy.out either doesn't match any of the prepared parsers or something went wrong with the calculation. Either contact John Stoppelman to fix it or write your own parser definition yourself.".format(os.getcwd()))
        return energy, elem, coords

    def readMP2energy(self, fnm):
        """
        Finds energy for a MP2 calculation.

        Parameters
        ----------
        fnm : string
            Filename for Psi4 output file
        """

        found_energy = False
        energy = None
        for line in open(fnm):
            ls = line.split()
            if "DF-MP2 Energies" in line:
                found_energy = True
            if found_energy is True and "Total Energy" in line:
                energy = float(ls[-2])
                found_energy = False
        return energy

    def read_std_energy(self, fnm):
        """
        Supposed to read any Psi4 energy output, but I'm not sure if that will work
        """

        found_energy = False
        energy = None
        for line in reversed(open(fnm).readlines()):
            ls = line.split()
            if "Total Energy" in line or "total energy" in line or "Total energy" in line:
                energy = float(ls[-1])
                break
        return energy

    def readCBSenergy(self, fnm):
        """
        Finds energy for a CBS calculation.

        Parameters
        ----------
        fnm : string
            Filename for Psi4 output file
        """

        energy = None
        for line in reversed(open(fnm).readlines()):
            ls = line.split()
            if len(ls)==3:
                if ls[0]=="total" and ls[1]=="CBS":
                    energy = float(ls[2])
                    break
        return energy

    def readGrad(self, fnm):
        """
        Finds gradient in a Psi4 output file.

        Parameters
        ----------
        fnm : string
            Filename for Psi4 output file
        """

        found_grad = False
        found_coords = False
        elem = []
        coords = []
        grad = []
        for line in open(fnm):
            ls = line.split()
            if "molecule {" in line:
                found_coords = True
            if found_coords is True:
                if len(ls)==4:
                    elem.append(ls[0])
                    coords.append(ls[1:4])
            if "}" in line:
                found_coords = False
            if "Total Gradient" in line or "Total gradient" in line or "New Matrix" in line:
                found_grad = True
            if found_grad is True:
                if len(ls)==4 and isfloat(ls[1]) and isfloat(ls[2]) and isfloat(ls[3]):
                    grad.append([float(s) for s in ls[1:4]])
        if len(coords)==0:
            raise Exception("The coordinates in {0}/grad.out have length zero".format(os.getcwd()))
        if len(grad) != len(coords):
            raise Exception("The length of the coordinates in {0}/grad.out is not equal to the length of the gradients".format(os.getcwd()))
        return np.asarray(grad)

    def runCalc(self, fnm, wq, run_script):
        """
        Run a Psi4 calculation, either with Work Queue or in serial.

        Parameters
        ----------
        fnm : string
            Filename for Psi4 output file
        wq : WorkQueue object or NoneType
            WorkQueue construct for calculations on multiple workers (recommended)
        run_script : string or NoneType
            Filename containing options for running Psi4 on clusters (optional)
        """

        pre = fnm.split('.')[0]
        if wq is not None:
            if run_script is None:
                queue_up_src_dest(wq,"psi4 {0} -n {1} &> {2}.log".format(fnm, self.nt, pre),
                input_files=[(os.path.join(os.getcwd(),fnm), fnm)],
                output_files=[(os.path.join(os.getcwd(),"{0}.out".format(pre)),"{0}.out".format(pre)),
                (os.path.join(os.getcwd(),"{0}.log".format(pre)),"{0}.log".format(pre))])
            else:
                shutil.copyfile(run_script, 'run_psi.sh')
                queue_up_src_dest(wq,"sh run_psi.sh {0} &> {1}.log".format(fnm, pre),
                input_files=[(os.path.join(os.getcwd(),fnm), fnm),
                (os.path.join(os.getcwd(), "run_psi.sh"), "run_psi.sh")],
                output_files=[(os.path.join(os.getcwd(),"{0}.out".format(pre)),"{0}.out".format(pre)),
                (os.path.join(os.getcwd(),"{0}.log".format(pre)),"{0}.log".format(pre))])
        else:
            _exec("psi4 {0} -n {1} &> {2}.log".format(fnm, self.nt, pre))


class Reopt:
    """
    This class handles the main functions of the code, including the clustering.
    The inputs are kwargs taken from the argparse input.
    """
    def __init__(self, **kwargs):
        #Get all of the user info needed for the code.

        #Need a ForceBalance directory for the targets and the MM options.
        self.fbdir = kwargs.get('fbdir')
        
        #Make work_queue object if supplied
        self.wq_port = kwargs.get('wq_port', None)
        if self.wq_port is not None: self.wq = makeWorkQueue(self.wq_port)
        else: self.wq = None
        
        #If supplied special indices for the clusters, then get the filename for this.
        #Also get the clustering algorithm and whether structures produced from a 
        #previous version of the code should be deleted.
        self.index_file = kwargs.get('custom_cluster_indices', None)
        self.redo_clusters = kwargs.get('redo_clusters')
        self.default_cluster_indices = kwargs.get('default_cluster_indices')

        #If a script has been supplied that allows for the MD jobs to be submitted one 
        #by one with work_queue, then get the filename for this.
        self.remote_md = kwargs.get('remote_md', None)

        #Get options for the QM calculations.
        self.qm_engine = kwargs.get('qm_engine')
        self.qm_method = kwargs.get('qm_method')
        self.cbs = kwargs.get('cbs')
        self.basis = kwargs.get('basis')
        self.grad = kwargs.get('grad')
        self.remote_qm = kwargs.get('remote_qm')
        self.nt = kwargs.get('nt')
        self.mem = kwargs.get('mem')
        self.chg = kwargs.get('charges')
        self.mu = kwargs.get('mult')

        if self.remote_qm is not None:
            self.remote_qm = os.path.abspath(self.remote_qm)

        #if self.custom_eng is not None:
        #    self.custom_eng = os.path.abspath(self.custom_eng)
	
        #Determine if energies will be plotted or not. They are plotted by default.
        self.plot = kwargs.get('plot', None)
        self.output_dir = kwargs.get('outputdir')

    def parseFBInput(self):
        """
        This reads through the provided ForceBalance directory and sets up
        the MD engine and options for the rest of the code. Make sure your input
        file contains the ForceBalance input file as a ".in" extension.
        """
        printcool("Reading Grids")

        self.target_list = []
        self.output_list = {}
        self.md_engine_opts = {}
        f_list = os.listdir(self.fbdir)
        infile = [i for i in f_list if ".in" in i][0]

        found_target = False
        mmopt = False
        for f in os.listdir("{0}/result".format(self.fbdir)):
            for ff in os.listdir("{0}/result/{1}".format(self.fbdir,f)):
                self.ffname = "{0}/result/{1}/{2}".format(self.fbdir,f,ff)
        
        for line in open("{0}/{1}".format(self.fbdir,infile)):
            ls = line.split()
            if "$target" in line: found_target = True
            if found_target is True:
                if "name" in line:
                    target_names = ls[1:]
                if "mmopt" in line:
                    mmopt = True
                if "type" in line:
                    if "AbInitio" in line and mmopt is False:
                        self.target_list.extend(target_names[:])
                        self.md_engine_name = line.split('_')[1]
                    else:
                        mmopt = False

        #Get engine specific keywords, although some engines don't require any.
        if "GMX" in self.md_engine_name:
            for line in open("{0}/{1}".format(self.fbdir,infile)):
                ls = line.split()
                if "gmxpath" in line: self.md_engine_opts["gmxpath"] = ls[1]
                if "gmxsuffix" in line: self.md_engine_opts["gmxsuffix"] = ls[1]
        elif "OpenMM" in self.md_engine_name:
            pdb_dict = {}
            for line in open("{0}/{1}".format(self.fbdir,infile)):
                ls = line.split()
                if "name" in line: name = ls[1]
                if "pdb" in line: pdb_dict[name] = ls[1]
        else:
            raise Exception("The MD Engine in the target {0} is currently not supported.".format(self.md_engine_name))   
        self.coord_set = {}
        self.top_set = {}
        for dnm in sorted(os.listdir("{0}/targets".format(self.fbdir))):
            if dnm in self.target_list:
                if '-' in dnm: res = dnm.split('-')[0]
                elif '_' in dnm: res = dnm.split('_')[0]
                else: res = dnm
                if res not in self.coord_set:
                    self.coord_set[res] = Molecule(os.path.join("{0}/targets".format(self.fbdir), dnm, 'all.gro'))
                    if "GMX" in self.md_engine_name: self.top_set[res] = os.path.join("{0}/targets".format(self.fbdir), dnm, 'topol.top')
                    elif "OpenMM" in self.md_engine_name: self.top_set[res] = os.path.join("{0}/targets".format(self.fbdir), dnm, pdb_dict[dnm]) 
                else: self.coord_set[res] += Molecule(os.path.join("{0}/targets".format(self.fbdir), dnm, 'all.gro'))
                self.output_list[res] = self.output_dir + "/" + res

        #The purpose for the md_engine object is to handle the specific routines for 
        #each md engine
        if "GMX" in self.md_engine_name:
            self.md_engine = GMXTask(self.md_engine_opts, self.wq, self.coord_set, self.top_set, self.ffname, self.remote_md)
            self.min_file = "gmx-min.gro"
        elif "OpenMM" in self.md_engine_name:
            self.md_engine = OpenMMTask(self.md_engine_opts, self.wq, self.coord_set, self.top_set, self.ffname, self.remote_md)
            self.min_file = "omm-min.pdb"

    def minGrids(self):
        """
        Minimizes the grid points by calling the corresponding function
        in the md_engine object.
        """
        printcool("Minimizing Grid Points")

        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        cwd = os.getcwd()
        for res in self.coord_set:
            if not os.path.exists(self.output_list[res]): os.makedirs(self.output_list[res])
            os.chdir(self.output_list[res])
            self.md_engine.min_grids(res, cwd, self.min_file)
            os.chdir(cwd)
        if self.wq is not None: wq_wait(self.wq)

    def cluster(self):
        """
        This part of the code clusters the previously MM-minimized structures
        by hybrid kmedoids or Scipy's linkage algorithm based off of the RMSD. 
        Edit the fcluster line if you need more variation in your clusters.
        """

        printcool("Clustering")

        if self.index_file is not None:
            self.indices = getClusterIndices(self.index_file, list(self.coord_set.keys()))
        else:
            self.indices = {}
        
        self.clusters = {}
        cwd = os.getcwd()
        for res in self.coord_set:
            cluster_dict = {}
            os.chdir(self.output_list[res])
            if os.path.isdir('Clusters') and self.redo_clusters is True:
                shutil.rmtree('Clusters')
                if os.path.isdir('ClusterOpt'):
                    shutil.rmtree('ClusterOpt')
            if not os.path.isdir('Clusters'):
                os.mkdir('Clusters')
                traj, M = self.md_engine.cluster(self.min_file)
                if res not in list(self.indices.keys()):
                    if self.default_cluster_indices == "heavy":
                        self.indices[res] = [a.index for a in traj.top.atoms if a.element.name != 'hydrogen']
                    elif self.default_cluster_indices == "all":
                        self.indices[res] = [a.index for a in traj.top.atoms]
                    else:
                        raise Exception("{0} is not a valid option for the default cluster indices.".format(self.default_cluster_indices))
                RMSD = md.rmsd(traj, traj, atom_indices=self.indices[res])
                cluster = linkage(np.reshape(RMSD,(len(RMSD),1)), method="centroid")
                clusters = fcluster(cluster, 0.001, "distance")
                for i in range(1,max(clusters)+1): cluster_dict[i] = []
                os.chdir('./Clusters')
                for i in range(1,max(clusters)+1): cluster_dict[i] = []
                for i in range(len(clusters)):
                    clusterNum = clusters[i]
                    cluster_dict[clusterNum].append(i)
                for cl in cluster_dict:
                    clusterTraj = M[cluster_dict[cl]]
                    clusterTraj.write("State_{0}.pdb".format(cl))
                if not res in self.clusters: self.clusters[res] = cluster_dict
            os.chdir(cwd)

    def clusterMinMM(self):
        """
        Minimize the newly formed cluster centers, again using the corresponding part of the 
        code in md_engine.
        """

        printcool("Cluster Center Minimization")
        cwd = os.getcwd()
        for res in self.coord_set:
            os.chdir(self.output_list[res])
            for pdbfnm in sorted(os.listdir('Clusters')):
                sn = int(os.path.splitext(pdbfnm)[0].split('_')[1].replace('State',''))
                cdnm = os.path.join('ClusterOpt', "{0}".format(sn))
                if not os.path.isdir(cdnm): os.makedirs(cdnm)
                os.chdir(cdnm)
                self.md_engine.cluster_min_MM(pdbfnm, cwd, cdnm, self.min_file, res)
                os.chdir('../../')
            os.chdir(cwd)
        if self.wq is not None: wq_wait(self.wq)
                
    def clusterSinglepointsMM(self):
        """
        Get the singlepoint energy of the MM-optmized clusters.

        Returns
        -------
        self.mm_energy : dictionary
            Dictionary containing the set of MM energies for each residue geometry
        """

        printcool("Cluster Center Single Points MM")
        self.mm_energy = {}
        cwd = os.getcwd()
        for res in self.coord_set:
            self.mm_energy[res] = []
            os.chdir(self.output_list[res])
            for k in os.listdir('ClusterOpt'):
                cdnm = os.path.join('ClusterOpt', "{0}".format(k))
                os.chdir(cdnm)
                energy = self.md_engine.cluster_singlepoint_MM(self.min_file, res)
                self.mm_energy[res].append(energy/4.184)
                os.chdir('../../')
            os.chdir(cwd)
        return self.mm_energy

    def clusterSinglepointsQM(self):
        """
        Get the QM single point energy
        """

        printcool("Cluster Center Single Points QM")
        if self.qm_engine=="Psi4":
            self.QM = Psi4Task(self.fbdir, self.qm_method, self.basis, self.cbs, self.grad, self.nt, self.mem)
        else:
            raise Exception("Only Psi4 is supported for QM calculations for now.")
        self.charges, self.mult = getChargesMult(self.target_list, self.chg, self.mu)
        cwd = os.getcwd()
        for res in self.coord_set:
            os.chdir(self.output_list[res])
            for cluster in os.listdir('ClusterOpt'):
                cdnm = os.path.join('ClusterOpt', "{0}".format(cluster))
                os.chdir(cdnm)
                M = Molecule(self.min_file)
                if not os.path.exists("energy.out"):
                    self.writeQMenergy(M, "energy.dat", res)
                    self.QM.runCalc("energy.dat", self.wq, self.remote_qm)
                if self.grad is True:
                    if not os.path.exists("grad.out"):
                        self.writeQMgrad(M, "grad.dat", res)
                        self.QM.runCalc("grad.dat", self.wq, self.remote_qm)
                os.chdir('../../')
            os.chdir(cwd)
        if self.wq is not None: wq_wait(self.wq)

    def readQMEng(self):
        """
        Run through the output directories and read the QM energy 
        outputs.

        Returns
        -------
        qm_energy : Dictionary
            Set of QM energies for each residue
        qm_grad : Dictionary
            Set of QM gradients for each residue
        qm_coords : Dictionary 
            Set of coordinates for each residue.
        """

        cwd = os.getcwd()
        qm_energy = {}
        qm_coords = {}
        qm_grad = {}
        for res in self.coord_set:
            os.chdir(self.output_list[res])
            qm_energy[res] = []
            qm_coords[res] = []
            if self.grad is True:
                qm_grad[res] = []
            for k in os.listdir('ClusterOpt'):
                cdnm = os.path.join('ClusterOpt', "{0}".format(k))
                os.chdir(cdnm)
                qm_eng, elem, coords = self.QM.readEnergy("energy.out")
                qm_energy[res].append(qm_eng)
                qm_coords[res].append(np.asarray(coords))
                if self.grad is True:
                    grad = self.QM.readGrad("grad.out")
                    qm_grad[res].append(grad)
                os.chdir('../../')
            os.chdir(cwd)
        return qm_energy, qm_grad, qm_coords
    
    def writeQMenergy(self, mol, fnm, res):
        """
        Write a QM energy input file.
        
        Parameters
        ----------
        mol : Molecule object
            Molecule object for the residue 
        fnm : string
            output filename
        res : string
            dictionary key for the charges and mult dictionaries
        """

        self.QM.writeEnergy(mol, fnm, self.charges[res], self.mult[res])

    def writeQMgrad(self, mol, fnm, res):
        """
        Write a QM energy input file.

        Parameters
        ----------
        mol : Molecule object
            Molecule object for the residue
        fnm : string
            output filename
        res : string
            dictionary key for the charges and mult dictionaries
        """

        self.QM.writeGrad(mol, fnm, self.charges[res], self.mult[res])

    def pltEnergies(self, mm_energy, qm_energy):
        """
        Plot the MM vs QM energies.

        Parameters
        ----------
        mm_energy : Dictionary
            Contains the set of MM energies for each residue
        qm_energy : Dictionary
            Contains the set of QM energies for each residue
        """

        if self.plot is True:
            cwd = os.getcwd()
            for res in mm_energy:
                os.chdir(self.output_list[res])

                qm = qm_energy[res]
                qm = np.asarray(qm)
                qm = qm * 627.51
                eng_min_arg = qm.argmin()
                qm -= qm[eng_min_arg]
                np.savetxt('qm_energy.txt', qm)

                mm = mm_energy[res]
                mm = np.array(mm)
                mm -= mm[eng_min_arg]
                np.savetxt('mm_energy.txt', mm)

                plt.figure(1, figsize=(6,6))
                fig, axes = plt.subplots(nrows=1, ncols=1)
                xymin = min(min(mm), min(qm))-1
                xymax = max(max(mm), max(qm))+1
                axes.set(aspect=1)
                axes.set(xlim=[xymin, xymax])
                axes.set(ylim=[xymin, xymax])
                plt.scatter(qm, mm, s=120, alpha=0.5)
                dxy = 1.0
                plt.plot(np.arange(xymin, xymax, dxy*0.01), np.arange(xymin, xymax, dxy*0.01), color='k', alpha=0.4, lw=5)
                plt.xlabel('QM energy')
                plt.ylabel('MM energy')
                plt.title('{0} QM vs. MM optimized energies'.format(res))
                plt.savefig('{0}_cluster.pdf'.format(res))
                os.chdir(cwd)
            
    def makeNewTargets(self, qm_energy, qm_coords, qm_grad):
        """
        Output a new ForceBalance "mmopt" target for each residue

        Parameters
        ----------
        qm_energy : Dictionary
            Dictionary containing the qm energies for each residue
        qm_coords : Dictionary
            Dictionary containing the coordinates for each residue
        qm_grad : Dictionary
            Dictionary containing the gradients for each residue
        """

        home = os.getcwd()
        if not os.path.isdir('{0}/new_targets'.format(self.output_dir)): os.mkdir('{0}/new_targets'.format(self.output_dir))
        os.chdir('{0}/new_targets'.format(self.output_dir))
        cwd = os.getcwd()
        for res in self.coord_set:
            if not os.path.isdir('{0}-mmopt'.format(res)): os.mkdir('{0}-mmopt'.format(res))
            os.chdir('{0}-mmopt'.format(res))
            coords = qm_coords[res]
            mol = self.coord_set[res]
            mol.xyzs = coords
            mol.qm_energies = qm_energy[res]
            if self.grad is True:
                mol.qm_grads = qm_grad[res]
            mol.comms = ["MM-optimized cluster center {0}".format(i) for i in range(len(qm_coords[res]))]
            mol.boxes = [mol.boxes[0] for i in range(len(qm_coords[res]))]
            mol.write('all.gro')
            mol.write('qdata.txt')
            if "GMX" in self.md_engine_name:
                shutil.copy('{0}/{1}'.format(home,self.top_set[res]), '.')
            elif "OpenMM" in self.md_engine_name:
                shutil.copy('{0}/{1}'.format(home,self.top_set[res]), '.')
            os.chdir(cwd)

def run_reopt(**kwargs):
    """
    Function for actually running all the code components.
    """

    reopt = Reopt(**kwargs)
    reopt.parseFBInput()
    reopt.minGrids()
    reopt.cluster()
    reopt.clusterMinMM()
    mm_energy = reopt.clusterSinglepointsMM()
    reopt.clusterSinglepointsQM()
    qm_energy, qm_grad, qm_coords = reopt.readQMEng()
    reopt.pltEnergies(mm_energy, qm_energy)
    reopt.makeNewTargets(qm_energy, qm_coords, qm_grad)

def main():
    parser = argparse.ArgumentParser(description="For use in conjunction with ForceBalance to identify spurious minima in the MM potential energy surface. The outputs from this program can then be added as ForceBalance targets for another optimization cycle. Currently works with OpenMM and GROMACS targets, with Psi4 as the QM engine.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fbdir', type=str, help="The name of the ForceBalance directory containing all of the needed data.")
    parser.add_argument('--outputdir', type=str, default='reopt_result', help="The name of the output directory for the calculation.")
    parser.add_argument('--plot', default=True, action='store_false', help="If true, will make a final plot of the QM vs. MM energies")
    parser.add_argument('--custom_cluster_indices', type=str, default=None, help="If custom indices are desired for the clustering, then provide a file with the desired indices for the target you want. An example file is provided in the ForceBalance tools directory.")
    parser.add_argument('--default_cluster_indices', type=str, default='heavy', help="The default clustering code will only pick out heavy atoms. Write 'all' if you want all indices to be used.")
    parser.add_argument('--redo_clusters', default=False, action="store_true", help="If true, then the cluster definitions you provided to the code can be changed without having to redo the whole code. Either provide a new collection of indices or use the program's default")
    parser.add_argument('--wq_port', type=int, default=None, help='Specify port number to use Work Queue to distribute optimization jobs.')
    parser.add_argument('--remote_md', type=str, default=None, help="You can specify a file that will allow for the work queue computing algorithms to run multiple MD jobs at once. An example file is included in the Examples directory.")
    parser.add_argument('--remote_qm', type=str, default=None, help="If you need to load specific environment variables for your QM program, you can submit a QM script that will load these options.")
    parser.add_argument('--qm_engine', type=str, default='Psi4', help="Enter which program you want to use for the QM calculations. Currently this only works with Psi4, but you can alter the existing Psi4 code template to match whichever program you would like to use.")
    parser.add_argument('--qm_method', type=str, default="MP2", help="Identify which QM method you would like to use for the energies and gradients. The only option currently is MP2.")
    parser.add_argument('--basis', type=str, default='aug-cc-pVTZ', help="Identify which basis set you would like to use. aug-cc-pVTZ is the default.")
    parser.add_argument('--cbs', action="store_true", default=False, help="If true, will do a cbs energy calculation.")
    parser.add_argument('--grad', default=True, action='store_false', help="If true, will use the QM program to compute the gradients on each molecule.")
    parser.add_argument('--charges', type=str, default=None, nargs='+', help="If the charges for your system is nonzero, enter the residue/system name followed by the charge. This information will be supplied to the QM calculation. The default value for the charge will be 0.")
    parser.add_argument('--mult', type=str, default=None, nargs='+', help="If the multiplicity for your system is not 1, enter the residue/system name followed by the multiplicity. This information will be supplied to the QM calculation. The default value for the multiplicity will be 1.")
    parser.add_argument('--nt', type=int, default=1, help="Specify the number of threads for the QM calculations.")
    parser.add_argument('--mem', type=int, default=4, help="Specify the memory your calculations will need. The default should generally be sufficient.")

    args = parser.parse_args()
    run_reopt(**vars(args))

if __name__ == "__main__":
    main()
