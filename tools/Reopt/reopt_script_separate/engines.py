#!/usr/bin/env python
import numpy as np
import mdtraj as md
from forcebalance.nifty import _exec, queue_up_src_dest, wq_wait, printcool
from forcebalance.molecule import *
from forcebalance.gmxio import *
from forcebalance.openmmio import *
import os, shutil

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

    def minGrids(self, res, cwd, min_file):
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
                if not os.path.exists('topol.top'): os.symlink('../topol.top', './topol.top')
                self.md_engine_opts['mol'] = self.coord_set[res][i]
                self.md_engine_opts['gmx_top'] = 'topol.top'
                engine = GMX(**self.md_engine_opts)
                if self.remote_md is None:
                    engine.optimize(0,**self.md_engine_opts)
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

    def clusterMinMM(self, pdbfnm, cwd, cdnm, min_file, res):
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
            engine = GMX(**self.md_engine_opts)
            if self.remote_md is None:
                engine.optimize(0, **self.md_engine_opts)
            else:
                if self.wq is None:
                    raise RuntimeError("Supply a work queue port for the code if you want to use the remote script functionality.")
                if not os.path.exists("min.mdp"):
                    min_opts = {"integrator" : "l-bfgs", "emtol" : 1e-4, "nstxout" : 0, "nstfout" : 0, "nsteps" : 10000, "nstenergy" : 1}
                    edit_mdp(fin="gmx.mdp", fout="min.mdp", options=min_opts)

                queue_up_src_dest(self.wq, command="sh rungmx.sh &> min.log", tag=cdnm,
                input_files = ([(os.path.join(os.getcwd(), i), i) for i in [self.ffname.split('/')[-1], 'conf.gro', 'topol.top', 'min.mdp']] +
                [(os.path.join(cwd, 'rungmx.sh'), 'rungmx.sh')]),
                output_files = [(os.path.join(os.getcwd(), 'min.log'), 'min.log'),
                                (os.path.join(os.getcwd(), min_file), min_file)])

    def clusterSinglepointMM(self, min_file, res):
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

    def minGrids(self, res, cwd, min_file):
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
        for i in range(len(self.coord_set[res])):
            print("\rRunning Frame {0}/{1}\r".format(i+1, len(self.coord_set[res])))
            idir = os.path.join(res, "{0:04d}".format(i))
            if not os.path.exists(idir): os.makedirs(idir)
            os.chdir(idir)
            if not os.path.exists(min_file):
                self.coord_set[res][i].write("conf.pdb")
                if not os.path.exists(self.ffname.split('/')[-1]): os.symlink('{0}/{1}'.format(cwd,self.ffname), './{0}'.format(self.ffname.split('/')[-1]))
                if not os.path.exists(self.top_set[res].split('/')[-1]): os.symlink('{0}/{1}'.format(cwd,self.top_set[res]), self.top_set[res].split('/')[-1])
                self.md_engine_opts['mol'] = self.coord_set[res][i]
                self.md_engine_opts['pdb'] = self.top_set[res].split('/')[-1]
                self.md_engine_opts['ffxml'] = self.ffname.split('/')[-1]
                self.md_engine_opts['precision'] = 'double'
                self.md_engine_opts['platform'] = 'Reference'
                engine = OpenMM(**self.md_engine_opts)
                engine.optimize(0)
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

    def clusterMinMM(self, pdbfnm, cwd, cdnm, min_file, res):
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
    
    def clusterSinglepointMM(self, min_file, res):
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


"""
This script handles the QM routines of the code.
Any new QM engines that you would like to add can 
follow the template of the Psi4 class below.
"""

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
        for all cases.
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

