#!/usr/bin/env python
import numpy as np
import mdtraj as md
from forcebalance.nifty import _exec, queue_up_src_dest, wq_wait, printcool
from forcebalance.molecule import *
from forcebalance.gmxio import *
from forcebalance.openmmio import *
import os, shutil

class MMTask:
    def __init__(self, md_engine_opts, wq, coord_set, top_set, ffname, remote_md):
        self.md_engine_opts = md_engine_opts
        self.wq = wq
        self.coord_set = coord_set
        self.top_set = top_set
        self.ffname = ffname
        self.remote_md = remote_md

    def cluster(self, fnm):
        ext = fnm.split('.')[1]
        _exec('find . -name {0} | sort | xargs cat > mmopt.{1}'.format(fnm, ext))
        traj = md.load('mmopt.{0}'.format(ext))
        M = Molecule('mmopt.{0}'.format(ext))
        return traj, M

class GMXTask(MMTask):
    def __init__(self, md_engine_opts, wq, coord_set, top_set, ffname, remote_md):
        #Get the work_queue and coordinates from the main Reopt_MM class
        super().__init__(md_engine_opts, wq, coord_set, top_set, ffname, remote_md)

    def minGrids(self, res, cwd, min_file):
        #Use specific GROMACS routines for grid minimization.
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
        #Cluster structures to provide back to the main code.
        traj, M = super().cluster(fnm)
        return traj, M

    def clusterMinMM(self, pdbfnm, cwd, cdnm, min_file, res):
        #Use GROMACS utilities to minimize clustered structures.
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
        #Use GROMACS routines to get the energy for a single molecule.
        self.md_engine_opts['mol'] = Molecule(min_file)
        self.md_engine_opts['gmx_top'] = 'topol.top'
        md = GMX(**self.md_engine_opts)
        energy = md.energy_one(shot=0)
        return energy

class OpenMMTask(MMTask):
    def __init__(self, md_engine_opts, wq, coord_set, top_set, ffname, remote_md):
        #Get the work_queue and coordinates from the main Reopt_MM class
        super().__init__(md_engine_opts, wq, coord_set, top_set, ffname, remote_md)

    def minGrids(self, res, cwd, min_file):
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
        #Cluster structures to provide back to the main code.
        traj, M = super().cluster(fnm)
        return traj, M

    def clusterMinMM(self, pdbfnm, cwd, cdnm, min_file, res):
        #Use GROMACS utilities to minimize clustered structures.
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
        #Use GROMACS routines to get the energy for a single molecule.
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
    def __init__(self, fbdir, qm_method, basis, cbs, grad, nt, mem):
        self.fbdir = fbdir
        self.qm_method = qm_method
        self.basis = basis
        self.cbs = cbs
        self.grad = grad
        self.nt = nt
        self.mem = mem

def isfloat(num):
    try: 
        float(num)
        return True
    except ValueError:
        return False

def getChargesMult(target_list, charges_input, mult_input):
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
    def __init__(self, fbdir, qm_method, basis, cbs, grad, nt, mem):
        super().__init__(fbdir, qm_method, basis, cbs, grad, nt, mem)
        loc = shutil.which("psi4")
        if loc is None:
            raise Exception("The psi4 executable has not been found. Make sure you have it installed in your path.")

    def writeEnergy(self, mol, fnm, charge, mult):
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
        found_energy = False
        energy = None
        for line in reversed(open(fnm).readlines()):
            ls = line.split()
            if "Total Energy" in line or "total energy" in line or "Total energy" in line:
                energy = float(ls[-1])
                break
        return energy

    def readCBSenergy(self, fnm):
        energy = None
        for line in reversed(open(fnm).readlines()):
            ls = line.split()
            if len(ls)==3:
                if ls[0]=="total" and ls[1]=="CBS":
                    energy = float(ls[2])
                    break
        return energy

    def readGrad(self, fnm):
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

