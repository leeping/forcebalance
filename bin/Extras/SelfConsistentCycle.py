#!/usr/bin/env python

""" @package GenerateQMData

Executable script for generating QM data for force, energy, electrostatic potential, and
other ab initio-based targets. """

import os, sys, re, glob
from forcebalance.forcefield import FF
from forcebalance.parser import parse_inputs
from forcebalance.nifty import *
from forcebalance.molecule import Molecule, format_xyz_coord
import numpy as np
import numpy.oldnumeric as nu
import work_queue
import random

# The default VdW radii from Chmiera, taken from AMBER.
# See http://www.cgl.ucsf.edu/chimera/1.5/docs/UsersGuide/midas/vdwtables.html
VdW99 = {'H' : 1.00, 'C' : 1.70, 'N' : 1.625, 'O' : 1.49, 'F' : 1.56, 'P' : 1.871, 'S' : 1.782, 'I' : 2.094, 'Cl' : 1.735, 'Br': 1.978}

class SCF_Simulation(object):
    def __init__(self, root, name):
        self.name = name
        self.root = os.path.join(os.path.abspath(root),name)
        if not os.path.exists(self.root):
            raise Exception("The root directory for simulation %s (%s) doesn't exist" % (self.name,self.root))
            
    def SetupGeneration(self):
        os.chdir(self.root)
        # Sanity checks
        if not os.path.exists("settings"):
            raise Exception("The settings directory doesn't exist for simulation %s - please create it and add the necessary files" % self.name)
        if options['gmxpath'] == None or options['gmxsuffix'] == None:
            warn_press_key('Please set the options gmxpath and gmxsuffix in the input file!')
        if not os.path.exists(os.path.join(options['gmxpath'],"mdrun"+options['gmxsuffix'])):
            warn_press_key('The mdrun executable pointed to by %s doesn\'t exist! (Check gmxpath and gmxsuffix)' % os.path.join(options['gmxpath'],"mdrun"+options['gmxsuffix']))
        GenDir = "gen_%03i" % self.generation
        GoInto(GenDir)
        GoInto("dynamics")
        self.LinkFromSettings("grompp.mdp")
        self.LinkFromSettings("min.mdp")
        self.LinkFromSettings("conf.gro")
        self.LinkFromSettings("topol.top")
        self.LinkFromGmxpath("mdrun")
        self.LinkFromGmxpath("grompp")
        self.LinkFromGmxpath("g_energy")
        self.LinkFromGmxpath("trjconv")
        Leave("dynamics")
        GoInto("msm")
        self.LinkFromSettings("conf.pdb")
        self.LinkFromSettings("AtomIndices.dat")
        self.LinkFromGmxpath("trjconv")
        os.chdir(self.root)
        
    def RunDynamics(self):
        os.chdir(self.root)
        GoInto(GenDir)
        GoInto("dynamics")
        if os.path.exists("md.log") and os.path.exists("traj.xtc"):
            print "Trajectory exists, not running dynamics"
            return
        _exec("./grompp -f min.mdp")
        _exec("./mdrun -v")
        _exec("./grompp -c confout.gro")
        _exec("./mdrun -v -stepout 10000")
        os.chdir(self.root)
    
    def BuildMSM(self):
        os.chdir(self.root)
        GoInto(GenDir)
        GoInto("msm")
        if os.path.exists("shots.gro"):
            print "shots.gro exists, not performing clustering"
        GoInto("RawData/RUN00")
        LinkFile(os.path.join(self.root,GenDir,"dynamics","traj.xtc"),os.path.join(os.getcwd(),"traj.xtc"))
        Leave("RawData/RUN00")
        _exec("ConvertDataToHDF.py -s conf.pdb -i RawData")

class ForceBalance_SCF(object):
    def __init__(self):
        self.Mao = 0
        self.root = os.getcwd()
        options, tgt_opts = parse_inputs(input_file)
        self.forcefield  = FF(options)
    
    def DetermineState():
        
    def Cycle():
        for Sim in self.SCF_Simulations:
            os.chdir(Sim.root)
            Sim.SetupGeneration()
            Sim.RunDynamics()
            Sim.BuildMSM()
            Sim.DoMBAR()
            Sim.RunQuantum()
            Sim.Gather()
        objective   = Objective(options, tgt_opts, forcefield)
        optimizer   = Optimizer(options, objective, forcefield)
        optimizer.Run()

def even_list(totlen, splitsize):
    """ Creates a list of number sequences divided as easily as possible.  
    
    Intended for even distribution of QM calculations.  However, this might
    become unnecessary if we always create one directory per calculation 
    (we need it to be this way for Q-Chem anyhow.) """
    joblens = np.zeros(splitsize,dtype=int)
    subsets = []
    for i in range(totlen):
        joblens[i%splitsize] += 1
    jobnow = 0
    for i in range(splitsize):
        subsets.append(range(jobnow, jobnow + joblens[i]))
        jobnow += joblens[i]
    return subsets

def generate_snapshots():
    print "I haven't implemented this yet"
    sys.exit(1)

def create_esp_surfaces(Molecule):
    from forcebalance.mslib import MSMS
    Rads = [VdW99[i] for i in Molecule.elem]
    #xyz = Molecule.xyzs[0]
    na = Molecule.na
    Mol_ESP = []
    printxyz=0
    for i, xyz in enumerate(Molecule.xyzs):
        print "Generating grid points for snapshot %i\r" % i
        esp_pts = []
        for j in [1.4, 1.6, 1.8, 2.0]:
            MS = MSMS(coords = list(xyz), radii = list(np.array(Rads)*j))
            MS.compute(density=0.5)
            vfloat, vint, tri = MS.getTriangles()
            a = range(len(vfloat))
            random.shuffle(a)
            if len(vfloat) < na:
                warn_press_key("I generated less ESP points than atoms!")
            for idx in a[:na]:
                esp_pts.append(vfloat[idx][:3])
        if printxyz:
            Out = []
            Out.append("%i" % (len(xyz) + len(esp_pts)))
            Out.append("Molecule plus ESP points")
            for j, x in enumerate(xyz):
                Out.append(format_xyz_coord(Molecule.elem[j], x))
            for esp_pt in esp_pts:
                Out.append(format_xyz_coord('chg',esp_pt))
            fout = open('Mao.xyz','w')
            for line in Out:
                print >> fout, line
            fout.close()
            sys.exit(1)
        Mol_ESP.append(esp_pts)
    return Mol_ESP

def do_quantum(wq_port):
    M = Molecule('shots.gro')
    M.add_quantum('../settings/qchem.in')
    digits = len(str(len(M)-1))
    formstr = '\"%%0%ii\"' % digits

    def read_quantum():
        Result = None
        os.chdir('calcs')
        for i in range(M.ns):
            dnm = eval(formstr % i)
            print "\rNow in directory %i" % i,
            if os.path.exists(dnm):
                os.chdir(dnm)
                if os.path.exists('qchem.out'):
                    Output = Molecule('qchem.out')
                    if os.path.exists('plot.esp'):
                        ESP = Molecule('plot.esp')
                        #print ESP.Data.keys()
                        Output.qm_espxyzs = list(ESP.qm_espxyzs)
                        Output.qm_espvals = list(ESP.qm_espvals)
                        #Output += Molecule('plot.esp')
                    if Result == None:
                        Result = Output
                    else:
                        Result += Output
                else:
                    raise Exception("The output file %s doesn't exist." % os.path.abspath('qchem.out'))
                os.chdir('..')
            else:
                raise Exception("The subdirectory %s doesn't exist." % os.path.abspath(dnm))
        os.chdir('..')
        return Result
    
    def run_quantum():
        ESP = create_esp_surfaces(M)
        work_queue.set_debug_flag('all')
        wq = work_queue.WorkQueue(wq_port, exclusive=False, shutdown=False)
        wq.specify_name('forcebalance')
        os.makedirs('calcs')
        os.chdir('calcs')
        for i in range(M.ns):
            dnm = eval(formstr % i)
            os.makedirs(dnm)
            os.chdir(dnm)
            M.edit_qcrems({'igdesp':len(ESP[i])})
            M.write("qchem.in", select=i)
            ESPBohr = np.array(ESP[i]) / bohrang
            np.savetxt('ESPGrid',ESPBohr)
            print "Queueing up job", dnm
            queue_up(wq, command = 'qchem40 qchem.in qchem.out', 
                     input_files = ["qchem.in", "ESPGrid"],
                     output_files = ["qchem.out", "plot.esp", "efield.dat"], verbose=False)
            os.chdir('..')
        for i in range(M.ns):
            wq_wait(wq)
        os.chdir('..')
    if os.path.exists('calcs'):
        print "calcs directory exists.  Reading calculation results."
        Result = read_quantum()
    else:
        print "calcs directory doesn't exist.  Setting up and running calculations."
        run_quantum()
        print "Now reading calculation results."
        Result = read_quantum()
    print "Writing results to qdata.txt."
    Result.write('qdata.txt')
    return Result

def gather_generations():
    shots = Molecule('shots.gro')
    qdata = Molecule('qdata.txt')
    A1    = np.array(shots.xyzs)
    A2    = np.array(qdata.xyzs)
    if A1.shape != A2.shape:
        raise Exception('shots.gro and qdata.txt appear to contain different data')
    elif np.max(np.abs((A1 - A2).flatten())) > 1e-4:
        raise Exception('shots.gro and qdata.txt appear to contain different xyz coordinates')
    shots.qm_energies = qdata.qm_energies
    shots.qm_forces   = qdata.qm_forces
    shots.qm_espxyzs     = qdata.qm_espxyzs
    shots.qm_espvals     = qdata.qm_espvals
    First = True
    if First:
        All = shots
    else:
        All += shots
    return All

def Generate(tgt_opt):
    print tgt_opt['name']
    Port = tgt_opt['wq_port']
    cwd = os.getcwd()
    simdir = os.path.join('simulations',tgt_opt['name'])
    if not os.path.exists(simdir):
        warn_press_key("%s doesn't exist!" % simdir)
    os.chdir(simdir)
    GDirs = glob.glob("gen_[0-9][0-9][0-9]")
    if len(GDirs) == 0:
        print "No gens exist."
        sys.exit()
    WriteAll = False
    All = None # Heh
    for d in GDirs:
        print "Now checking", d
        os.chdir(d)
        if os.path.exists('shots.gro') and os.path.exists('qdata.txt'):
            print "Both shots.gro and qdata.txt exist"
        elif os.path.exists('shots.gro'):
            print "shots.gro exists"
            print "I need to GENERATE qdata.txt now."
            do_quantum(Port)
        elif os.path.exists('qdata.txt'):
            warn_press_key('qdata.txt exists.')
        else:
            print "I need to GENERATE shots.gro now."
            generate_snapshots()
            do_quantum(Port)
        if All == None:
            All = gather_generations()
        else:
            All += gather_generations()
        os.chdir('..')
    All.write('all.gro')
    All.write('qdata.txt')
    os.chdir(cwd)

def main():
    options, tgt_opts = parse_inputs(sys.argv[1])
    
    """ Instantiate a ForceBalance project and call the optimizer. """
    print "\x1b[1;97m Welcome to ForceBalance version 0.12! =D\x1b[0m"
    if len(sys.argv) != 2:
        print "Please call this program with only one argument - the name of the input file."
        sys.exit(1)

    for S in tgt_opts:
        print os.getcwd()
        Generate(S)
    
    # P = Project(sys.argv[1])
    # P.Optimizer.Run()

if __name__ == "__main__":
    main()
