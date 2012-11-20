#!/usr/bin/env python

from numpy import *
import os, sys
from forcebalance.molecule import Molecule
from forcebalance.nifty import *
import work_queue

wq_port = 1729
work_queue.set_debug_flag('all')
wq = work_queue.WorkQueue(port=wq_port, exclusive=False, shutdown=False)
#wq.specify_master_mode(WORK_QUEUE_MASTER_MODE_CATALOG)
wq.specify_name('chloroform')
print('Work Queue listening on %d' % (wq.port))

Dirs = ["3.4-3.5", "3.6-3.7", "3.7-3.8", "3.8-3.9", "3.9-4.0", 
        "4.0-4.1", "4.1-4.2", "4.2-4.3", "4.3-4.4", "4.4-4.5", 
        "4.5-4.6", "4.6-4.7", "4.7-4.8", "4.8-4.9", "4.9-5.0", 
        "5.0-5.1", "5.1-5.2", "5.2-5.3", "5.3-5.4", "5.4-5.5", 
        "5.5-5.6", "5.6-5.7"]

dists = []
dists2 = []
M = None
for d in Dirs:
    os.chdir(d)
    for f in os.listdir('.'):
        if os.path.splitext(f)[1] == '.xyz':
            if M == None:
                M = Molecule(f,ftype="tinker")
                dists.append(linalg.norm(M.xyzs[0][5] - M.xyzs[0][0]))
            else:
                N = Molecule(f,ftype="tinker")
                M.xyzs.append(N.xyzs[0])
                M.comms.append('')
                dists.append(linalg.norm(N.xyzs[0][5] - N.xyzs[0][0]))
    os.chdir('..')

#print argsort(array(dists))

Mopt = Molecule('chcl3.xyz_2',ftype='tinker')

M = M[argsort(array(dists))]
M.align()

print "Aligning monomers..."
for i in range(len(M)):
    M1 = M[i].atom_select(range(5))
    M1.xyzs = [M1.xyzs[0], Mopt.xyzs[0]]
    M1.align(center=False)
    M2 = M[i].atom_select(range(5,10))
    M2.xyzs = [M2.xyzs[0], Mopt.xyzs[0]]
    M2.align(center=False)
    M.xyzs[i] = vstack((M1.xyzs[1],M2.xyzs[1]))

M.add_quantum("qtemp.in")

Energies = []
os.chdir("qchem")
for i in range(len(M)):
    dnm = "%04i" % i
    if not os.path.exists(dnm):
        os.makedirs(dnm)
    os.chdir(dnm)
    Cnvgd = False
    if os.path.exists("qchem.out"):
        try:
            O = Molecule("qchem.out")
            Cnvgd = True
            Energies.append(O.qm_energies[0])
        except: 
            print "Phailed to read in molecule, be careful"
            raw_input()
    if not Cnvgd:
        print "Launching calculation in %s" % os.getcwd()
        M[i].write("qchem.in")
        queue_up(wq,"qchem40 qchem.in qchem.out",input_files=["qchem.in"],output_files=["qchem.out"])
    else:
        print "\rCalculation converged in %s" % os.getcwd(),
    os.chdir('..')

wq_wait(wq)

M.qm_energies = Energies
M.resid=[1,1,1,1,1,2,2,2,2,2]
M.resname=['CFM' for i in range(10)]
M.write('all.gro')
M.write('qdata.txt')

#M.write('dimers.xyz')
