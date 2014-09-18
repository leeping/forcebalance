#!/usr/bin/env python

from forcebalance.molecule import Molecule
from optparse import OptionParser
from copy import deepcopy
import networkx as nx
import numpy as np
import os, sys

""" Example usage: ./dscan.py --coords optimize-tz.pdb --phi1 20 6 9 12 --phi2 6 9 12 31 --scan 15 --thre 1.0 """

parser = OptionParser()
parser.add_option('--coords', help='Input pdb (geometry) file')
parser.add_option('--phi1', help='Quartet of atoms for 1D dihedral scan (or first quartet for 2D scan', type=int, nargs=4)
parser.add_option('--phi2', help='Second quartet of atoms for 2D', type=int, nargs=4)
parser.add_option('--scan', help='Increment by which we wish to scan the dihedral angle', type=int)
parser.add_option('--thre', help='Scale the following clash radii: H-[NO] = 1.5, Hvy-Hvy = 2.4, H-Hvy = 2.0, H-H = 1.8', type=float, default=1.0)
(opts, args) = parser.parse_args()

# Get dihedral angle in radians.
def get_dihedral(mol, atms):
    return mol.measure_dihedrals(atms[0], atms[1], atms[2], atms[3])[0] * np.pi / 180

# http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
def rotation_matrix(axis,theta):
    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

# Note that r_angle must be in radians.
def rotate(qs, rotation_axis, r_angle):
    v_arr = []
    # Rotation matrix method.
    for q in range(len(qs)):
        v_arr.append(np.dot(rotation_matrix(rotation_axis, r_angle), qs[q]))
    return v_arr

def divide_molecule(mol, a0, a1):
    """ Divide a molecule into two sets of atom indices - a stationary group and a rotated group """
    G = deepcopy(mol.molecules[0])
    bond = (a0, a1)
    if bond not in G.edges() and bond[::-1] not in G.edges():
        raise RuntimeError('%s must be a bond in the molecule' % str(bond))
    G.remove_edge(bond[0], bond[1])
    Gsplit = list(nx.connected_component_subgraphs(G))
    if len(Gsplit) != 2:
        raise RuntimeError('%s must divide the molecule into two segments' % str(bond))
    G0 = Gsplit[0]
    G1 = Gsplit[1]
    if a0 in G0.node and a1 in G1.node:
        stationary_grp = G0.nodes()
        rotation_grp = G1.nodes()
    elif a1 in G0.node and a0 in G1.node:
        stationary_grp = G1.nodes()
        rotation_grp = G0.nodes()
    else:
        raise RuntimeError('Spoo!')
    return list(rotation_grp), list(stationary_grp)

def get_rotated_xyz(mol, quartets, incs):
    xyz = mol.xyzs[0].copy()
    clash = False
    phis = [get_dihedral(mol, quartet) for quartet in quartets]
    one_three = [(i[0], i[2]) for i in mol.find_angles()]
    check_grps = []
    for quartet, inc, phi in zip(quartets, incs, phis):
        g_rotate, g_stationary = divide_molecule(mol, quartet[1], quartet[2])
        check_grps.append([g_rotate, g_stationary])
        rotate_xyzs = [xyz[q] for q in g_rotate]
        stationary_xyzs = [xyz[q] for q in g_stationary]
        # Get b-c axis.
        bc_axis = xyz[quartet[2]] - xyz[quartet[1]]
        # Also shift rotation group so quartet[1] is coordinate zero.
        rotate_xyzs -= xyz[quartet[1]]
        # Radians.
        rad_inc = inc * (np.pi / 180) - phi
        # Rotate coords via rotation matrix.
        rotated_xyzs = rotate(rotate_xyzs, bc_axis, rad_inc)
        # Shift back.
        rotated_xyzs += xyz[quartet[1]]
        # Calculate distances to see if there are clashes
        # between the stationary and rotation group.
        bonds = mol.Data['bonds']
        # Update coordinates in molecule list.
        for atm in range(len(g_rotate)):
            xyz[g_rotate[atm]] = rotated_xyzs[atm]

    for grp1, grp2 in check_grps:
        for i1 in grp1:
            for i2 in grp2:
                e12 = ''.join(sorted([mol.elem[i1], mol.elem[i2]]))
                i12 = tuple(sorted([i1, i2]))
                if i12 in mol.bonds or i12 in one_three: continue
                if e12 in ['HN', 'HO', 'FH']:
                    thre = opts.thre * 1.5
                elif e12 == 'HH':
                    thre = opts.thre * 1.8
                elif 'H' in e12:
                    thre = opts.thre * 2.0
                else:
                    thre = opts.thre * 2.4
                dxij = xyz[i2] - xyz[i1]
                if np.dot(dxij, dxij) < (thre**2):
                    print mol.atomname[i1], mol.atomname[i2], np.linalg.norm(dxij)
                    clash = True
    return xyz, clash

def main():
    M = Molecule(opts.coords)
    if len(M.molecules) != 1:
        raise RuntimeError('Input coordinates must be a single contiguous molecule')
    if opts.phi1 == None:
        raise RuntimeError('phi1 (the first quartet of atoms) must be provided')
    xyzout = []
    commout = []
    xyzcsh = []
    commcsh = []
    if opts.phi2 != None: # Two dimensional scan
        for inc1 in range(0, 360, opts.scan):
            for inc2 in range(0, 360, opts.scan):
                xyzrot, clash = get_rotated_xyz(M, [opts.phi1, opts.phi2], [inc1, inc2])
                print inc1, inc2, "Clash" if clash else "Ok"
                comm = "Dihedrals %s, %s set to %i, %i" % (str(opts.phi1), str(opts.phi2), inc1, inc2)
                if clash: 
                    xyzcsh.append(xyzrot.copy())
                    commcsh.append(comm)
                else: 
                    xyzout.append(xyzrot.copy())
                    commout.append(comm)
    else: # One dimensional scan
        for inc1 in range(0, 360, opts.scan):
            xyzrot, clash = get_rotated_xyz(M, [opts.phi1], [inc1])
            print inc1, "Clash" if clash else "Ok"
            comm = "Dihedral %s set to %i" % (str(opts.phi1), inc1)
            if clash: 
                xyzcsh.append(xyzrot.copy())
                commcsh.append(comm)
            else: 
                xyzout.append(xyzrot.copy())
                commout.append(comm)
    if len(xyzout) > 0:
        M.xyzs = xyzout
        M.comms = commout
        M.write(os.path.splitext(opts.coords)[0]+"_out"+os.path.splitext(opts.coords)[1])
    if len(xyzcsh) > 0:
        M.xyzs = xyzcsh
        M.comms = commcsh
        M.write(os.path.splitext(opts.coords)[0]+"_clash"+os.path.splitext(opts.coords)[1])
    
if __name__ == "__main__":
    main()
