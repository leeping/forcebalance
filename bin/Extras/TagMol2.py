#!/usr/bin/env python

import numpy as np
import itertools
import os, sys
import networkx as nx
from collections import defaultdict
from forcebalance.molecule import Molecule
from forcebalance import Mol2

#=================================================#
#|          Mol2 File Tagging Script             |#
#|                 Lee-Ping Wang                 |#
#|                                               |#
#| This script takes a mol2 file as an argument. |#
#|                                               |#
#| The mol2 file is assumed to be atom-typed and |#
#| charged (from something like antechamber).    |#
#|                                               |#
#| On output, a new mol2 file is generated that  |#
#| contains ForceBalance parameterization tags   |#
#| printed to the 'status bit' field for each    |#
#| atom.  Atoms that are chemically equivalent   |#
#| up to a number of bonds (hard-coded by the    |#
#| Max variable) are taken to be equivalent and  |#
#| tagged with the RPT keyword instead of the    |#
#| PARM keyword.                                 |#
#|                                               |#
#| This script also symmetrizes the charges and  |#
#| makes sure that the decimal numbers add up    |#
#| EXACTLY to an integer.                        |#
#|                                               |#
#| Required: networkx package                    |#
#|           Mol2 and molecule modules in        |#
#|           'forcebalance' directory            |#
#|                                               |#
#=================================================#

#===============
# Currently, the Mol2 class is not able to print the end
# of the Mol2 file, which is needed by AMBER.
# So we put it in manually.
Ending = """@<TRIPOS>SUBSTRUCTURE
     1 <1>         1 TEMP              0 ****  ****    0 ROOT
"""

#===============
# Derived class from networkx graph;
# this is a graph description of molecules.
# Stolen from nanoreactor scripts, in order to be standalone
class MolG(nx.Graph):
    def __eq__(self, other):
        # This defines whether two MyG objects are "equal" to one another.
        return nx.is_isomorphic(self,other,node_match=nodematch)
    def __hash__(self):
        ''' The hash function is something we can use to discard two things that are obviously not equal.  Here we neglect the hash. '''
        return 1
    def L(self):
        ''' Return a list of the sorted atom numbers in this graph. '''
        return sorted(self.nodes())
    def AStr(self):
        ''' Return a string of atoms, which serves as a rudimentary 'fingerprint' : '99,100,103,151' . '''
        return ','.join(['%i' % i for i in self.L()])
    def e(self):
        ''' Return an array of the elements.  For instance ['H' 'C' 'C' 'H']. '''
        elems = nx.get_node_attributes(self,'e')
        return [elems[i] for i in self.L()]
    def ef(self):
        ''' Create an Empirical Formula '''
        Formula = list(self.e())
        return ''.join([('%s%i' % (k, Formula.count(k)) if Formula.count(k) > 1 else '%s' % k) for k in sorted(set(Formula))])
    def x(self):
        ''' Get a list of the coordinates. '''
        coors = nx.get_node_attributes(self,'x')
        return np.array([coors[i] for i in self.L()])

def col(vec):
    """Given any list, array, or matrix, return a 1-column matrix."""
    return np.mat(np.array(vec).reshape(-1, 1))

def row(vec):
    """Given any list, array, or matrix, return a 1-row matrix."""
    return np.mat(np.array(vec).reshape(1, -1))

def flat(vec):
    """Given any list, array, or matrix, return a single-index array."""
    return np.array(vec).reshape(-1)


def build_graph(M):
    M.require('bonds')
    G = MolG()
    for i, a in enumerate(M.elem):
        G.add_node(i)
        nx.set_node_attributes(G,'n',{i:M.atomname[i]})
        nx.set_node_attributes(G,'e',{i:a})
        nx.set_node_attributes(G,'x',{i:M.xyzs[0][i]})
    for i in enumerate(M.bonds):
        G.add_edge(i[0], i[1])
    return G

def get_equivalent_atoms(MyG):
    GDat = MyG.nodes(data=True)
    GDict = {}
    for i in GDat:
        GDict[i[0]] = i[1]
    
    PairPaths = nx.all_pairs_shortest_path_length(MyG)
    Walks = []
    Max   = 20
    for A in PairPaths:
        Walk = defaultdict(list)
        for B in PairPaths[A]:
            if PairPaths[A][B] > Max: continue
            Walk[PairPaths[A][B]].append(GDict[B]['e'])
        for idx, elist in Walk.items():
            Walk[idx] = ''.join([('%s%i' % (k, elist.count(k)) if elist.count(k) > 1 else '%s' % k) for k in sorted(set(elist))])
            # ef = 
            # Walk[i] = 
        Walks.append(Walk)
    J = 0
    Map = []
    Suffix = []
    MyList = []
    for i, wi in enumerate(Walks):
        UniqueFlag = True
        atomi = GDict[i]['n']
        for j, wj in enumerate(Walks):
            atomj = GDict[j]['n']
            if i <= j: continue
            if wi == wj:
                Repeat = atomj
                UniqueFlag = False
                break
        MyList.append(J)
        if UniqueFlag:
            Map.append([i])
            J += 1
            Suffix.append(" # PARM 8")
        else:
            Map[MyList[j]].append(i)
            Suffix.append(" # RPT 8 COUL:%s /RPT" % Repeat)

    QMat = np.zeros((len(GDat), len(GDat)),dtype=float)
    for i in Map:
        for ii, jj in list(itertools.product(i, i)):
            QMat[ii, jj] = 1.0 / len(i)
    
    return QMat, Suffix

def charge_as_array(M2Mol, QMat):
    oldq = np.array([atom.charge for atom in M2Mol.atoms])
    def get_symq(q):
        return np.array([float("% .6f" % i) for i in flat(np.mat(QMat) * col(oldq))])
    J = 0
    M = 0
    oldq  = get_symq(oldq)
    print "Total charge is % .6f" % sum(oldq)
    print "Doing something stupid to make sure all of the charges add up to EXACTLY an integer."
    CorrQ = (float(int(round(sum(oldq)))) - sum(oldq)) / len(oldq)
    print "Adding % .6f to all charges" % CorrQ
    oldq += CorrQ
    while True:
        print "Adjusting charge element %i by" % (M%len(oldq)), J,
        oldq[M%len(oldq)] += J
        newq = get_symq(oldq)
        print ": Total charge is now % .6f" % sum(newq)
        if abs(float(int(round(sum(newq)))) - sum(newq)) < 1e-8:
            break
        oldq[M%len(oldq)] -= J
        if J <= 0:
            J = -J + (1e-6 if M%len(oldq) == 0 else 0)
        else:
            J = -J
        M += 1
        if M == 10000:
            raise Exception("Tried 10,000 iterations of charge adjustment, I probably screwed up.")

    return list(newq)

def update_mol2(M2Mol, newq, newsuf):
    for i, a in enumerate(M2Mol.atoms):
        a.set_charge(newq[i])
        a.set_status_bit(newsuf[i])

def main():
    M = Molecule(sys.argv[1])
    MyG = build_graph(M)
    QMat, Suffix = get_equivalent_atoms(MyG)
    M2   = Mol2.mol2_set(sys.argv[1]).compounds.items()[0][1]
    NewQ = charge_as_array(M2, QMat)
    update_mol2(M2, NewQ, Suffix)
    
    

    if len(sys.argv) >= 3:
        with open(sys.argv[2],'w') as f: 
            print >> f, M2
            print >> f, Ending
    else:
        print M2
        print Ending
    
if __name__ == "__main__":
    main()
