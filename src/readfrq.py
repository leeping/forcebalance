#!/usr/bin/env python

import os, sys, re
import numpy as np
from molecule import Molecule
from nifty import isint, isfloat

np.set_printoptions(precision=4)

def print_mode(M, mode):
    print '\n'.join(['%-3s' % M.elem[ii] + ' '.join(['% 7.3f' % j for j in i]) for ii, i in enumerate(mode)])

def read_frq_gau(gauout):
    VMode = 0
    frqs = []
    modes = []
    for line in open(gauout).readlines():
        VModeNxt = None
        if line.strip().startswith('Frequencies'):
            VMode = 2
        if VMode == 2:
            s = line.split()
            if 'Frequencies' in line:
                nfrq = len(s) - 2
                frqs += [float(i) for i in s[2:]]
            if re.match('^[ \t]*Atom', line):
                VModeNxt = 3
                readmodes = [[] for i in range(nfrq)]
            if 'Imaginary Frequencies' in line:
                VMode = 0
        if VMode == 3:
            s = line.split()
            if len(s) != nfrq*3+2:
                VMode = 0
                modes += readmodes[:]
            else:
                for i in range(nfrq):
                    readmodes[i].append([float(s[j]) for j in range(2+3*i,5+3*i)])
        if VModeNxt != None: VMode = VModeNxt
    unnorm = [np.array(i) for i in modes]
    return np.array(frqs), [i/np.linalg.norm(i) for i in unnorm]

def read_frq_qc(qcout):
    VMode = 0
    frqs = []
    modes = []
    for line in open(qcout).readlines():
        VModeNxt = None
        if 'VIBRATIONAL ANALYSIS' in line:
            VMode = 1
        if VMode > 0 and line.strip().startswith('Mode:'):
            VMode = 2
        if VMode == 2:
            s = line.split()
            if 'Frequency:' in line:
                nfrq = len(s) - 1
                frqs += [float(i) for i in s[1:]]
            if re.match('^ +X', line):
                VModeNxt = 3
                readmodes = [[] for i in range(nfrq)]
            if 'Imaginary Frequencies' in line:
                VMode = 0
        if VMode == 3:
            s = line.split()
            if len(s) != nfrq*3+1:
                VMode = 2
                modes += readmodes[:]
            elif 'TransDip' not in s:
                for i in range(nfrq):
                    readmodes[i].append([float(s[j]) for j in range(1+3*i,4+3*i)])
        # print VMode, line,
        if VModeNxt != None: VMode = VModeNxt
    unnorm = [np.array(i) for i in modes]
    return np.array(frqs), [i/np.linalg.norm(i) for i in unnorm]

def read_frq_psi(psiout):
    """ """
    VMode = 0
    XMode = 0
    EMode = 0
    frqs = []
    modes = []
    xyzs = []
    xyz = []
    elem = []
    for line in open(psiout).readlines():
        VModeNxt = None
        if 'Frequency:' in line:
            VModeNxt = 1
            if line.split()[-1].endswith('i'):
                frqs.append(-1*float(line.split()[-1][:-1]))
                # frqs.append(0.0) # After the optimization this mode is going to be useless...
            else:
                frqs.append(float(line.split()[-1]))
        if VMode == 1:
            if re.match('^[ \t]+X', line):
                VModeNxt = 2
                readmode = []
        if VMode == 2:
            s = line.split()
            if len(s) != 5:
                VMode = 0
                modes.append(readmode[:])
            else:
                m = float(s[-1])
                # Un-massweight the eigenvectors so that the output matches Q-Chem or Gaussian.
                readmode.append([float(i)/np.sqrt(m) for i in s[1:4]])
        if VModeNxt != None: VMode = VModeNxt
        if XMode == 1:
            s = line.split()
            if len(s) == 4 and isfloat(s[1]) and isfloat(s[2]) and isfloat(s[3]):
                e = s[0]
                xyz.append([float(i) for i in s[1:4]])
                if EMode == 1:
                    elem.append(e)
            elif len(xyz) > 0:
                xyzs.append(np.array(xyz))
                xyz = []
                XMode = 0
        if line.strip().startswith("Geometry (in Angstrom)"):
            XMode = 1
            EMode = len(elem) == 0
    unnorm = [np.array(i) for i in modes]
    return np.array(frqs), [i/np.linalg.norm(i) for i in unnorm], elem, np.array(xyzs[-1])

def scale_freqs(arr):
    """ Apply harmonic vibrational scaling factors. """
    # Scaling factors are taken from:
    # Jeffrey P. Merrick, Damian Moran, and Leo Radom
    # An Evaluation of Harmonic Vibrational Frequency Scale Factors
    # J. Phys. Chem. A 2007, 111, 11683-11700
    #----
    # The dividing line is just a guess (used by A. K. Wilson)
    div = 1000.0
    # High-frequency scaling factor for MP2/aTZ
    hscal = 0.960
    # Low-frequency scaling factor for MP2/aTZ
    lscal = 1.012
    print "  Freq(Old)  Freq(New)  RawShift  NewShift   DShift"
    def scale_one(frq):
        if frq > div:
            if hscal < 1.0:
                # Amount that the frequency is above the dividing line
                above = (frq-div)
                # Maximum frequency shift
                maxshf = (div/hscal-div)
                # Close to the dividing line, the frequency should be
                # scaled less because we don't want the orderings of
                # the frequencies to switch.
                # Far from the dividing line, we want the frequency shift
                # to approach the uncorrected shift.
                # 1.0/(1.0 + maxshf/above) is a scale of how far we are from the dividing line.
                att = 1.0/(1.0 + maxshf/above)
                # shift is the uncorrected shift.
                shift = (hscal - 1.0) * frq
                newshift = att*shift
                print "%10.3f %10.3f  % 9.3f % 9.3f % 8.3f" % (frq, frq+newshift, shift, newshift, newshift-shift)
                newfrq = frq+newshift
                return newfrq
            else:
                return frq*hscal
        elif frq <= div:
            if lscal > 1.0:
                below = (div-frq)
                maxshf = (div-div/lscal)
                att = 1.0/(1.0 + maxshf/below)
                shift = (lscal - 1.0) * frq
                newshift = att*shift
                print "%10.3f %10.3f  % 9.3f % 9.3f % 8.3f" % (frq, frq+newshift, shift, newshift, newshift-shift)
                newfrq = frq+newshift
                return newfrq
            else:
                return frq*lscal
    return np.array([scale_one(i) for i in arr])

def main():
    Mqc = Molecule(sys.argv[2])
    psifrqs, psimodes, _, __ = read_frq_psi(sys.argv[1])
    qcfrqs, qcmodes = read_frq_qc(sys.argv[2])
    gaufrqs, gaumodes = read_frq_gau(sys.argv[3])
    for i, j, ii, jj, iii, jjj in zip(psifrqs, psimodes, qcfrqs, qcmodes, gaufrqs, gaumodes):
        print "PsiFreq:", i, "QCFreq", ii, "GauFreq", iii
        print "PsiMode:", np.linalg.norm(j)
        print_mode(Mqc, j)
        print "QCMode:", np.linalg.norm(jj)
        if np.linalg.norm(j - jj) < np.linalg.norm(j + jj):
            print_mode(Mqc, jj)
        else:
            print_mode(Mqc, -1 * jj)
        print "GauMode:", np.linalg.norm(jjj)
        if np.linalg.norm(j - jjj) < np.linalg.norm(j + jjj):
            print_mode(Mqc, jjj)
        else:
            print_mode(Mqc, -1 * jjj)
        
        print "DMode (QC-Gau):", 
        if np.linalg.norm(jj - jjj) < np.linalg.norm(jj + jjj):
            print np.linalg.norm(jj - jjj)
            print_mode(Mqc, jj - jjj)
        else:
            print np.linalg.norm(jj + jjj)
            print_mode(Mqc, jj + jjj)

        print "DMode (QC-Psi):", 
        if np.linalg.norm(jj - j) < np.linalg.norm(jj + j):
            print np.linalg.norm(jj - j)
            print_mode(Mqc, jj - j)
        else:
            print np.linalg.norm(jj + j)
            print_mode(Mqc, jj + j)

if __name__ == "__main__":
    main()
