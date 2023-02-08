#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import range
import os, sys, re
import numpy as np
from .molecule import Molecule, Elements
from .nifty import isint, isfloat

np.set_printoptions(precision=4)

def print_mode(M, mode):
    print('\n'.join(['%-3s' % M.elem[ii] + ' '.join(['% 7.3f' % j for j in i]) for ii, i in enumerate(mode)]))

def read_frq_gau(gauout):
    XMode = 0
    xyz = []
    elem = []
    elemThis = []
    VMode = 0
    frqs = []
    intens = []
    modes = []
    for line in open(gauout).readlines():
        line = line.strip().expandtabs()
        if XMode >= 1:
            # Perfectionist here; matches integer, element, and three floating points
            if re.match("^[0-9]+ +[0-9]+ +[0-9]+( +[-+]?([0-9]*\.)?[0-9]+){3}$", line):
                XMode = 2
                sline = line.split()
                elemThis.append(Elements[int(sline[1])])
                xyz.append([float(i) for i in sline[3:]])
            elif XMode == 2: # Break out of the loop if we encounter anything other than atomic data
                if elem == []:
                    elem = elemThis
                elif elem != elemThis:
                    logger.error('Gaussian output parser will not work if successive calculations have different numbers of atoms!\n')
                    raise RuntimeError
                elemThis = []
                xyz = np.array(xyz)
                XMode = -1
        elif XMode == 0 and "Coordinates (Angstroms)" in line:
            XMode = 1
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
            if 'IR Inten' in line:
                intens += [float(i) for i in s[3:]]
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
        if VModeNxt is not None: VMode = VModeNxt
    unnorm = [np.array(i) for i in modes]
    return np.array(frqs), [i/np.linalg.norm(i) for i in unnorm], np.array(intens), elem, xyz

def read_frq_tc(tcout, scrdir='scr'):
    # Unfortunately, TeraChem's frequency data is scattered in the output file and scratch folder
    lineCounter = -100
    xyzpath = os.path.join(os.path.split(os.path.abspath(tcout))[0], scrdir, 'CentralGeometry.initcond.xyz')
    tcdat = os.path.join(os.path.split(os.path.abspath(tcout))[0], scrdir, 'Frequencies.dat')
    if not os.path.exists(xyzpath):
        raise RuntimeError("%s doesn't exist; please provide a scratch folder to this function" % xyzpath)
    if not os.path.exists(tcdat):
        raise RuntimeError("%s doesn't exist; please provide a scratch folder to this function" % tcdat)
    Mxyz = Molecule(xyzpath)

    # This piece of Yudong's code reads the intensities
    found_vib = False
    freqs = []
    intensities = []
    for line in open(tcout):
        if 'Vibrational Frequencies/Thermochemical Analysis After Removing Rotation and Translation' in line:
            found_vib = True
        if found_vib:
            ls = line.split()
            if len(ls) == 8 and ls[0].isdigit():
                freqs.append(float(ls[2]))
                intensities.append(float(ls[3]))
            elif len(ls) == 3 and ls[2].endswith('i'):
                freqs.append(-1*float(ls[2][:-1]))
                intensities.append(0.0)
            if line.strip() == '':
                break
    if found_vib is False:
        raise RuntimeError("No frequency data was found in file %s" % filename)
        
    for lineNumber, line in enumerate(open(tcdat).readlines()):
        s = line.split()
        if lineNumber == 0:
            numAtoms = int(s[-1])
        elif lineNumber == 1:
            numModes = int(s[-1])
            # Make list of unnormalized modes to be read in
            frqs = np.zeros(numModes, dtype=float)
            unnorm = [np.zeros(3*numAtoms, dtype=float) for i in range(numModes)]
        elif all([isint(i) for i in s]):
            lineCounter = 0
            modeNumbers = [int(i) for i in s]
        elif lineCounter == 1:
            theseFrqs = [float(i) for i in s]
            if len(theseFrqs) != len(modeNumbers):
                raise RuntimeError('Parser error! Expected # frequencies to equal # modes')
            for i in range(len(theseFrqs)):
                frqs[modeNumbers[i]] = theseFrqs[i]
        elif lineCounter >= 3:
            if lineCounter%3 == 0:
                if not isint(s[0]):
                    raise RuntimeError('Parser error! Expected integer at start of line')
                disps = [float(i) for i in s[1:]]
            else:
                disps = [float(i) for i in s]
            idx = lineCounter-3
            if len(disps) != len(modeNumbers):
                raise RuntimeError('Parser error! Expected # displacements to equal # modes')
            for i in range(len(disps)):
                unnorm[modeNumbers[i]][lineCounter-3] = disps[i]
            if idx == 3*numAtoms-1:
                lineCounter = -100
        lineCounter += 1
    if np.max(np.abs(np.array(frqs)-np.array(freqs))) > 1.0:
        raise RuntimeError("Inconsistent frequencies from TeraChem output and scratch")
    return np.array(frqs), [i/np.linalg.norm(i) for i in unnorm], np.array(intensities), Mxyz.elem, Mxyz.xyzs[0]

def read_frq_qc(qcout):
    XMode = 0
    xyz = []
    elem = []
    elemThis = []
    VMode = 0
    frqs = []
    modes = []
    intens = []
    for line in open(qcout).readlines():
        line = line.strip().expandtabs()
        if XMode >= 1:
            # Perfectionist here; matches integer, element, and three floating points
            if re.match("^[0-9]+ +[A-Z][A-Za-z]?( +[-+]?([0-9]*\.)?[0-9]+){3}$", line):
                XMode = 2
                sline = line.split()
                elemThis.append(sline[1])
                xyz.append([float(i) for i in sline[2:]])
            elif XMode == 2: # Break out of the loop if we encounter anything other than atomic data
                if elem == []:
                    elem = elemThis
                elif elem != elemThis:
                    logger.error('Q-Chem output parser will not work if successive calculations have different numbers of atoms!\n')
                    raise RuntimeError
                elemThis = []
                xyz = np.array(xyz)
                XMode = -1
        elif XMode == 0 and  re.match("Standard Nuclear Orientation".lower(), line.lower()):
            XMode = 1
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
            if re.match('^X +Y +Z', line):
                VModeNxt = 3
                readmodes = [[] for i in range(nfrq)]
            if 'IR Intens:' in line:
                intens += [float(i) for i in s[2:]]
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
        if VModeNxt is not None: VMode = VModeNxt
    unnorm = [np.array(i) for i in modes]
    return np.array(frqs), [i/np.linalg.norm(i) for i in unnorm], np.array(intens), elem, xyz

def read_frq_psi_current(psiout):
    """ """
    VMode = 0
    XMode = 0
    EMode = 0
    frq_num = 0
    mode_num = 0
    skip_modes = 0
    frqs = []
    modes = []
    xyzs = []
    xyz = []
    elem = []
    readmodes = {}
    for line in open(psiout).readlines():
        VModeNxt = None
        if 'rotation-like modes' in line:
            #Output may contain rotation modes, skip these if so
            skip_modes = int(line.split('include ')[1].split('un-projected')[0])
        if 'Freq [cm^-1]' in line:
            VModeNxt = 1
            s = line.split()
            for mode in s[2:]:
                frq_num += 1
                if frq_num > skip_modes:
                    if mode.endswith('i'):
                        frqs.append(-1*float(mode[:-1]))
                        # frqs.append(0.0) # After the optimization this mode is going to be useless...
                    else:
                        frqs.append(float(mode))
        if VMode == 1:
            if re.match('^\s*[0-9]', line) and mode_num >= skip_modes:
                s = line.split()
                line_modes = int(len(s[2:])/3)
                for mode in range(line_modes):
                    if mode not in readmodes:
                        readmodes[mode] = []
                    readmodes[mode].append([float(m) for m in s[2+mode*3:2+mode*3+3]])
            elif re.match('^\s*[0-9]', line):
                s = line.split()
                mode_num += int(len(s[2:])/3)
                VMode = 0
                VModeNxt = None
            elif re.match('^[ \t\r\n\s]*$', line):
                VMode = 0
                VModeNxt = None
                for mode in readmodes.keys():
                    modes.append(readmodes[mode])
                readmodes = {}
        if VModeNxt is not None: VMode = VModeNxt
        if XMode == 1:
            s = line.split()
            if len(s) == 5 and isfloat(s[1]) and isfloat(s[2]) and isfloat(s[3]):
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
            s = line.split()
            if len(s) == 5 and isfloat(s[1]) and isfloat(s[2]) and isfloat(s[3]):
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
    #Eigenvectors are now normalized and non-mass weighted, so can just supply these directly without modification
    modes = [np.array(i) for i in modes]
    return np.array(frqs), modes, np.zeros_like(frqs), elem, np.array(xyzs[-1])

def read_frq_psi_legacy(psiout):
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
        if VModeNxt is not None: VMode = VModeNxt
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
    return np.array(frqs), [i/np.linalg.norm(i) for i in unnorm], np.zeros_like(frqs), elem, np.array(xyzs[-1])

def read_frq_fb(vfnm):
    """ Read ForceBalance-formatted vibrational data from a vdata.txt file. """
    ## Number of atoms
    na = -1
    ref_eigvals = []
    ref_eigvecs = []
    an = 0
    ln = 0
    cn = -1
    elem = []
    for line in open(vfnm):
        line = line.split('#')[0] # Strip off comments
        s = line.split()
        if len(s) == 1 and na == -1:
            na = int(s[0])
            xyz = np.zeros((na, 3))
            cn = ln + 1
        elif ln == cn:
            pass
        elif an < na and len(s) == 4:
            elem.append(s[0])
            xyz[an, :] = np.array([float(i) for i in s[1:]])
            an += 1
        elif len(s) == 1:
            ref_eigvals.append(float(s[0]))
            ref_eigvecs.append(np.zeros((na, 3)))
            an = 0
        elif len(s) == 3:
            ref_eigvecs[-1][an, :] = np.array([float(i) for i in s])
            an += 1
        elif len(s) == 0:
            pass
        else:
            logger.info(line + '\n')
            logger.error("This line doesn't comply with our vibration file format!\n")
            raise RuntimeError
        ln += 1
    ref_eigvals = np.array(ref_eigvals)
    ref_eigvecs = np.array(ref_eigvecs)
    for v2 in ref_eigvecs:
        v2 /= np.linalg.norm(v2)
    return ref_eigvals, ref_eigvecs, np.zeros_like(ref_eigvals), elem, xyz

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
    print("  Freq(Old)  Freq(New)  RawShift  NewShift   DShift")
    def scale_one(frq):
        if frq > div:
            if hscal < 1.0:
                # Amount that the frequency is above the dividing line
                # above = (frq-div)
                # Maximum frequency shift
                # maxshf = (div/hscal-div)
                # Close to the dividing line, the frequency should be
                # scaled less because we don't want the orderings of
                # the frequencies to switch.
                # Far from the dividing line, we want the frequency shift
                # to approach the uncorrected shift.
                # 1.0/(1.0 + maxshf/above) is a scale of how far we are from the dividing line.
                # att = 1.0/(1.0 + maxshf/above)
                # shift is the uncorrected shift.
                att = (frq-div)/(frq-hscal*div)
                shift = (hscal - 1.0) * frq
                newshift = att*shift
                print("%10.3f %10.3f  % 9.3f % 9.3f % 8.3f" % (frq, frq+newshift, shift, newshift, newshift-shift))
                newfrq = frq+newshift
                return newfrq
            else:
                return frq*hscal
        elif frq <= div:
            if lscal > 1.0:
                # below = (div-frq)
                # maxshf = (div-div/lscal)
                # att = 1.0/(1.0 + maxshf/below)
                att = (frq-div)/(frq-lscal*div)
                shift = (lscal - 1.0) * frq
                newshift = att*shift
                print("%10.3f %10.3f  % 9.3f % 9.3f % 8.3f" % (frq, frq+newshift, shift, newshift, newshift-shift))
                newfrq = frq+newshift
                return newfrq
            else:
                return frq*lscal
    return np.array([scale_one(i) for i in arr])

def read_frq_gen(fout):
    ln = 0
    for line in open(fout):
        if 'TeraChem' in line:
            return read_frq_tc(fout)
        elif 'Q-Chem' in line:
            return read_frq_qc(fout)
        elif 'Psi4' in line and 'release' in line:
            ls = line.split()
            version = ls[1]
            if version >= '1.2':
                return read_frq_psi_current(fout)
            else:
                return read_frq_psi_legacy(fout)
        elif 'Gaussian' in line:
            return read_frq_gau(fout)
        elif 'ForceBalance' in line:
            return read_frq_fb(fout)
        ln += 1
    raise RuntimeError('Cannot determine format')

def main():
    Mqc = Molecule(sys.argv[2])
    psifrqs, psimodes, _, __, ___ = read_frq_gen(sys.argv[1])
    qcfrqs, qcmodes, _, __, ___ = read_frq_gen(sys.argv[2])
    gaufrqs, gaumodes, _, __, ___ = read_frq_gen(sys.argv[3])
    for i, j, ii, jj, iii, jjj in zip(psifrqs, psimodes, qcfrqs, qcmodes, gaufrqs, gaumodes):
        print("PsiFreq:", i, "QCFreq", ii, "GauFreq", iii)
        print("PsiMode:", np.linalg.norm(j))
        print_mode(Mqc, j)
        print("QCMode:", np.linalg.norm(jj))
        if np.linalg.norm(j - jj) < np.linalg.norm(j + jj):
            print_mode(Mqc, jj)
        else:
            print_mode(Mqc, -1 * jj)
        print("GauMode:", np.linalg.norm(jjj))
        if np.linalg.norm(j - jjj) < np.linalg.norm(j + jjj):
            print_mode(Mqc, jjj)
        else:
            print_mode(Mqc, -1 * jjj)
        
        print("DMode (QC-Gau):", end=' ') 
        if np.linalg.norm(jj - jjj) < np.linalg.norm(jj + jjj):
            print(np.linalg.norm(jj - jjj))
            print_mode(Mqc, jj - jjj)
        else:
            print(np.linalg.norm(jj + jjj))
            print_mode(Mqc, jj + jjj)

        print("DMode (QC-Psi):", end=' ') 
        if np.linalg.norm(jj - j) < np.linalg.norm(jj + j):
            print(np.linalg.norm(jj - j))
            print_mode(Mqc, jj - j)
        else:
            print(np.linalg.norm(jj + j))
            print_mode(Mqc, jj + j)

if __name__ == "__main__":
    main()
