""" modified vibration.py for internal coordinate hessian fitting
"""
from __future__ import division

from builtins import zip
from builtins import range
import os
import shutil
from forcebalance.nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohr2ang, warn_press_key, pvec1d, pmat2d
import numpy as np
from numpy.linalg import multi_dot
from forcebalance.target import Target
from forcebalance.molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
from subprocess import PIPE
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
# from ._assign import Assign
from scipy import optimize
from collections import OrderedDict
#from _increment import Vibration_Build

from forcebalance.output import getLogger
from forcebalance.optimizer import Counter
from geometric.internal import PrimitiveInternalCoordinates, Distance, Angle, Dihedral, OutOfPlane

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

import copy
logger = getLogger(__name__)

Bohr2nm = 0.0529177210903
bohr2ang     = 0.529177210903
Hartree2kJmol = 2625.4996394798254

class Hessian(Target):
    def __init__(self,options,tgt_opts,forcefield):
        """Initialization."""
        
        # Initialize the SuperClass!
        super(Hessian,self).__init__(options,tgt_opts,forcefield)
        #======================================#
        # Options that are given by the parser #
        #======================================#
        self.set_option(tgt_opts,'hess_normalize_type')
        ## Option for how much data to write to disk.
        self.set_option(tgt_opts,'writelevel','writelevel')
        ## Option for normal mode calculation w/ or w/o geometry optimization
        self.set_option(tgt_opts,'optimize_geometry', default=1)
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## Build internal coordinates.
        self._build_internal_coordinates()
        ## The vdata.txt file that contains the qm hessian.
        self.hfnm = os.path.join(self.tgtdir,"hdata.txt")
        ## The vdata.txt file that contains the vibrations.
        self.vfnm = os.path.join(self.tgtdir,"vdata.txt")
        ## Read in the reference data
        self.read_reference_data()

        ## Build keyword dictionaries to pass to engine.
        engine_args = OrderedDict(list(self.OptionDict.items()) + list(options.items()))
        engine_args.pop('name', None)
        ## Create engine object.
        self.engine = self.engine_(target=self, **engine_args)

        ## create wts and denominator
        self.get_wts()
        self.denom = 1

    def _build_internal_coordinates(self):
        m  = Molecule(os.path.join(self.tgtdir, "input.mol2"))
        IC = PrimitiveInternalCoordinates(m)
        self.IC = IC

    def read_reference_data(self): # HJ: copied from vibration.py and modified 
        """ Read the reference hessian data from a file. """
        self.ref_Hq_flat = np.loadtxt(self.hfnm)
        Hq_size =int(np.sqrt(len(self.ref_Hq_flat)))
        self.ref_Hq = self.ref_Hq_flat.reshape((Hq_size, Hq_size))

        """ Read the reference vibrational data from a file. """
        ## Number of atoms
        self.na = -1
        self.ref_eigvals = []
        self.ref_eigvecs = []
        an = 0
        ln = 0
        cn = -1
        for line in open(self.vfnm):
            line = line.split('#')[0] # Strip off comments
            s = line.split()
            if len(s) == 1 and self.na == -1:
                self.na = int(s[0])
                xyz = np.zeros((self.na, 3))
                cn = ln + 1
            elif ln == cn:
                pass
            elif an < self.na and len(s) == 4:
                xyz[an, :] = np.array([float(i) for i in s[1:]])
                an += 1
            elif len(s) == 1:
                if float(s[0]) < 0:
                    logger.warning('Warning: Setting imaginary frequency = % .3fi to zero.\n' % abs(float(s[0])))
                    self.ref_eigvals.append(0.0)
                else:
                    self.ref_eigvals.append(float(s[0]))
                self.ref_eigvecs.append(np.zeros((self.na, 3)))
                an = 0
            elif len(s) == 3:
                self.ref_eigvecs[-1][an, :] = np.array([float(i) for i in s])
                an += 1
            elif len(s) == 0:
                pass
            else:
                logger.info(line + '\n')
                logger.error("This line doesn't comply with our vibration file format!\n")
                raise RuntimeError
            ln += 1
        self.ref_eigvals = np.array(self.ref_eigvals)
        self.ref_eigvecs = np.array(self.ref_eigvecs)
        self.ref_xyz = np.array(xyz)
        for v2 in self.ref_eigvecs:
            v2 /= np.linalg.norm(v2)
        return

    def get_wts(self):
        
        nb = len([ic for ic in self.IC.Internals if isinstance(ic,Distance) ])
        nba = nb + len([ic for ic in self.IC.Internals if isinstance(ic,Angle) ])
        nbap = nba + len([ic for ic in self.IC.Internals if isinstance(ic,Dihedral) ])

        Hq_size =int(np.sqrt(len(self.ref_Hq_flat)))
        if  self.hess_normalize_type == 1 or self.hess_normalize_type == 2 : 
            wts = np.ones((Hq_size, Hq_size))

            bb = np.sum(self.ref_Hq[:nb,:nb]* self.ref_Hq[:nb,:nb])
            ba = 2*np.sum(self.ref_Hq[:nb, nb:nba]*self.ref_Hq[:nb, nb:nba]) 
            bp = 2*np.sum(self.ref_Hq[:nb, nba:nbap]*self.ref_Hq[:nb, nba:nbap])
            bi = 2*np.sum(self.ref_Hq[:nb, nbap:]*self.ref_Hq[:nb, nbap:])
            aa = np.sum(self.ref_Hq[nb:nba, nb:nba]*self.ref_Hq[nb:nba, nb:nba])
            ap = 2*np.sum(self.ref_Hq[nb:nba, nba:nbap]*self.ref_Hq[nb:nba, nba:nbap])
            ai = 2*np.sum(self.ref_Hq[nb:nba, nbap:]*self.ref_Hq[nb:nba, nbap:])
            pp = np.sum(self.ref_Hq[nba:nbap, nba:nbap]*self.ref_Hq[nba:nbap, nba:nbap])
            pi = 2*np.sum(self.ref_Hq[nba:nbap, nbap:]*self.ref_Hq[nba:nbap, nbap:])
            ii = np.sum(self.ref_Hq[nbap:, nbap:]*self.ref_Hq[nbap:, nbap:])

            wts[:nb,:nb] = 1/ bb if bb != 0 else 0
            wts[:nb, nb:nba] = wts[nb:nba,:nb] = 1/ba if ba != 0 else 0
            wts[:nb, nba:nbap] = wts[nba:nbap, :nb] = 1/ bp  if bp != 0 else 0
            wts[:nb, nbap:] = wts[nbap:,:nb]  = 1/ bi if bi != 0 else 0
            wts[nb:nba, nb:nba] = 1/ aa if aa != 0 else 0
            wts[nb:nba, nba:nbap] = wts[nba:nbap, nb:nba] =  1/ ap if ap != 0 else 0
            wts[nb:nba, nbap:] = wts[nbap:, nb:nba]  = 1/ ai if ai != 0 else 0
            wts[nba:nbap, nba:nbap] =  1/ pp if pp != 0 else 0
            wts[nba:nbap, nbap:]  = wts[nbap:, nba:nbap] =  1/ pi if pi != 0 else 0
            wts[nbap:, nbap:]  = 1/ ii if ii != 0 else 0

            if self.hess_normalize_type == 2:
                # put more weights on diagonal elements
                wts[:nb,:nb] *= 100
                wts[nb:nba, nb:nba] *= 100
                wts[nba:nbap, nba:nbap] *= 100
                wts[nbap:, nbap:] *= 100
            self.wts = wts.flatten()
        
        # for off diagonals, divide by the len(H)
        elif self.hess_normalize_type == 3:
            wts = np.ones((Hq_size, Hq_size))
            for i in range(Hq_size):
                for j in range(Hq_size):
                    if i != j: 
                        wts[i][j] = 1 / (Hq_size)
            self.wts = wts.flatten()

        else: 
            self.wts = np.ones(len(self.ref_Hq_flat))
        # normalize weights
        self.wts /= np.sum(self.wts)

    def indicate(self):
        """ Print qualitative indicator. """
        # if self.reassign == 'overlap' : count_assignment(self.c2r)
        banner = "Hessian"
        headings = ["Diagonal", "Reference", "Calculated", "Difference"]
        data = OrderedDict([(i, ["%.4f" % self.ref_Hq.diagonal()[i], "%.4f" % self.Hq.diagonal()[i], "%.4f" % (self.Hq.diagonal()[i] - self.ref_Hq.diagonal()[i])]) for i in range(len(self.ref_Hq))])
        self.printcool_table(data, headings, banner)
        return

    def hessian_driver(self):
        if hasattr(self, 'engine') and hasattr(self.engine, 'hessian'):
            if self.optimize_geometry == 1: 
                return self.engine.hessian()
            else:
                return self.engine.hessian(optimize=False)
        else:
            logger.error('Internal coordinate hessian calculation not supported, try using a different engine\n')
            raise NotImplementedError

    def vib_overlap(self, v1, v2): # HJ: copied from vibration.py
        """
        Calculate overlap between vibrational modes for two Cartesian displacements.
        Parameters
        ----------
        v1, v2 : np.ndarray
            The two sets of Cartesian displacements to compute overlap for,
            3*N_atoms values each.
        Returns
        -------
        float
            Overlap between mass-weighted eigenvectors
        """
        sqrtm = np.sqrt(np.array(self.engine.AtomLists['Mass']))
        v1m = v1.copy()
        v1m *= sqrtm[:, np.newaxis]
        v1m /= np.linalg.norm(v1m)
        v2m = v2.copy()
        v2m *= sqrtm[:, np.newaxis]
        v2m /= np.linalg.norm(v2m)
        return np.abs(np.dot(v1m.flatten(), v2m.flatten()))

    def converting_to_int_vec(self, xyz, dx):
        dx = np.array(dx).flatten()
        Bmat = self.IC.wilsonB(xyz)
        dq = multi_dot([Bmat,dx])
        return dq

    def calc_int_normal_mode(self, xyz, cart_normal_mode):                
        ninternals_eff= len([ic for ic in self.IC.Internals if isinstance(ic,(Distance, Angle, Dihedral, OutOfPlane))])
        int_normal_mode = []
        for idx, vec in enumerate(cart_normal_mode): 
            # convert cartesian coordinates displacement to internal coordinates
            dq = self.converting_to_int_vec(xyz, vec)
            int_normal_mode.append(dq[:ninternals_eff]) # disregard Translations and Rotations
        return np.array(int_normal_mode)

    def get(self, mvals, AGrad=False, AHess=False):
        """ Evaluate objective function. """
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}
        def compute(mvals_):
            self.FF.make(mvals_)
            Xx, Gx, Hx, freqs, normal_modes, M_opt = self.hessian_driver()   
            # convert into internal hessian 
            Xx *= 1/ Bohr2nm
            Gx *= Bohr2nm/ Hartree2kJmol
            Hx *= Bohr2nm**2/ Hartree2kJmol
            Hq = self.IC.calcHess(Xx, Gx, Hx)
            compute.Hq_flat = Hq.flatten()
            compute.freqs = freqs  
            compute.normal_modes = normal_modes
            compute.M_opt = M_opt
            diff = Hq - self.ref_Hq

            return (np.sqrt(self.wts)/self.denom) * (compute.Hq_flat - self.ref_Hq_flat)

        V = compute(mvals)
        Answer['X'] = np.dot(V,V) * len(compute.freqs) # HJ: len(compute.freqs) is multiplied to match the scale of X2 with vib freq target X2
        # compute gradients and hessian
        dV = np.zeros((self.FF.np,len(V)))
        if AGrad or AHess:
            for p in self.pgrad:
                dV[p,:], _ = f12d3p(fdwrap(compute, mvals, p), h = self.h, f0 = V)

        for p in self.pgrad:
            Answer['G'][p] = 2*np.dot(V, dV[p,:]) * len(compute.freqs)
            for q in self.pgrad:
                Answer['H'][p,q] = 2*np.dot(dV[p,:], dV[q,:]) * len(compute.freqs)
                    
        if not in_fd():
            self.Hq_flat = compute.Hq_flat
            self.Hq = self.Hq_flat.reshape(self.ref_Hq.shape)
            self.objective = Answer['X']
            self.FF.make(mvals)
        
        if self.writelevel > 0:
            # 1. write HessianCompare.txt
            hessian_comparison = np.array([
                self.ref_Hq_flat,
                compute.Hq_flat,
                compute.Hq_flat - self.ref_Hq_flat,
                np.sqrt(self.wts)/self.denom
            ]).T
            np.savetxt("HessianCompare.txt", hessian_comparison, header="%11s  %12s  %12s  %12s" % ("QMHessian", "MMHessian", "Delta(MM-QM)", "Weight"), fmt="% 12.6e")
                    
            # 2. rearrange the MM properties and draw scatter plot of vibrational frequencies and overlap matrix of normal modes 
            plt.switch_backend('agg')
            fig, axs = plt.subplots(1,2, figsize=(10,6))

            # rearrange MM vibrational frequencies using overlap between normal modes in redundant internal coordinates 
            ref_int_normal_modes = self.calc_int_normal_mode(self.ref_xyz, self.ref_eigvecs)
            int_normal_modes = self.calc_int_normal_mode(np.array(compute.M_opt.xyzs[0]), compute.normal_modes)
            a = np.array([[(1.0-np.abs(np.dot(v1/np.linalg.norm(v1),v2/np.linalg.norm(v2)))) for v2 in int_normal_modes] for v1 in ref_int_normal_modes]) 
            row, c2r = optimize.linear_sum_assignment(a)
            # old arrangement method, which uses overlap between mass weighted vibrational modes in cartesian coordinates
            # a = np.array([[(1.0-self.vib_overlap(v1, v2)) for v2 in compute.normal_modes] for v1 in self.ref_eigvecs])
            # row, c2r = optimize.linear_sum_assignment(a)

            freqs_rearr = compute.freqs[c2r]
            normal_modes_rearr = compute.normal_modes[c2r]
            
            # 3. Save rearranged frequencies and normal modes into a file for post-analysis
            with open('mm_vdata.txt', 'w') as outfile: 
                outfile.writelines('%s\n' % line for line in compute.M_opt.write_xyz([0]))
                outfile.write('\n')
                for freq, normal_mode in zip(freqs_rearr, normal_modes_rearr):
                    outfile.write(f'{freq}\n')
                    for nx, ny, nz in normal_mode:
                        outfile.write(f'{nx:13.4f} {ny:13.4f} {nz:13.4f}\n')
                    outfile.write('\n')
            outfile.close()

            overlap_matrix = np.array([[(self.vib_overlap(v1, v2)) for v2 in normal_modes_rearr] for v1 in self.ref_eigvecs])
            qm_overlap_matrix = np.array([[(self.vib_overlap(v1, v2)) for v2 in self.ref_eigvecs] for v1 in self.ref_eigvecs])

            axs[0].scatter(self.ref_eigvals, freqs_rearr, label='MM vibrational frequencies(rearr.)')
            axs[0].plot(self.ref_eigvals,self.ref_eigvals, 'k-')
            axs[0].legend()
            axs[0].set_xlabel(r'QM vibrational frequency ($cm^{-1}$)')
            axs[0].set_ylabel(r'MM vibrational frequency ($cm^{-1}$)')
            mae = np.sum(np.abs(self.ref_eigvals - freqs_rearr))/ len(self.ref_eigvals) # 
            axs[0].set_title(f'QM vs. MM vibrational frequencies\n MAE= {mae:.2f}')
            x0,x1 = axs[0].get_xlim()
            y0,y1 = axs[0].get_ylim()
            axs[0].set_aspect((x1-x0)/(y1-y0))
    
            # move ax x axis to top 
            axs[1].xaxis.tick_top() 
            # move ax x ticks inside 
            axs[1].tick_params(axis="y", direction='in')
            axs[1].tick_params(axis="x", direction='in')
            # draw matrix
            im = axs[1].imshow(overlap_matrix, cmap= 'OrRd', vmin=0,vmax=1)
            # colorbar
            aspect = 20
            pad_fraction = 0.5
            divider = make_axes_locatable(axs[1])
            width = axes_size.AxesY(axs[1], aspect=1./aspect)
            pad = axes_size.Fraction(pad_fraction, width)
            cax = divider.append_axes("right", size=width, pad=pad)
            cax.yaxis.tick_right()
            cax.xaxis.set_visible(False) 
            plt.colorbar(im, cax=cax) 
            corr_coef = cal_corr_coef(overlap_matrix)
            err = np.linalg.norm(qm_overlap_matrix - overlap_matrix)/np.linalg.norm(qm_overlap_matrix) # measure of error in matrix (Relative error)
            axs[1].set_title(f'QM vs. MM normal modes\n Correlation coef. ={corr_coef:.4f}, Error={err:.4f}')

            # # move ax x axis to top 
            # axs[2].xaxis.tick_top() 
            # # move ax x ticks inside 
            # axs[2].tick_params(axis="y", direction='in')
            # axs[2].tick_params(axis="x", direction='in')
            # # draw matrix
            # im = axs[2].imshow(qm_overlap_matrix, cmap= 'OrRd', vmin=0,vmax=1)
            # # colorbar
            # aspect = 20
            # pad_fraction = 0.5
            # divider = make_axes_locatable(axs[2])
            # width = axes_size.AxesY(axs[2], aspect=1./aspect)
            # pad = axes_size.Fraction(pad_fraction, width)
            # cax = divider.append_axes("right", size=width, pad=pad)
            # cax.yaxis.tick_right()
            # cax.xaxis.set_visible(False) 
            # plt.colorbar(im, cax=cax) 
            # axs[2].set_title(f'(QM normal modes for reference)')
 
            plt.tight_layout() 
            plt.subplots_adjust(top=0.85)
            fig.suptitle('Hessian: iteration %i\nSystem: %s' % (Counter(), self.name))
            fig.savefig('vibfreq_scatter_plot_n_overlap_matrix.pdf')

            # # draw qm and mm normal mode overlay
            # fig, axs = plt.subplots(len(normal_modes_rearr), 1, figsize=(4, 4*len(normal_modes_rearr)), subplot_kw={'projection':'3d'})
            # def render_normal_modes(elem, xyz, eigvecs, color, qm=False, ref_eigvals=None, eigvals_rearr=None):
            #     for idx, eigvec in enumerate(eigvecs):
            #         x, y, z = xyz.T
            #         u, v, w = eigvec.T *5
            #         origin = np.array([x, y, z])
            #         axs[idx].quiver(*origin, u, v, w, color=color)
                    
            #         axs[idx].set_xlabel('x')
            #         axs[idx].set_ylabel('y')
            #         axs[idx].set_zlabel('z')    
            #         if qm:
            #             axs[idx].set_title(f'normal mode #{idx} (blue:QM({ref_eigvals[idx]:.2f}), red:MM({eigvals_rearr[idx]:.2f}))')
            #             axs[idx].scatter(x, y, z, color='black', s=30)
            #             axs[idx].set_xlim(min(u+x), max(u+x))
            #             axs[idx].set_ylim(min(v+y), max(v+y))
            #             axs[idx].set_zlim(min(w+z), max(w+z))
            #             for i, elm in enumerate(elem):
            #                 axs[idx].text(x[i], y[i], z[i],elm)

            # render_normal_modes(compute.M_opt.elem, self.ref_xyz, self.ref_eigvecs, 'blue', qm=True, ref_eigvals=self.ref_eigvals, eigvals_rearr=freqs_rearr)
            # render_normal_modes(compute.M_opt.elem, np.array(compute.M_opt.xyzs[0]), normal_modes_rearr, 'red')
            
            # plt.tight_layout()
            # plt.savefig('mm_vdata.pdf') 

            return Answer

def cal_corr_coef(A): 
    # equations from https://math.stackexchange.com/a/1393907
    size = len(A)
    j = np.ones(size)
    r = np.array(range(1,size+1))
    r2 = r*r 
    n  = np.dot(np.dot(j, A),j.T)
    sumx=np.dot(np.dot(r, A),j.T)
    sumy=np.dot(np.dot(j, A),r.T)
    sumx2=np.dot(np.dot(r2, A),j.T)
    sumy2=np.dot(np.dot(j, A),r2.T)
    sumxy=np.dot(np.dot(r, A),r.T)
    r = (n*sumxy - sumx*sumy)/(np.sqrt(n*sumx2 - (sumx)**2)* np.sqrt(n*sumy2 - (sumy)**2))
    return r