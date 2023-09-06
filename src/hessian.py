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
from forcebalance.vibration import read_reference_vdata, vib_overlap

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

        from geometric.internal import PrimitiveInternalCoordinates

        m  = Molecule(os.path.join(self.tgtdir, "input.mol2"))
        IC = PrimitiveInternalCoordinates(m)
        self.IC = IC

    def read_reference_data(self): # HJ: copied from vibration.py and modified
        """ Read the reference hessian data from a file. """
        self.ref_Hq_flat = np.loadtxt(self.hfnm)
        Hq_size =int(np.sqrt(len(self.ref_Hq_flat)))
        self.ref_Hq = self.ref_Hq_flat.reshape((Hq_size, Hq_size))

        """ Read the reference vibrational data from a file. """
        self.na, self.ref_xyz, self.ref_eigvals, self.ref_eigvecs = read_reference_vdata(self.vfnm)

        return

    def get_wts(self):

        from geometric.internal import Distance, Angle, Dihedral
        nb = len([ic for ic in self.IC.Internals if isinstance(ic,Distance) ])
        nba = nb + len([ic for ic in self.IC.Internals if isinstance(ic,Angle) ])
        nbap = nba + len([ic for ic in self.IC.Internals if isinstance(ic,Dihedral) ])

        Hq_size =int(np.sqrt(len(self.ref_Hq_flat)))
        if  self.hess_normalize_type == 0 :
            self.wts = np.ones(len(self.ref_Hq_flat))
        else:
            raise NotImplementedError
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
        if hasattr(self, 'engine') and hasattr(self.engine, 'normal_modes'):
            if self.optimize_geometry == 1:
                return self.engine.normal_modes(for_hessian_target=True)
            else:
                return self.engine.normal_modes(optimize=False, for_hessian_target=True)
        else:
            logger.error('Internal coordinate hessian calculation not supported, try using a different engine\n')
            raise NotImplementedError

    def converting_to_int_vec(self, xyz, dx):
        dx = np.array(dx).flatten()
        Bmat = self.IC.wilsonB(xyz)
        dq = multi_dot([Bmat,dx])
        return dq

    def calc_int_normal_mode(self, xyz, cart_normal_mode):
        from geometric.internal import Distance, Angle, Dihedral, OutOfPlane
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

            # 2. rearrange MM vibrational frequencies using overlap between normal modes in redundant internal coordinates
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
                    outfile.write("%13.4f\n" % freq)
                    for nx, ny, nz in normal_mode:
                        outfile.write("%13.4f %13.4f %13.4f\n" % (nx, ny, nz))
                    outfile.write('\n')
            outfile.close()

            # 4. draw a scatter plot of vibrational frequencies and an overlap matrix of normal modes in cartessian coordinates
            draw_vibfreq_scatter_plot_n_overlap_matrix(self.name, self.engine, self.ref_eigvals, self.ref_eigvecs, freqs_rearr, normal_modes_rearr)
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

def draw_normal_modes(elem, ref_xyz, ref_eigvals, ref_eigvecs, mm_xyz, freqs_rearr, normal_modes_rearr):

    import matplotlib.pyplot as plt

    # draw qm and mm normal mode overlay
    fig, axs = plt.subplots(len(normal_modes_rearr), 1, figsize=(4, 4*len(normal_modes_rearr)), subplot_kw={'projection':'3d'})
    def render_normal_modes(elem, xyz, eigvecs, color, qm=False, ref_eigvals=None, eigvals_rearr=None):
        for idx, eigvec in enumerate(eigvecs):
            x, y, z = xyz.T
            u, v, w = eigvec.T *5
            origin = np.array([x, y, z])
            axs[idx].quiver(origin[0], origin[1], origin[2], u, v, w, color=color)

            axs[idx].set_xlabel('x')
            axs[idx].set_ylabel('y')
            axs[idx].set_zlabel('z')
            if qm:
                axs[idx].set_title('normal mode #%i (blue:QM(%.2f), red:MM(%.2f))' % (idx, ref_eigvals[idx], eigvals_rearr[idx]))
                axs[idx].scatter(x, y, z, color='black', s=30)
                axs[idx].set_xlim(min(u+x), max(u+x))
                axs[idx].set_ylim(min(v+y), max(v+y))
                axs[idx].set_zlim(min(w+z), max(w+z))
                for i, elm in enumerate(elem):
                    axs[idx].text(x[i], y[i], z[i],elm)

    render_normal_modes(elem, ref_xyz, ref_eigvecs, 'blue', qm=True, ref_eigvals=ref_eigvals, eigvals_rearr=freqs_rearr)
    render_normal_modes(elem, np.array(mm_xyz), normal_modes_rearr, 'red')

    plt.tight_layout()
    plt.savefig('mm_vdata.pdf')

def draw_vibfreq_scatter_plot_n_overlap_matrix(name, engine, ref_eigvals, ref_eigvecs, freqs_rearr, normal_modes_rearr):

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

    plt.switch_backend('agg')
    fig, axs = plt.subplots(1,2, figsize=(10,6))
    overlap_matrix = np.array([[(vib_overlap(engine, v1, v2)) for v2 in normal_modes_rearr] for v1 in ref_eigvecs])
    qm_overlap_matrix = np.array([[(vib_overlap(engine,v1, v2)) for v2 in ref_eigvecs] for v1 in ref_eigvecs])

    axs[0].scatter(ref_eigvals, freqs_rearr, label='MM vibrational frequencies(rearr.)')
    axs[0].plot(ref_eigvals,ref_eigvals, 'k-')
    axs[0].legend()
    axs[0].set_xlabel(r'QM vibrational frequency ($cm^{-1}$)')
    axs[0].set_ylabel(r'MM vibrational frequency ($cm^{-1}$)')
    mae = np.sum(np.abs(ref_eigvals - freqs_rearr))/ len(ref_eigvals)
    axs[0].set_title('QM vs. MM vibrational frequencies\n MAE= %2f' % mae)
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
    axs[1].set_title('QM vs. MM normal modes\n Correlation coef. =%.4f, Error=%.4f' % (corr_coef, err))

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
    fig.suptitle('Hessian: iteration %i\nSystem: %s' % (Counter(), name))
    fig.savefig('vibfreq_scatter_plot_n_overlap_matrix.pdf')
