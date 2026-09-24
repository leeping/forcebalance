""" Summarize a completed run of both cases: parameter recovery table + RDF comparison plot.

Reads each case's optimize.out (for the fitted sigma) and the last iteration's
npt_result.p (for the fitted g(r) curve), compares against results/rdf_truth_*.dat,
and writes results/summary.csv and results/rdf_comparison.png.

Usage (after both case_peak30/optimize.in and case_peak38/optimize.in have been
run to completion with ForceBalance.py):
    python summarize.py
"""
import glob
import os
import re
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forcebalance.nifty import lp_load

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
SIGMA0_ANG = 3.4
CASES = [('peak30', 3.0), ('peak38', 3.8)]
LJMIN_FACTOR = 2 ** (1.0 / 6.0)


def read_fitted_sigma(case_dir):
    with open(os.path.join(case_dir, 'optimize.out')) as f:
        text = f.read()
    block = text.rsplit('Final physical parameters:', 1)[-1]
    m = re.search(r'\[\s*([\-0-9.eE]+)\s*\]', block)
    return float(m.group(1)) * 10.0  # nm -> Angstrom


def read_fitted_gr(case_dir):
    iters = sorted(glob.glob(os.path.join(case_dir, 'optimize.tmp', 'LJ_RDF', 'iter_*')))
    last_iter = iters[-1]
    pt_dir = glob.glob(os.path.join(last_iter, '*K-*atm'))[0]
    d = lp_load(os.path.join(pt_dir, 'npt_result.p'))
    rdf_data = d[-1]           # RDF_data: one list of snapshot g(r) arrays per RDF target
    snapshots = rdf_data[0]    # single RDF target ("name Ar&name Ar")
    return np.mean(snapshots, axis=0)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (label, sigma_true) in zip(axes, CASES):
        case_dir = os.path.join(HERE, f'case_{label}')
        target = np.loadtxt(os.path.join(RESULTS_DIR, f'rdf_truth_{label}.dat'))
        r, g_target = target[:, 0], target[:, 1]
        g_fit = read_fitted_gr(case_dir)
        sigma_opt = read_fitted_sigma(case_dir)

        peak_target = r[np.argmax(g_target)]
        peak_fit = r[np.argmax(g_fit)]
        ljmin_opt = LJMIN_FACTOR * sigma_opt
        error_pct = 100 * abs(sigma_opt - sigma_true) / sigma_true

        rows.append(dict(target=label, sigma_true_ang=sigma_true, sigma0_ang=SIGMA0_ANG,
                          sigma_fit_ang=round(sigma_opt, 4), error_pct=round(error_pct, 3),
                          rdf_peak_target_ang=round(peak_target, 3), rdf_peak_fit_ang=round(peak_fit, 3),
                          lj_min_fit_ang=round(ljmin_opt, 3)))

        ax.plot(r, g_target, color='#2a78d6', lw=2, label='target g(r)')
        ax.plot(r, g_fit, color='#eb6834', lw=2, ls='--', label='fitted g(r)')
        ax.axvline(ljmin_opt, color='#8a8368', lw=1, ls=':', label=r'$2^{1/6}\sigma_{fit}$')
        ax.set_title(f'true $\\sigma$={sigma_true} Å  →  fitted {sigma_opt:.3f} Å')
        ax.set_xlabel('r (Å)')
        ax.set_ylabel('g(r)')
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'rdf_comparison.png'), dpi=150)
    print(f"Wrote {os.path.join(RESULTS_DIR, 'rdf_comparison.png')}")

    csv_path = os.path.join(RESULTS_DIR, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
