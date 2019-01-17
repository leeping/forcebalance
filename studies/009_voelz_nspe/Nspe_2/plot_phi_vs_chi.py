from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
import os, sys
import numpy as np

import matplotlib.pyplot as plt
from scipy import loadtxt

data = loadtxt('energies.dat')
chi, phi, E = data[:,1], data[:,2], data[:,3]
chi_bins, phi_bins = int(chi/15), int(phi/30)

nchi = int(360/15) + 1
nphi = int(360/30) + 1
E_2D = np.max(E)*np.ones( (nphi, nchi), dtype=np.float )
for i in range(len(chi)):
    E_2D[phi_bins[i], chi_bins[i]] = E[i]

print(E_2D)

plt.figure()
#plt.pcolor(E_2D.transpose())
plt.contour(E_2D)

plt.show()

