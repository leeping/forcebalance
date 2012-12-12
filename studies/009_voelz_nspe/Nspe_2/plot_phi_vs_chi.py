import os, sys
import numpy as np

import matplotlib.pyplot as plt
from scipy import loadtxt

data = loadtxt('energies.dat')
chi, phi, E = data[:,1], data[:,2], data[:,3]
chi_bins, phi_bins = chi/15, phi/30

nchi = 360/15 + 1
nphi = 360/30 + 1
E_2D = np.max(E)*np.ones( (nphi, nchi), dtype=np.float )
for i in range(len(chi)):
    E_2D[phi_bins[i], chi_bins[i]] = E[i]

print E_2D

plt.figure()
#plt.pcolor(E_2D.transpose())
plt.contour(E_2D)

plt.show()

