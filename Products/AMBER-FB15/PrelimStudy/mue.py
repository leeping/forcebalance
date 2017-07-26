#!/usr/bin/env python

import numpy as np

QM_MM = np.loadtxt('EnergyCompare.txt')
QM_Wt = np.zeros_like(QM_MM[:,3]) + 1.0
QM_Wt /= np.sum(QM_Wt)

D = QM_MM[:,0]-QM_MM[:,1]
D -= np.mean(D)
D /= 4.184
D = np.abs(D)
print np.dot(D,QM_Wt)
