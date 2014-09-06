#!/usr/bin/env python

from mslib import MSMS
from forcebalance.nifty import lp_load, lp_dump
import numpy as np
import os

# Designed to be called from GenerateQMData.py
# I wrote this because MSMS seems to have a memory leak

xyz, radii, density = lp_load(open('msms_input.p'))
MS = MSMS(coords = list(xyz), radii = radii)
MS.compute(density=density)
vfloat, vint, tri = MS.getTriangles()
with open(os.path.join('msms_output.p'), 'w') as f: lp_dump(vfloat, f)
