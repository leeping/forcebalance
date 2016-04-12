#!/usr/bin/env python

# Very simple file format conversion script, works in 90% of situations
# and good for saving keystrokes.

from forcebalance.molecule import Molecule
from sys import argv

def main():
    topfnm = argv[3] if len(argv) >= 4 else None
    M = Molecule(argv[1], top=topfnm)
    M.write(argv[2])

if __name__ == "__main__":
    main()
