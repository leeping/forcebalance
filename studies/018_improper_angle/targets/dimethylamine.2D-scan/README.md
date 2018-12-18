This folder contains a simple procedure for generating the "target" from a geomeTRIC scan-final.xyz file.

1) First copy the scan-final.xyz file into the current folder as "all.xyz".
2) Run make-qdata.py.  This parses out the QM energies which were stored in all.xyz as comments and saves as "qdata.txt".
3) Add .mol2 files for each molecule in the system.
4) Add .pdb file containing the whole system.  (Should be the same atoms as all.xyz, but with atom names and residue names).
   LPW created this .pdb file by running "filecnv.py mol1.mol2 mol1.pdb", where filecnv.py uses ForceBalance's file format conversion tool.
   If you created a .pdb file along with the .mol2 file then you should probably use the one you created.
5) Run make-labels.py to look at which parameters are being loaded from the force field.
6) Add "parameterize" attributes to forcefield/smirnoff99Frosst.xml that matches the output from make-labels.py
   You should only add parameters that are "relevant" to the optimization, i.e. they should correspond to the degrees of freedom being scanned.
