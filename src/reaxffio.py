#!/usr/bin/env python

from collections import namedtuple
import itertools

#===============================================================#
#| This code is used for interpreting a ReaxFF parameter file. |#
#===============================================================#
# 
# Shorter ReaxFF parameter names are taken from source code
#
# These parameters are explained following the "combustion paper"
# supporting information by Chenoweth, van Duin and Goddard; JPCA, 2008.
#
# First, uncorrected bond orders are calculated from interatomic distances.
# 
# ro(sigma, pi, pipi) : equilibrium bond lengths for sigma, pi, and second pi bond
# p_bo1-p_bo6 : per-bond parameters for calculating bond order (in exponential)
# 
# With the uncorrected bond orders, we can calculate an uncorrected
# overcoordination. The uncorrected overcoordination / bond orders are
# used to calculate several functions f1-f5 as defined in Equations 4b-4f.
# These functions are then multiplied onto the uncorrected bond orders
# to get corrected bond orders.
# 
# p_boc1-p_boc5 : parameters for correcting bond orders and overcoordination
#             in Equations 4a-4f. boc1 and boc2 are global parameters.
#             boc3-boc5 are per-atom parameters.
# Val(boc) : valency of the atom. This is usually an integer (one for H, 4 for C/O).
#
# With corrected bond orders we now have a corrected overcoordination.
# Then the bond energy is calculated in Equation 6.
# 
# p_be1, p_be2 : per-atom bond energy parameters that go into an exponential
# De(sigma, pi, pipi) : per-bond dissociation energies of the "sub-bonds"
#
# An energy penalty is added for deviations from the number of ideal lone pairs.
# In equation 7 a "lone pair overcoordination" is defined and used to calculate
# the "number of lone pairs" in equation 8. Then the "excess lone pairs" are
# calculated in equation 9 and used to make an energy penalty in equation 10.
#
# Val(e) : number of valence electrons in the element (one for H, 4 for C, 6 for O)
# p_lp1 : global parameter for calculating number of lone pairs
# p_lp2 : per-atom parameter for multiplicative term for lone pair energy penalty
#
# Now overcoordination is calculated. If the number of lone pairs is different
# from the ideal, then overcoordination is decreased.
#
# p_ovun1: per-bond parameter for overcoordination energy
# p_ovun2: per-atom parameter for overcoordination energy
# p_ovun3, p_ovun4: global parameters used to calculate the correction factor
# for overcoordination when the number of lone pairs changes from the ideal
#
# Now undercoordination stabilization is calculated. This is only used when there a pair
# of undercoordinated atoms have pi-bonding character.
#
# p_ovun2 : per-atom parameter makes an appearance again
# p_ovun5 : per-atom parameter in undercoordination (force constant)
# p_ovun6, p_ovun7, p_ovun8 : global parameters in undercoordination
# 
# Now angle energy is calculated. We first calculate switching functions f7 and f8
# which depend on bond order and overcoordination. This is used to calculate
# the angle energy which is called E_val. (Note: there is no switching function f6)
# 
# p_val1, p_val2 : per-angle parameters that appear in energy expression
#              (val1 is a force constant)
# p_val3 : per-atom parameter used to calculate switching function f7
# p_val4 : per-angle parameter used to calculate switching function f7
# p_val5 : per-angle parameter used to calculate switching function f8
# -- Note that there's a typo in the force field file; val7 appears twice.
# -- To correct this typo we had to go into the source code.
# p_val6 : global parameter used to calculate switching function f8
# p_val7 : per-angle parameter used to calculate switching function f8
# p_val8-p_val10 : global parameters used to calculate sum of bond orders
# Theta_oo : Constant that is subtracted from pi in calculating Theta_o
# 
# Now energy penalty is added to atoms involved in two double bonds like allene.
# A switching function f9 is calculated from overcoordination.
#
# p_pen1 : per-angle parameter (force constant)
# p_pen2 : per-angle parameter used to calculate penalty energy
# p_pen3, p_pen4: global parameters used to calculate switching function f9
#
# Now a three-body conjugation term is calculated which was added later to
# account for -NO2-groups. This is new in 2006. coa must stand for "conjugation-angle"
# 
# p_coa1 : per-angle parameter (force constant)
# p_coa2-p_coa4 : global parameters
#
# Now torsion energy terms are added. Switching functions f10 and f11 are calculated
#
# p_tor1 : per-torsion parameter
# p_tor2 : global parameters used to calculate switching function f10
# p_tor3, p_tor4 : global parameters used to calculate switching function f11
# V1, V2, V3 : per-torsion parameters (force constants; height of torsion barriers)
#
# Now four-body conjugation terms are added. These stabilize the energy.
# Also switching function f12 is calculated.
# cot must stand for "conjugation torsion"
#
# p_cot1 : per-torsion parameter (force constant)
# p_cot2 : global parameter (exponent; inverse B.O.)
#
# Now hydrogen bond interactions are calculated. These are in a separate section
# from the angle terms.
# r(hb) : per-hydrogen-bond parameter (equilibrium distance)
# p_hb1 : per-hydrogen-bond parameter (force constant)
# p_hb2 : per-hydrogen-bond parameter (B.O. decay constant)
# p_hb3 : per-hydrogen-bond parameter (dimensionless distance decay constant)
#
# A term is added to penalize the C2 molecule.
#
# kc2 : global parameter (force constant); seems to be unused
# 
# Triple bonded terms are added to describe carbon monoxide.
#
# p_trip1 : global parameter (force constant)
# p_trip2 : global parameter (B.O.-squared decay constant)
# p_trip3 : global parameter (overcoordination growth constant)
# p_trip4 : global parameter (B.O. decay constant)
#
# van der Waals interactions are tapered off at long distances.
# swa, swb : taper radii for van der Waals
# rvdW : per-atom parameter used in calculating vdW energy
# alfa : per-atom parameter (with parameter?) used in calculating vdW energy
# vdW1 : global parameter (exponent of distance) used to calculate function f13
# gammaw : per-atom parameter (inverse distance) used to calculate function f13
# 
# Note the "off-diagonal" terms which are like explicit pairwise terms
# that override "combining rules".
#
# Coulomb interactions are calculated using the EEM method. 
# gamma : per-atom parameter (inverse distance) for damping Coulomb interactions

GlobalParamNames = ["p_boc1", "p_boc2", "p_coa2", "p_trip4", "p_trip3", "p_kc2","p_ovun6", "p_trip2",
                    "p_ovun7", "p_ovun8", "p_trip1", "swa", "swb", "unused1","p_val7", "p_lp1", "p_val9",
                    "p_val10", "unused2", "p_pen2", "p_pen3", "p_pen4", "unused3", "p_tor2", "p_tor3", "p_tor4",
                    "unused4","p_cot2", "p_vdW1", "cutoff", "p_coa4", "p_ovun4", "p_ovun3", "p_val8",
                    "unused5", "unused6", "unused7", "unused8", "p_coa3"]

# The atom section starts with an integer for the number of atoms, followed by blocks of
# parameters for each atom.
# Each sub-list indicates the parameter names on the lines in an atom-block
AtomParamNames = [["atomID", "ro_sigma", "Val", "atom_mass", "Rvdw", "Dij", "gamma", "ro_pi", "Val_e"],
                  ["alfa", "gamma_w", "Val_angle", "ovun5", "unused1", "chiEEM", "etaEEM", "unused2"],
                  ["ro_pipi", "p_lp2", "Heat_increment", "p_boc4", "p_boc3", "p_boc5", "unused3", "unused4"],
                  ["p_ovun2", "p_val3", "unused5", "Val_boc", "p_val5", "unused6", "unused7", "unused8"]]

# The bonds section starts with an integer for the number of bonds, followed by blocks of
# parameters for each bond. Each block starts with integer indices of the atom types
# that are involved (starting from 1).
BondParamNames = [["at1", "at2", "De_sigma", "De_pi", "De_pipi", "p_be1", "p_bo5", "13corr", "unused1", "p_bo6"],
                  ["p_ovun1", "p_be2", "p_bo3", "p_bo4", "unused2", "p_bo1", "p_bo2", "unused3"]]

# Format follows the bonds section
OffdiagParamNames = ["Dij", "RvdW", "alfa", "ro_sigma", "ro_pi", "ro_pipi"]

# Three atom indices are used in angles
AngleParamNames = ["Theta_oo", "p_val1", "p_val2", "p_coa1", "p_val7", "p_pen1", "p_val4"]

# Four atom indices are used in torsions
TorsionParamNames = ["V1", "V2", "V3", "p_tor1", "p_cot1", "unused1", "unused2"]

# Three atom indices are used in hydrogen bonds
HbondParamNames = ["r_hb", "p_hb1", "p_hb2", "p_hb3"]

UnusedParamNames = {"global": [i for i in GlobalParamNames if 'unused' in i] + ["cutoff"],
                    "atom": ([i for i in itertools.chain(*AtomParamNames) if 'unused' in i] +
                             ["atomID", "Val", "atom_mass", "Val_e", "Val_angle", "Heat_increment", "Val_boc"]),
                    "bond": [i for i in itertools.chain(*BondParamNames) if 'unused' in i] + ["13corr"],
                    "offdiag": [i for i in OffdiagParamNames if 'unused' in i],
                    "angle": [i for i in AngleParamNames if 'unused' in i],
                    "torsion": [i for i in TorsionParamNames if 'unused' in i],
                    "hbond": [i for i in HbondParamNames if 'unused' in i]}

class ReaxFF_Reader(BaseReader):
    """Finite state machine for parsing ReaxFF force field files.

    This class is instantiated when we begin to read in a file.  The
    feed(line) method updates the state of the machine, informing it
    of the current interaction type.  Using this information we can
    look up the interaction type and parameter type for building the
    parameter ID.
    """
    
    def __init__(self,fnm):
        super(ReaxFF_Reader,self).__init__(fnm)
        # ## The parameter dictionary (defined in this file)
        # self.pdict  = pdict
        ## The atoms that are defined in the force field
        self.atoms   = []
        ## List of the atoms involved in the current interaction
        self.involved = []
        ## The number of global parameters (I think this is almost always 39)
        self.nglob = 0
        ## The number of atom types
        self.natom = 0
        ## The number of bond types
        
        ## The global parameter that we're currently on
        self.globn = -1
        ## The number of the global parameter that we're currently on
        self.paramnr = defaultdict(int)
        ## The section of the force field that we're currently on
        self.section_length = OrderedDict([("global_header", 2), ("global_params", None),
                                     ("atom_header", 4), ("atom_params", None),
                                     ("bond_header", 2), ("bond_params", None),
                                     ("offdiag_header", 1), ("offdiag_params", None),
                                     ("angle_header", 1), ("angle_params", None),
                                     ("torsion_header", 1), ("torsion_params", None),
                                     ("hbond_header", 1), ("hbond_params", None)])
        ## The current section that we're in
        self.this_section = None
        ## The line number within a section
        self.line_section = 0
        ## The line number within a block
        self.blockline = 0
                  
    def feed(self, line):
        """ Given a line, determine the interaction type and the atoms involved (the suffix).

        The format of the ReaxFF file is like this:
        
        Comment line
        N_global_params comments...
        (followed by N_global_params lines of global parameters)
        N_atom_types ! Nr of atoms; atomID;ro(sigma); Val;atom mass;Rvdw;Dij;gamma;ro(pi);Val(e)
            alfa;gamma(w);Val(angle);p(ovun5);n.u.;chiEEM;etaEEM;n.u.
            ro(pipi);p(lp2);Heat increment;p(boc4);p(boc3);p(boc5),n.u.;n.u.
            p(ovun2);p(val3);n.u.;Val(boc);p(val5);n.u.;n.u.;n.u.
        (The above are just comments but the ReaxFF code reads in three blank lines.)
        Then N_atom_types blocks of these:
        C    1.3825   4.0000  12.0000   1.9133   0.1853   0.9000   1.1359   4.0000
             9.7602   2.1346   4.0000  33.2433  79.5548   5.8678   7.0000   0.0000
             1.2104   0.0000 199.0303   8.6991  34.7289  13.3894   0.8563   0.0000
            -2.8983   2.5000   1.0564   4.0000   2.9663   0.0000   0.0000   0.0000
        N_bond_types ! Nr of bonds; at1;at2;De(sigma);De(pi);De(pipi);p(be1);p(bo5);13corr;n.u.;p(bo6),p(ovun1)
                      p(be2);p(bo3);p(bo4);n.u.;p(bo1);p(bo2)
        Then N_bond_types blocks of these:
        1  1 156.5953 100.0397  80.0000  -0.8157  -0.4591   1.0000  37.7369   0.4235
             0.4527  -0.1000   9.2605   1.0000  -0.0750   6.8316   1.0000   0.0000
        This is followed by off-diagonal terms, angle terms, torsions and finally hydrogen bonds.

        The parameter names should be something like this:
        Parameter_Class / Atoms.Involved / Parameter_Type
        Global/p_ovun7
        Atom/p_boc4/C
        Offdiag/C.O/alfa
        """
        
        s          = line.split()
        self.ln   += 1

        section_end = 0
        for sname in self.section_length:
            section_end += self.section_length[sname]
            if self.ln <= section_end: break

        if sname != self.this_section:
            self.line_section = 1
        else:
            self.line_section += 1
            
        self.this_section = sname

        if self.this_section == "global_header":
            if self.line_section == 2:
                self.nglob = int(s[0])
                self.section_length["globals"] = int(s[0])
        elif self.this_section == "globals":
            self.field_names = [GlobalParamNames[line_section-1]]
        elif self.this_section == "atom_header":
            if self.line_section == 1:
                self.natom = int(s[0])
                self.section_length["atoms"] = self.natom*4
        elif self.this_section == "atoms":
            self.field_names = AtomParamNames[(line_section-1)%4]
        elif self.this_section == "bond_header":
            if self.line_section == 1:
                self.nbond = int(s[0])
                self.section_length["bonds"] = self.nbond*2
        elif self.this_section == "bonds":
            self.field_names = BondParamNames[(line_section-1)%2]
        elif self.this_section == "offdiag_header":
            self.noffdiag = int(s[0])
            self.section_length(["offdiag"]) = self.noffdiag
        elif self.this_section == "offdiag":
            self.field_names = OffdiagParamNames
        elif self.this_section == "angle_header":
            self.nangle = int(s[0])
            self.section_length(["angle"]) = self.nangle
        elif self.this_section == "angle":
            self.field_names = AngleParamNames
        elif self.this_section == "torsion_header":
            self.ntorsion = int(s[0])
            self.section_length(["torsion"]) = self.ntorsion
        elif self.this_section == "torsion":
            self.field_names = TorsionParamNames
        elif self.this_section == "hbond_header":
            self.nhbond = int(s[0])
            self.section_length(["hbond"]) = self.nhbond
        elif self.this_section == "hbond":
            self.field_names = HbondParamNames
        
        print self.section, line,
