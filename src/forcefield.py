""" @package forcebalance.forcefield

Force field module.

In ForceBalance a 'force field' is built from a set of files containing
physical parameters.  These files can be anything that enter into any
computation - our original program was quite dependent on the GROMACS
force field format, but this program is set up to allow very general
input formats.

We introduce several important concepts:

1) Adjustable parameters are allocated into a vector.

To cast the force field optimization as a math problem, we treat all
of the parameters on equal footing and write them as indices in a parameter vector.

2) A mapping from interaction type to parameter number.

Each element in the parameter vector corresponds to one or more
interaction types.  Whenever we change the parameter vector and
recompute the objective function, this amounts to changing the
physical parameters in the simulations, so we print out new
force field files for external programs.  In addition, when
these programs are computing the objective function we are often
in low-level subroutines that compute terms in the energy and
force.  If we need an analytic derivative of the objective
function, then these subroutines need to know which index of the
parameter vector needs to be modified.

This is done by way of a hash table: for example, when we are
computing a Coulomb interaction between atom 4 and atom 5, we
can build the words 'COUL4' and 'COUL5' and look it up in the
parameter map; this gives us two numbers (say, 10 and 11)
corresponding to the eleventh and twelfth element of the
parameter vector.  Then we can compute the derivatives of
the energy w/r.t. these parameters (in this case, COUL5/rij
and COUL4/rij) and increment these values in the objective function
gradient.

In custom-implemented force fields (see counterpoisematch.py)
the hash table can also be used to look up parameter values for
computation of interactions.  This is probably not the fastest way
to do things, however.

3) Distinction between physical and mathematical parameters.

The optimization algorithm works in a space that is related to, but
not exactly the same as the physical parameter space.  The reasons
for why we do this are:

a) Each parameter has its own physical units.  On the one hand
it's not right to treat different physical units all on the same
footing, so nondimensionalization is desirable.  To make matters
worse, the force field parameters can be small as 1e-8 or as
large as 1e+6 depending on the parameter type.  This means the
elements of the objective function gradient / Hessian have
elements that differ from each other in size by 10+ orders of
magnitude, leading to mathematical instabilities in the optimizer.

b) The parameter space can be constrained, most notably for atomic
partial charges where we don't want to change the overall charge
on a molecule.  Thus we wish to project out certain movements
in the mathematical parameters such that they don't change the physical
parameters.

c) We wish to regularize our optimization so as to avoid changing
our parameters in very insensitive directions (linear dependencies).
However, the sensitivity of the objective function to changes in the
force field depends on the physical units!

For all of these reasons, we introduce a 'transformation matrix'
which maps mathematical parameters onto physical parameters.  The
diagonal elements in this matrix are rescaling factors; they take the
mathematical parameter and magnify it by this constant factor.  The
off-diagonal elements correspond to rotations and other linear
transformations, and currently I just use them to project out the
'increase the net charge' direction in the physical parameter space.

Note that with regularization, these rescaling factors are equivalent
to the widths of prior distributions in a maximum likelihood framework.
Because there is such a correspondence between rescaling factors and
choosing a prior, they need to be chosen carefully.  This is work in
progress.  Another possibility is to sample the width of the priors from
a noninformative distribution -- the hyperprior (we can choose the
Jeffreys prior or something).  This is work in progress.

Right now only GROMACS parameters are supported, but this class is extensible,
we need more modules!

@author Lee-Ping Wang
@date 04/2012

"""

import os
import sys
import numpy as np
from numpy import sin, cos, tan, exp, log, sqrt, pi
from re import match, sub, split
import forcebalance
from forcebalance import gmxio, qchemio, tinkerio, custom_io, openmmio, amberio, psi4io
from forcebalance.finite_difference import in_fd
from forcebalance.nifty import *
from string import count
from copy import deepcopy
try:
    from lxml import etree
except: pass
import traceback
import itertools
from collections import OrderedDict, defaultdict
from forcebalance.output import getLogger
logger = getLogger(__name__)

FF_Extensions = {"itp" : "gmx",
                 "top" : "gmx",
                 "in"  : "qchem",
                 "prm" : "tinker",
                 "gen" : "custom",
                 "xml" : "openmm",
                 "frcmod" : "frcmod",
                 "mol2" : "mol2",
                 "gbs"  : "gbs",
                 "grid" : "grid"
                 }

""" Recognized force field formats. """
FF_IOModules = {"gmx": gmxio.ITP_Reader ,
                "qchem": qchemio.QCIn_Reader ,
                "tinker": tinkerio.Tinker_Reader ,
                "custom": custom_io.Gen_Reader ,
                "openmm" : openmmio.OpenMM_Reader,
                "frcmod" : amberio.FrcMod_Reader,
                "mol2" : amberio.Mol2_Reader,
                "gbs" : psi4io.GBS_Reader,
                "grid" : psi4io.Grid_Reader
                }

def determine_fftype(ffname,verbose=False):
    """ Determine the type of a force field file.  It is possible to
    specify the file type explicitly in the input file using the
    syntax 'force_field.ext:type'.  Otherwise this function will try
    to determine the force field type by extension. """

    fsplit = ffname.split('/')[-1].split(':')
    fftype = None
    if verbose: logger.info("Determining file type of %s ..." % fsplit[0])
    if len(fsplit) == 2:
        if fsplit[1] in FF_IOModules:
            if verbose: logger.info("We're golden! (%s)\n" % fsplit[1])
            fftype = fsplit[1]
        else:
            if verbose: logger.info("\x1b[91m Warning: \x1b[0m %s not in supported types (%s)!\n" % (fsplit[1],', '.join(FF_IOModules.keys())))
    elif len(fsplit) == 1:
        if verbose: logger.info("Guessing from extension (you may specify type with filename:type) ...")
        ffname = fsplit[0]
        ffext = ffname.split('.')[-1]
        if ffext in FF_Extensions:
            guesstype = FF_Extensions[ffext]
            if guesstype in FF_IOModules:
                if verbose: logger.info("guessing %s -> %s!\n" % (ffext, guesstype))
                fftype = guesstype
            else:
                if verbose: logger.info("\x1b[91m Warning: \x1b[0m %s not in supported types (%s)!\n" % (fsplit[0],', '.join(FF_IOModules.keys())))
        else:
            if verbose: logger.info("\x1b[91m Warning: \x1b[0m %s not in supported extensions (%s)!\n" % (ffext,', '.join(FF_Extensions.keys())))
    if fftype is None:
        if verbose: logger.warning("Force field type not determined!\n")
        #sys.exit(1)
    return fftype

# Thanks to tos9 from #python on freenode. :)
class BackedUpDict(dict):
    def __init__(self, backup_dict):
        super(BackedUpDict, self).__init__()
        self.backup_dict = backup_dict
    def __missing__(self, key):
        try:
            return self.backup_dict[self['AtomType']][key]
        except:
            logger.error('The key %s does not exist as an atom attribute or as an atom type attribute!\n' % key)
            raise KeyError

class FF(forcebalance.BaseClass):
    """ Force field class.

    This class contains all methods for force field manipulation.
    To create an instance of this class, an input file is required
    containing the list of force field file names.  Everything else
    inside this class pertaining to force field generation is self-contained.

    For details on force field parsing, see the detailed documentation for addff.

    """
    def __init__(self, options, verbose=True, printopt=True):

        """Instantiation of force field class.

        Many variables here are initialized to zero, but they are filled out by
        methods like addff, rsmake, and mktransmat.

        """
        super(FF, self).__init__(options)
        #======================================#
        # Options that are given by the parser #
        #======================================#
        ## As these options proliferate, the force field class becomes less standalone.
        ## I need to think of a good solution here...
        ## The root directory of the project
        self.set_option(None, None, 'root', os.getcwd())
        ## File names of force fields
        self.set_option(options,'forcefield','fnms')
        ## Directory containing force fields, relative to project directory
        self.set_option(options,'ffdir')
        ## Priors given by the user :)
        self.set_option(options,'priors')
        ## Whether to constrain the charges.
        self.set_option(options,'constrain_charge')
        ## Whether to constrain the charges.
        self.set_option(options,'logarithmic_map')
        ## Switch for AMOEBA direct or mutual.
        self.set_option(options, 'amoeba_pol')
        ## AMOEBA mutual dipole convergence tolerance.
        self.set_option(options, 'amoeba_eps')
        ## Switch for rigid water molecules
        self.set_option(options, 'rigid_water')
        ## Bypass the transformation and use physical parameters directly
        self.set_option(options, 'use_pvals')
        ## Allow duplicate parameter names (internally construct unique names)
        self.set_option(options, 'duplicate_pnames')

        #======================================#
        #     Variables which are set here     #
        #======================================#
        # A lot of these variables are filled out by calling the methods.

        ## The content of all force field files are stored in memory
        self.ffdata       = {}
        self.ffdata_isxml = {}
        ## The mapping of interaction type -> parameter number
        self.map         = {}
        ## The listing of parameter number -> interaction types
        self.plist       = []
        ## A listing of parameter number -> atoms involved
        self.patoms      = []
        ## A list where pfields[i] = [pid,'file',line,field,mult,cmd],
        ## basically a new way to modify force field files; when we modify the
        ## force field file, we go to the specific line/field in a given file
        ## and change the number.
        self.pfields     = []
        ## List of rescaling factors
        self.rs          = []
        ## The transformation matrix for mathematical -> physical parameters
        self.tm          = None
        ## The transpose of the transformation matrix
        self.tmI         = None
        ## Indices to exclude from optimization / Hessian inversion
        self.excision    = None
        ## The total number of parameters
        self.np          = 0
        ## Initial value of physical parameters
        self.pvals0      = []
        ## A dictionary of force field reader classes
        self.Readers           = OrderedDict()
        ## A list of atom names (this is new, for ESP fitting)
        self.atomnames   = []

        # Read the force fields into memory.
        for fnm in self.fnms:
            if verbose:
                logger.info("Reading force field from file: %s\n" % fnm)
            self.addff(fnm)

        ## WORK IN PROGRESS ##
        # This is a dictionary of {'AtomType':{'Mass' : float, 'Charge' : float, 'ParticleType' : string ('A', 'S', or 'D'), 'AtomicNumber' : int}}
        self.FFAtomTypes = OrderedDict()
        for Reader in self.Readers.values():
            for k, v in Reader.AtomTypes.items():
                if k in self.FFAtomTypes:
                    warn_press_key('Trying to insert atomtype %s into the force field, but it is already there' % k)
                self.FFAtomTypes[k] = v
        # This is an ordered dictionary of {'Molecule' : [{'AtomType' : string, 'ResidueNumber' : int, 'ResidueName' : string,
        #                                                  'AtomName' : string, 'ChargeGroup' : int, 'Charge' : float}]}
        # Each value in the dictionary is a list of BackedUpDictionaries.
        # If we query a BackedUpDictionary and the value does not exist,
        # then it will query the backup dictionary using the 'AtomType' value.
        # Thus, if we look up the mass of 'HW1' or 'HW2' in the dictionary, it will
        # return the mass for 'HW' in the AtomTypes dictionary.
        self.FFMolecules = OrderedDict()
        for Reader in self.Readers.values():
            for Molecule, AtomList in Reader.Molecules.items():
                for FFAtom in AtomList:
                    FFAtomWithDefaults = BackedUpDict(self.FFAtomTypes)
                    for k, v in FFAtom.items():
                        FFAtomWithDefaults[k] = v
                    self.FFMolecules.setdefault(Molecule, []).append(FFAtomWithDefaults)

        # Set the initial values of parameter arrays.
        ## Initial value of physical parameters
        self.pvals0 = np.array(self.pvals0)

        # Prepare various components of the class for first use.
        ## Creates plist from map.
        self.list_map()
        if verbose:
            ## Prints the plist to screen.
            bar = printcool("Starting parameter indices, physical values and IDs")
            self.print_map()
            logger.info(bar)
        ## Make the rescaling factors.
        self.rsmake(printfacs=verbose)
        ## Make the transformation matrix.
        self.mktransmat()
        ## Redirection dictionary (experimental).
        self.redirect = {}
        ## Destruction dictionary (experimental).
        self.linedestroy_save = []
        self.prmdestroy_save = []
        self.linedestroy_this = []
        self.prmdestroy_this = []
        ## Print the optimizer options.
        if printopt: printcool_dictionary(self.PrintOptionDict, title="Setup for force field")

    @classmethod
    def fromfile(cls, fnm):
        ffdir = os.path.split(fnm)[0]
        fnm = os.path.split(fnm)[1]
        options = {'forcefield' : [fnm], 'ffdir' : ffdir, 'duplicate_pnames' : True}
        return cls(options, verbose=False, printopt=False)

    def addff(self,ffname,xmlScript=False):
        """ Parse a force field file and add it to the class.

        First, figure out the type of force field file.  This is done
        either by explicitly specifying the type using for example,
        <tt> ffname force_field.xml:openmm </tt> or we figure it out
        by looking at the file extension.

        Next, parse the file.  Currently we support two classes of
        files - text and XML.  The two types are treated very
        differently; for XML we use the parsers in libxml (via the
        python lxml module), and for text files we have our own
        in-house parsing class.  Within text files, there is also a
        specialized GROMACS and TINKER parser as well as a generic
        text parser.

        The job of the parser is to determine the following things:
        1) Read the user-specified selection of parameters being fitted
        2) Build a mapping (dictionary) of <tt> parameter identifier -> index in parameter vector </tt>
        3) Build a list of physical parameter values
        4) Figure out where to replace the parameter values in the force field file when the values are changed
        5) Figure out which parameters need to be repeated or sign-flipped

        Generally speaking, each parameter value in the force field
        file has a <tt> unique parameter identifier <tt>.  The
        identifier consists of three parts - the interaction type, the
        parameter subtype (within that interaction type), and the
        atoms involved.

        --- If XML: ---

        The force field file is read in using the lxml Python module.  Specify
        which parameter you want to fit using by adding a 'parameterize' element
        to the end of the force field XML file, like so.

        @code
        <AmoebaVdwForce type="BUFFERED-14-7">
           <Vdw class="74" sigma="0.2655" epsilon="0.056484" reduction="0.910" parameterize="sigma, epsilon, reduction" />
        @endcode

        In this example, the parameter identifier would look like <tt> Vdw/74/epsilon </tt>.

        --- If GROMACS (.itp) or TINKER (.prm) : ---

        Follow the rules in the ITP_Reader or Tinker_Reader derived
        class.  Read the documentation in the class documentation or
        the 'feed' method to learn more.  In all cases the parameter
        is tagged using <tt> # PRM 3 </tt> (where # denotes a comment,
        the word PRM stays the same, and 3 is the field number starting
        from zero.)

        --- If normal text : ---

        The parameter identifier is simply built using the file name,
        line number, and field.  Thus, the identifier is unique but
        completely noninformative (which is not ideal for our
        purposes, but it works.)

        --- Endif ---

        @warning My program currently assumes that we are only using one MM program per job.
        If we use CHARMM and GROMACS to perform simulations as part of the same TARGET, we will
        get messed up.  Maybe this needs to be fixed in the future, with program prefixes to
        parameters like C_ , G_ .. or simply unit conversions, you get the idea.

        @warning I don't think the multiplier actually works for analytic derivatives unless
        the interaction calculator knows the multiplier as well.  I'm sure I can make this
        work in the future if necessary.

        @param[in] ffname Name of the force field file

        """
        fftype = determine_fftype(ffname)
        ffname = ffname.split(':')[0]

        # Set the Tinker PRM file, which will help for running programs like "analyze".
        if fftype == "tinker":
            if hasattr(self, "tinkerprm"):
                warn_press_key("There should only be one TINKER parameter file")
            else:
                self.tinkerprm = ffname

        # Set the OpenMM XML file, which will help for running OpenMM.
        if fftype == "openmm":
            if hasattr(self, "openmmxml"):
                warn_press_key("There should only be one OpenMM XML file - confused!!")
            else:
                self.openmmxml = ffname

        if fftype == "mol2":
            if hasattr(self, "amber_mol2"):
                warn_press_key("There should only be one .mol2 file - confused!!")
            else:
                self.amber_mol2 = ffname

        if fftype == "frcmod":
            if hasattr(self, "amber_frcmod"):
                warn_press_key("There should only be one .frcmod file - confused!!")
            else:
                self.amber_frcmod = ffname

        # Determine the appropriate parser from the FF_IOModules dictionary.
        # If we can't figure it out, then use the base reader, it ain't so bad. :)
        Reader = FF_IOModules.get(fftype, forcebalance.BaseReader)

        # Open the force field using an absolute path and read its contents into memory.
        absff = os.path.join(self.root,self.ffdir,ffname)

        # Create an instance of the Reader.
        # The reader is essentially a finite state machine that allows us to
        # build the pid.
        self.Readers[ffname] = Reader(ffname)
        if fftype == "openmm":
            ## Read in an XML force field file as an _ElementTree object
            self.ffdata[ffname] = etree.parse(absff)
            self.ffdata_isxml[ffname] = True
            # Process the file
            self.addff_xml(ffname)
        else:
            ## Read in a text force field file as a list of lines
            self.ffdata[ffname] = [line.expandtabs() for line in open(absff).readlines()]
            self.ffdata_isxml[ffname] = False
            # Process the file
            self.addff_txt(ffname, fftype,xmlScript)
        if hasattr(self.Readers[ffname], 'atomnames'):
            if len(self.atomnames) > 0:
                sys.stderr.write('Found more than one force field containing atom names; skipping the second one (%s)\n' % ffname)
            else:
                self.atomnames += self.Readers[ffname].atomnames

    def check_dupes(self, pid, ffname, ln, pfld):
        """ Check to see whether a given parameter ID already exists, and provide an alternate if needed. """
        pid_ = pid

        have_pids = [f[0] for f in self.pfields]

        if pid in have_pids:
            pid0 = pid
            extranum = 0
            dupfnms = [i[1] for i in self.pfields if pid == i[0]]
            duplns  = [i[2] for i in self.pfields if pid == i[0]]
            dupflds = [i[3] for i in self.pfields if pid == i[0]]
            while pid in have_pids:
                pid = "%s%i" % (pid0, extranum)
                extranum += 1
            def warn_or_err(*args):
                if self.duplicate_pnames:
                    logger.warn(*args)
                else:
                    logger.error(*args)
            warn_or_err("Encountered an duplicate parameter ID (%s)\n" % pid_)
            warn_or_err("file %s line %i field %i duplicates:\n"
                        % (os.path.basename(ffname), ln+1, pfld))
            for dupfnm, dupln, dupfld in zip(dupfnms, duplns, dupflds):
                warn_or_err("file %s line %i field %i\n" % (dupfnm, dupln+1, dupfld))
            if self.duplicate_pnames:
                logger.warn("Parameter name has been changed to %s\n" % pid)
            else:
                raise RuntimeError
        return pid

    def addff_txt(self, ffname, fftype, xmlScript):
        """ Parse a text force field and create several important instance variables.

        Each line is processed using the 'feed' method as implemented
        in the reader class.  This essentially allows us to create the
        correct parameter identifier (pid), because the pid comes from
        more than the current line, it also depends on the section that
        we're in.

        When 'PRM' or 'RPT' is encountered, we do several things:
        - Build the parameter identifier and insert it into the map
        - Point to the file name, line number, and field where the parameter may be modified

        Additionally, when 'PRM' is encountered:
        - Store the physical parameter value (this is permanent; it's the original value)
        - Increment the total number of parameters

        When 'RPT' is encountered we don't expand the parameter vector
        because this parameter is a copy of an existing one.  If the
        parameter identifier is preceded by MINUS_, we chop off the
        prefix but remember that the sign needs to be flipped.

        """

        for ln, line in enumerate(self.ffdata[ffname]):
            try:
                self.Readers[ffname].feed(line)
            except:
                logger.warning(traceback.format_exc() + '\n')
                logger.warning("The force field reader crashed when trying to read the following line:\n")
                logger.warning(line + '\n')
                traceback.print_exc()
                warn_press_key("The force field parser got confused!  The traceback and line in question are printed above.")
            sline = self.Readers[ffname].Split(line)

            kwds = list(itertools.chain(*[[i, "/%s" % i] for i in ['PRM', 'PARM', 'RPT', 'EVAL']]))

            marks = OrderedDict()
            for k in kwds:
                if sline.count(k) > 1:
                    logger.error(line)
                    logger.error("The above line contains multiple occurrences of the keyword %s\n" % k)
                    raise RuntimeError
                elif sline.count(k) == 1:
                    marks[k] = (np.array(sline) == k).argmax()
            marks['END'] = len(sline)

            pmark = marks.get('PRM',None)
            if pmark is None: pmark = marks.get('PARM',None)
            rmark = marks.get('RPT',None)
            emark = marks.get('EVAL',None)

            if pmark is not None:
                pstop = min([i for i in marks.values() if i > pmark])
                pflds = [int(i) for i in sline[pmark+1:pstop]] # The integers that specify the parameter word positions
                for pfld in pflds:
                    # For each of the fields that are to be parameterized (indicated by PRM #),
                    # assign a parameter type to it according to the Interaction Type -> Parameter Dictionary.
                    pid = self.Readers[ffname].build_pid(pfld)
                    if xmlScript:
                        pid = 'Script/'+sline[pfld-2]+'/'
                    pid = self.check_dupes(pid, ffname, ln, pfld)
                    self.map[pid] = self.np
                    # This parameter ID has these atoms involved.
                    self.patoms.append([self.Readers[ffname].molatom])
                    # Also append pid to the parameter list
                    self.assign_p0(self.np,float(sline[pfld]))
                    self.assign_field(self.np,pid,ffname,ln,pfld,1)
                    self.np += 1
            if rmark is not None:
                parse = rmark + 1
                stopparse = min([i for i in marks.values() if i > rmark])
                while parse < stopparse:
                    # Between RPT and /RPT, the words occur in pairs.
                    # First is a number corresponding to the field that contains the dependent parameter.
                    # Second is a string corresponding to the 'pid' that this parameter depends on.
                    pfld = int(sline[parse])
                    try:
                        prep = self.map[sline[parse+1].replace('MINUS_','')]
                    except:
                        sys.stderr.write("Valid parameter IDs:\n")
                        count = 0
                        for i in self.map:
                            sys.stderr.write("%25s" % i)
                            if count%3 == 2:
                                sys.stderr.write("\n")
                            count += 1
                        sys.stderr.write("\nOffending ID: %s\n" % sline[parse+1])

                        logger.error('Parameter repetition entry in force field file is incorrect (see above)\n')
                        raise RuntimeError
                    pid = self.Readers[ffname].build_pid(pfld)
                    pid = self.check_dupes(pid, ffname, ln, pfld)
                    self.map[pid] = prep
                    # This repeated parameter ID also has these atoms involved.
                    self.patoms[prep].append(self.Readers[ffname].molatom)
                    self.assign_field(prep,pid,ffname,ln,pfld,"MINUS_" in sline[parse+1] and -1 or 1)
                    parse += 2
            if emark is not None:
                parse = emark + 1
                stopparse = min([i for i in marks.values() if i > emark])
                while parse < stopparse:
                    # Between EVAL and /EVAL, the words occur in pairs.
                    # First is a number corresponding to the field that contains the dependent parameter.
                    # Second is a Python command that determines how to calculate the parameter.
                    pfld = int(sline[parse])
                    evalcmd = sline[parse+1] # This string is actually Python code!!
                    pid = self.Readers[ffname].build_pid(pfld)
                    pid = self.check_dupes(pid, ffname, ln, pfld)
                    # EVAL parameters have no corresponding parameter index
                    #self.map[pid] = None
                    #self.map[pid] = prep
                    # This repeated parameter ID also has these atoms involved.
                    #self.patoms[prep].append(self.Readers[ffname].molatom)
                    self.assign_field(None,pid,ffname,ln,pfld,None,evalcmd)
                    parse += 2

    def addff_xml(self, ffname):
        """ Parse an XML force field file and create important instance variables.

        This was modeled after addff_txt, but XML and text files are
        fundamentally different, necessitating two different methods.

        We begin with an _ElementTree object.  We search through the tree
        for the 'parameterize' and 'parameter_repeat' keywords.  Each time
        the keyword is encountered, we do the same four actions that
        I describe in addff_txt.

        It's hard to specify precisely the location in an XML file to
        change a force field parameter.  I can create a list of tree
        elements (essentially pointers to elements within a tree), but
        this method breaks down when I copy the tree because I have no
        way to refer to the copied tree elements.  Fortunately, lxml
        gives me a way to represent a tree using a flat list, and my
        XML file 'locations' are represented using the positions in
        the list.

        @warning The sign-flip hasn't been implemented yet.  This
        shouldn't matter unless your calculation contains repeated
        parameters with opposite sign.

        """

        #check if xml file contains a script
        #throw error if more than one script
        #write script into .txt file and parse as text
        fflist = list(self.ffdata[ffname].iter())
        scriptElements = [elem for elem in fflist if elem.tag=='Script']
        if len(scriptElements) > 1:
            logger.error('XML file'+ffname+'contains more than one script! Consolidate your scripts into one script!\n')
            raise RuntimeError
        elif len(scriptElements)==1:
            Script = scriptElements[0].text
            ffnameList = ffname.split('.')
            ffnameScript = ffnameList[0]+'Script.txt'
            absScript = os.path.join(self.root, self.ffdir, ffnameScript)
            if os.path.exists(absScript):
                logger.error('XML file '+absScript+' already exists on disk! Please delete it\n')
                raise RuntimeError
            wfile = forcebalance.nifty.wopen(absScript)
            wfile.write(Script)
            wfile.close()
            self.addff(ffnameScript, xmlScript=True)
            os.unlink(absScript)

        for e in self.ffdata[ffname].getroot().xpath('//@parameterize/..'):
            parameters_to_optimize = sorted([i.strip() for i in e.get('parameterize').split(',')])
            for p in parameters_to_optimize:
                if p not in e.attrib:
                    logger.error("Parameter \'%s\' is not found for \'%s\', please check %s" % (p, e.get('type'), ffname) )
                    raise RuntimeError
                pid = self.Readers[ffname].build_pid(e, p)
                self.map[pid] = self.np
                self.assign_p0(self.np,float(e.get(p)))
                self.assign_field(self.np,pid,ffname,fflist.index(e),p,1)
                self.np += 1

        for e in self.ffdata[ffname].getroot().xpath('//@parameter_repeat/..'):
            for field in e.get('parameter_repeat').split(','):
                parameter_name = field.strip().split('=')[0]
                if parameter_name not in e.attrib:
                    logger.error("Parameter \'%s\' is not found for \'%s\', please check %s" % (parameter_name, e.get('type'), ffname) )
                    raise RuntimeError
                dest = self.Readers[ffname].build_pid(e, parameter_name)
                src  = field.strip().split('=')[1]
                if src in self.map:
                    self.map[dest] = self.map[src]
                else:
                    warn_press_key("Warning: You wanted to copy parameter from %s to %s, but the source parameter does not seem to exist!" % (src, dest))
                self.assign_field(self.map[dest],dest,ffname,fflist.index(e),dest.split('/')[1],1)

        for e in self.ffdata[ffname].getroot().xpath('//@parameter_eval/..'):
            for field in e.get('parameter_eval').split(','):
                parameter_name = field.strip().split('=')[0]
                if parameter_name not in e.attrib:
                    logger.error("Parameter \'%s\' is not found for \'%s\', please check %s" % (parameter_name, e.get('type'), ffname) )
                    raise RuntimeError
                dest = self.Readers[ffname].build_pid(e, parameter_name)
                evalcmd  = field.strip().split('=')[1]
                self.assign_field(None,dest,ffname,fflist.index(e),dest.split('/')[1],None,evalcmd)

    def make(self,vals=None,use_pvals=False,printdir=None,precision=12):
        """ Create a new force field using provided parameter values.

        This big kahuna does a number of things:
        1) Creates the physical parameters from the mathematical parameters
        2) Creates force fields with physical parameters substituted in
        3) Prints the force fields to the specified file.

        It does NOT store the mathematical parameters in the class state
        (since we can only hold one set of parameters).

        @param[in] printdir The directory that the force fields are printed to; as usual
        this is relative to the project root directory.
        @param[in] vals Input parameters.  I previously had an option where it uses
        stored values in the class state, but I don't think that's a good idea anymore.
        @param[in] use_pvals Switch for whether to bypass the coordinate transformation
        and use physical parameters directly.

        """
        if type(vals)==np.ndarray and vals.ndim != 1:
            logger.error('Please only pass 1-D arrays\n')
            raise RuntimeError
        if len(vals) != self.np:
            logger.error('Input parameter np.array (%i) not the required size (%i)\n' % (len(vals), self.np))
            raise RuntimeError
        if use_pvals or self.use_pvals:
            logger.info("Using physical parameters directly!\r")
            pvals = vals.copy().flatten()
        else:
            pvals = self.create_pvals(vals)

        OMMFormat = "%%.%ie" % precision
        def TXTFormat(number, precision):
            SciNot = "%% .%ie" % precision
            if abs(number) < 1000 and abs(number) > 0.001:
                Decimal = "%% .%if" % precision
                Num = Decimal % number
                Mum = Decimal % (-1 * number)
                if (float(Num) == float(Mum)):
                    return Decimal % abs(number)
                else:
                    return Decimal % number
            else:
                Num = SciNot % number
                Mum = SciNot % (-1 * number)
                if (float(Num) == float(Mum)):
                    return SciNot % abs(number)
                else:
                    return SciNot % number

        pvals = list(pvals)
        # pvec1d(vals, precision=4)
        newffdata = deepcopy(self.ffdata)

        # The dictionary that takes parameter names to physical values.
        PRM = {i:pvals[self.map[i]] for i in self.map}

        #======================================#
        #     Print the new force field.       #
        #======================================#

        xml_lines = OrderedDict([(fnm, list(newffdata[fnm].iter())) for fnm in self.fnms if self.ffdata_isxml[fnm]])

        for i in range(len(self.pfields)):
            pfield = self.pfields[i]
            pid,fnm,ln,fld,mult,cmd = pfield
            # XML force fields are easy to print.
            # Our 'pointer' to where to replace the value
            # is given by the position of this line in the
            # iterable representation of the tree and the
            # field number.
            # if type(newffdata[fnm]) is etree._ElementTree:
            if cmd is not None:
                try:
                    # Bobby Tables, anyone?
                    if any([x in cmd for x in "system", "subprocess", "import"]):
                        warn_press_key("The command %s (written in the force field file) appears to be unsafe!" % cmd)
                    wval = eval(cmd.replace("PARM","PRM"))
                    # Attempt to allow evaluated parameters to be functions of each other.
                    PRM[pid] = wval
                except:
                    logger.error(traceback.format_exc() + '\n')
                    logger.error("The command %s (written in the force field file) cannot be evaluated!\n" % cmd)
                    raise RuntimeError
            else:
                wval = mult*pvals[self.map[pid]]
            if self.ffdata_isxml[fnm]:
                xml_lines[fnm][ln].attrib[fld] = OMMFormat % (wval)
                # list(newffdata[fnm].iter())[ln].attrib[fld] = OMMFormat % (wval)
            # Text force fields are a bit harder.
            # Our pointer is given by the line and field number.
            # We take care to preserve whitespace in the printout
            # so that the new force field still has nicely formated
            # columns.
            else:
                # Split the string into whitespace and data fields.
                sline       = self.Readers[fnm].Split(newffdata[fnm][ln])
                whites      = self.Readers[fnm].Whites(newffdata[fnm][ln])
                # Align whitespaces and fields (it should go white, field, white, field)
                if newffdata[fnm][ln][0] != ' ':
                    whites = [''] + whites
                # Subtract one whitespace, unless the line begins with a minus sign.
                if not match('^-',sline[fld]) and len(whites[fld]) > 1:
                    whites[fld] = whites[fld][:-1]
                # Actually replace the field with the physical parameter value.
                if precision == 12:
                    newrd  = "% 17.12e" % (wval)
                else:
                    newrd  = TXTFormat(wval, precision)
                # The new word might be longer than the old word.
                # If this is the case, we can try to shave off some whitespace.
                Lold = len(sline[fld])
                if not match('^-',sline[fld]):
                    Lold += 1
                Lnew = len(newrd)
                if Lnew > Lold:
                    Shave = Lnew - Lold
                    if Shave < (len(whites[fld+1])+2):
                        whites[fld+1] = whites[fld+1][:-Shave]
                sline[fld] = newrd
                # Replace the line in the new force field.
                newffdata[fnm][ln] = ''.join([(whites[j] if (len(whites[j]) > 0 or j == 0) else ' ')+sline[j] for j in range(len(sline))])+'\n'

        if printdir is not None:
            absprintdir = os.path.join(self.root,printdir)
        else:
            absprintdir = os.getcwd()

        if not os.path.exists(absprintdir):
            logger.info('Creating the directory %s to print the force field\n' % absprintdir)
            os.makedirs(absprintdir)

        for fnm in newffdata:
            if self.ffdata_isxml[fnm]:
                with wopen(os.path.join(absprintdir,fnm)) as f: newffdata[fnm].write(f)
            elif 'Script.txt' in fnm:
                # if the xml file contains a script, ForceBalance will generate
                # a temporary .txt file containing the script and any updates.
                # We copy the updates made in the .txt file into the xml file by:
                #   First, find xml file corresponding to this .txt file
                #   Second, copy context of the .txt file into the text attribute
                #           of the script element (assumed to be the last element)
                #   Third. open the updated xml file as in the if statement above
                tempText = "".join(newffdata[fnm])
                fnmXml = fnm.split('Script')[0]+'.xml'
                Ntemp = len(list(newffdata[fnmXml].iter()))
                list(newffdata[fnmXml].iter())[Ntemp-1].text = tempText
                '''
                scriptElements = [elem for elem in fflist if elem.tag=='Script']
                if len(scriptElements) > 1:
                logger.error('XML file'+ffname+'contains more than one script! Consolidate your scripts into one script!\n')
                raise RuntimeError
                else:
                '''
                with wopen(os.path.join(absprintdir,fnmXml)) as f: newffdata[fnmXml].write(f)
            else:
                with wopen(os.path.join(absprintdir,fnm)) as f: f.writelines(newffdata[fnm])

        return pvals

    def make_redirect(self,mvals):
        Groups = defaultdict(list)
        for p, pid in enumerate(self.plist):
            if 'Exponent' not in pid or len(pid.split()) != 1:
                warn_press_key("Fusion penalty currently implemented only for basis set optimizations, where parameters are like this: Exponent:Elem=H,AMom=D,Bas=0,Con=0")
            Data = dict([(i.split('=')[0],i.split('=')[1]) for i in pid.split(':')[1].split(',')])
            if 'Con' not in Data or Data['Con'] != '0':
                warn_press_key("More than one contraction coefficient found!  You should expect the unexpected")
            key = Data['Elem']+'_'+Data['AMom']
            Groups[key].append(p)
        #pvals = self.FF.create_pvals(mvals)
        #print "pvals: ", pvals

        pvals = self.create_pvals(mvals)
        logger.info("pvals:\n")
        logger.info(str(pvals) + '\n')

        Thresh = 1e-4

        for gnm, pidx in Groups.items():
            # The group of parameters for a particular element / angular momentum.
            pvals_grp = pvals[pidx]
            # The order that the parameters come in.
            Order = np.argsort(pvals_grp)
            for p in range(len(Order) - 1):
                # The pointers to the parameter indices.
                pi = pidx[Order[p]]
                pj = pidx[Order[p+1]]
                # pvals[pi] is the SMALLER parameter.
                # pvals[pj] is the LARGER parameter.
                dp = np.log(pvals[pj]) - np.log(pvals[pi])
                if dp < Thresh:
                    if pi in self.redirect:
                        pk = self.redirect[pi]
                    else:
                        pk = pi
                    #if pj not in self.redirect:
                        #print "Redirecting parameter %i to %i" % (pj, pk)
                        #self.redirect[pj] = pk
        #print self.redirect

    def find_spacings(self):
        Groups = defaultdict(list)
        for p, pid in enumerate(self.plist):
            if 'Exponent' not in pid or len(pid.split()) != 1:
                warn_press_key("Fusion penalty currently implemented only for basis set optimizations, where parameters are like this: Exponent:Elem=H,AMom=D,Bas=0,Con=0")
            Data = dict([(i.split('=')[0],i.split('=')[1]) for i in pid.split(':')[1].split(',')])
            if 'Con' not in Data or Data['Con'] != '0':
                warn_press_key("More than one contraction coefficient found!  You should expect the unexpected")
            key = Data['Elem']+'_'+Data['AMom']
            Groups[key].append(p)

        pvals = self.create_pvals(np.zeros(self.np))
        logger.info("pvals:\n")
        logger.info(str(pvals) + '\n')

        spacdict = {}
        for gnm, pidx in Groups.items():
            # The group of parameters for a particular element / angular momentum.
            pvals_grp = pvals[pidx]
            # The order that the parameters come in.
            Order = np.argsort(pvals_grp)
            spacs = []
            for p in range(len(Order) - 1):
                # The pointers to the parameter indices.
                pi = pidx[Order[p]]
                pj = pidx[Order[p+1]]
                # pvals[pi] is the SMALLER parameter.
                # pvals[pj] is the LARGER parameter.
                dp = np.log(pvals[pj]) - np.log(pvals[pi])
                spacs.append(dp)
            if len(spacs) > 0:
                spacdict[gnm] = np.mean(np.array(spacs))
        return spacdict

    def create_pvals(self,mvals):
        """Converts mathematical to physical parameters.

        First, mathematical parameters are rescaled and rotated by
        multiplying by the transformation matrix, followed by adding
        the original physical parameters.

        @param[in] mvals The mathematical parameters
        @return pvals The physical parameters

        """
        if isinstance(mvals, list):
            mvals = np.array(mvals)
        for p in self.redirect:
            mvals[p] = 0.0
        if self.logarithmic_map:
            try:
                pvals = np.exp(mvals.flatten()) * self.pvals0
            except:
                logger.exception(str(mvals) + '\n')
                logger.error('What the hell did you do?\n')
                raise RuntimeError
        else:
            pvals = flat(np.matrix(self.tmI)*col(mvals)) + self.pvals0
        concern= ['polarizability','epsilon','VDWT']
        # Guard against certain types of parameters changing sign.

        for i in range(self.np):
            if any([j in self.plist[i] for j in concern]) and pvals[i] * self.pvals0[i] < 0:
                #print "Parameter %s has changed sign but it's not allowed to! Setting to zero." % self.plist[i]
                pvals[i] = 0.0
        # Redirect parameters (for the fusion penalty function.)
        for p in self.redirect:
            pvals[p] = pvals[self.redirect[p]]
        # if not in_fd():
        #     print pvals
        #print "pvals = ", pvals

        return pvals


    def create_mvals(self,pvals):
        """Converts physical to mathematical parameters.

        We create the inverse transformation matrix using SVD.

        @param[in] pvals The physical parameters
        @return mvals The mathematical parameters
        """

        if self.logarithmic_map:
            logger.error('create_mvals has not been implemented for logarithmic_map\n')
            raise RuntimeError
        mvals = flat(invert_svd(self.tmI) * col(pvals - self.pvals0))

        return mvals

    def rsmake(self,printfacs=True):
        """Create the rescaling factors for the coordinate transformation in parameter space.

        The proper choice of rescaling factors (read: prior widths in maximum likelihood analysis)
        is still a black art.  This is a topic of current research.

        @todo Pass in rsfactors through the input file

        @param[in] printfacs List for printing out the resecaling factors

        """
        typevals = OrderedDict()
        rsfactors = OrderedDict()
        rsfac_list = []
        ## Takes the dictionary 'BONDS':{3:'B', 4:'K'}, 'VDW':{4:'S', 5:'T'},
        ## and turns it into a list of term types ['BONDSB','BONDSK','VDWS','VDWT']

        if any([self.Readers[i].pdict == "XML_Override" for i in self.fnms]):
            termtypelist = ['/'.join([i.split('/')[0],i.split('/')[1]]) for i in self.map]
        else:
            termtypelist = itertools.chain(*sum([[[i+self.Readers[f].pdict[i][j] for j in self.Readers[f].pdict[i] if isint(str(j))] for i in self.Readers[f].pdict] for f in self.fnms],[]))
            #termtypelist = sum([[i+self.Readers.pdict[i][j] for j in self.Readers.pdict[i] if isint(str(j))] for i in self.Readers.pdict],[])
        for termtype in termtypelist:
            for pid in self.map:
                if termtype in pid:
                    typevals.setdefault(termtype, []).append(self.pvals0[self.map[pid]])
        for termtype in typevals:
            # The old, horrendously complicated rule
            # rsfactors[termtype] = exp(mean(log(abs(array(typevals[termtype]))+(abs(array(typevals[termtype]))==0))))
            # The newer, maximum rule (thanks Baba)
            maxval = max(abs(np.array(typevals[termtype])))
            # When all initial parameter values are zero, it could be a problem...
            if maxval == 0:
                maxval += 1
            rsfactors[termtype] = maxval
            rsfac_list.append(termtype)
            # Physically motivated overrides
            rs_override(rsfactors,termtype)
        # Overrides from input file
        for termtype in self.priors:
            while termtype in rsfac_list:
                rsfac_list.remove(termtype)
            rsfac_list.append(termtype)
            rsfactors[termtype] = self.priors[termtype]

        # for line in os.popen("awk '/rsfactor/ {print $2,$3}' %s" % pkg.options).readlines():
        #     rsfactors[line.split()[0]] = float(line.split()[1])
        if printfacs:
            bar = printcool("Rescaling Factors by Type (Lower Takes Precedence):",color=1)
            logger.info(''.join(["   %-35s  : %.5e\n" % (i, rsfactors[i]) for i in rsfac_list]))
            logger.info(bar)
        self.rs_ord = OrderedDict([(i, rsfactors[i]) for i in rsfac_list])
        ## The array of rescaling factors
        self.rs = np.ones(len(self.pvals0))
        self.rs_type = OrderedDict()
        have_rs = []
        for pnum in range(len(self.pvals0)):
            for termtype in rsfac_list:
                if termtype in self.plist[pnum]:
                    if pnum not in have_rs:
                        self.rs[pnum] = rsfactors[termtype]
                        self.rs_type[pnum] = termtype
                    elif self.rs[pnum] != rsfactors[termtype]:
                        self.rs[pnum] = rsfactors[termtype]
                        self.rs_type[pnum] = termtype
                    have_rs.append(pnum)
        ## These parameter types have no rescaling factors defined, so they are just set to unity
        for pnum in range(len(self.pvals0)):
            if pnum not in have_rs:
                self.rs_type[pnum] = self.plist[pnum][0]
        if printfacs:
            bar = printcool("Rescaling Types / Factors by Parameter Number:",color=1)
            self.print_map(vals=["   %-28s  : %.5e" % (self.rs_type[pnum], self.rs[pnum]) for pnum in range(len(self.pvals0))])
            logger.info(bar)

    def make_rescale(self, scales, mvals=None, G=None, H=None, multiply=True, verbose=False):
        """ Obtain rescaled versions of the inputs according to dictionary values
        in "scales" (i.e. a replacement or multiplicative factor on self.rs_ord).
        Note that self.rs and self.rs_ord are not updated in this function. You
        need to do that outside.

        The purpose of this function is to simulate the effect of changing the
        parameter scale factors in the force field. If the scale factor is N,
        then a unit change in the mathematical parameters produces a change of
        N in the physical parameters. Thus, for a given point in the physical
        parameter space, the mathematical parameters are proportional to 1/N,
        the gradient is proportional to N, and the Hessian is proportional to N^2.

        Parameters
        ----------
        mvals : numpy.ndarray
            Parameters to be transformed, if desired.  Must be same length as number of parameters.
        G : numpy.ndarray
            Gradient to be transformed, if desired.  Must be same length as number of parameters.
        H : numpy.ndarray
            Hessian to be transformed, if desired.  Must be square matrix with side-length = number of parameters.
        scales : OrderedDict
            A dictionary with the same keys as self.rs_ord and floating point values.
            These represent the values with which to multiply the existing scale factors
            (if multiply == True) or replace them (if multiply == False).
            Pro Tip: Create this variable from a copy of self.rs_ord
        multiply : bool
            When set to True, the new scale factors are the existing scale factors
        verbose : bool
            Loud and noisy

        Returns
        -------
        answer : OrderedDict
            Output dictionary containing :
            'rs' : New parameter scale factors (multiplied by scales if multiply=True, or replaced if multiply=False)
            'rs_ord' : New parameter scale factor dictionary
            'mvals' : New parameter values (if mvals is provided)
            'G' : New gradient (if G is provided)
            'H' : New Hessian (if H is provided)
        """
        if type(scales) != OrderedDict:
            raise RuntimeError('scales must have type OrderedDict')
        if scales.keys() != self.rs_ord.keys():
            raise RuntimeError('scales should have same keys as self.rs_ord')
        # Make the new dictionary of rescaling factors
        if multiply == False:
            rsord_out = deepcopy(scales)
        else:
            rsord_out = OrderedDict([(k, scales[k]*self.rs_ord[k]) for k in scales.keys()])
        answer = OrderedDict()
        answer['rs_ord'] = rsord_out
        # Make the new array of rescaling factors
        rs_out = np.array([rsord_out[self.rs_type[p]] for p in range(self.np)])
        answer['rs'] = rs_out
        if mvals is not None:
            if mvals.shape != (self.np,):
                raise RuntimeError('mvals has the wrong shape')
            mvals_out = deepcopy(mvals)
            for p in range(self.np):
                # Remember that for the same physical parameters, the mathematical
                # parameters are inversely proportional to scale factors
                mvals_out[p] *= (float(self.rs[p])/float(rs_out[p]))
            answer['mvals'] = mvals_out
        if G is not None:
            if G.shape != (self.np,):
                raise RuntimeError('G has the wrong shape')
            G_out = deepcopy(G)
            for p in range(self.np):
                # The gradient should be proportional to the scale factors
                G_out[p] *= (float(rs_out[p])/float(self.rs[p]))
            answer['G'] = G_out
        if H is not None:
            if H.shape != (self.np,self.np):
                raise RuntimeError('H has the wrong shape')
            H_out = deepcopy(H)
            for p in range(self.np):
                # The Hessian should be proportional to the product of two scale factors
                H_out[p, :] *= (float(rs_out[p])/float(self.rs[p]))
            for p in range(self.np):
                H_out[:, p] *= (float(rs_out[p])/float(self.rs[p]))
            answer['H'] = H_out
        # The final parameters, gradient and Hessian should be consistent with the
        # returned scale factors.
        return answer

    def mktransmat(self):
        """ Create the transformation matrix to rescale and rotate the mathematical parameters.

        For point charge parameters, project out perturbations that
        change the total charge.

        First build these:

        'qmap'    : Just a list of parameter indices that point to charges.

        'qid'     : For each parameter in the qmap, a list of the affected atoms :)
                    A potential target for the molecule-specific thang.

        Then make this:

        'qtrans2' : A transformation matrix that rotates the charge parameters.
                    The first row is all zeros (because it corresponds to increasing the charge on all atoms)
                    The other rows correspond to changing one of the parameters and decreasing all of the others
                    equally such that the overall charge is preserved.

        'qmat2'   : An identity matrix with 'qtrans2' pasted into the right place

        'transmat': 'qmat2' with rows and columns scaled using self.rs

        'excision': Parameter indices that need to be 'cut out' because they are irrelevant and
                    mess with the matrix diagonalization

        @todo Only project out changes in total charge of a molecule, and perhaps generalize to
        fragments of molecules or other types of parameters.
        @todo The AMOEBA selection of charge depends not only on the atom type, but what that atom is bonded to.
        """
        self.qmap   = []
        self.qid    = []
        self.qid2   = []
        qnr    = 1
        concern= ['COUL','c0','charge']
        qmat2 = np.eye(self.np)

        def insert_mat(qtrans2, qmap):
            # Write the qtrans2 block into qmat2.
            x = 0
            for i in range(self.np):
                if i in qmap:
                    y = 0
                    for j in qmap:
                        qmat2[i, j] = qtrans2[x, y]
                        y += 1
                    x += 1

        def build_qtrans2(tq, qid, qmap):
            """ Build the matrix that ensures the net charge does not change. """
            nq = len(qmap)
            # tq = Total number of atomic charges that are being optimized on the molecule
            # NOTE: This may be greater than the number of charge parameters (nq)
            # The reason for the "one" here is because LP wanted to have multiple charge constraints
            # at some point in the future
            cons0 = np.ones((1,tq))
            cons = np.zeros((cons0.shape[0], nq))
            # Identity matrix equal to the number of charge parameters
            qtrans2 = np.eye(nq)
            # This is just one
            for i in range(cons.shape[0]):
                # Loop over the number of charge parameters
                for j in range(cons.shape[1]):
                    # Each element of qid is a list that points to atom indices.
                    # LPW: This code is breaking when we're not optimizing ALL the charges
                    # Replace cons0[i][k-1] with all ones
                    # cons[i][j] = sum([cons0[i][k-1] for k in qid[j]])
                    cons[i][j] = float(len(qid[j]))
                cons[i] /= np.linalg.norm(cons[i])
                for j in range(i):
                    cons[i] = orthogonalize(cons[i], cons[j])
                qtrans2[i,:] = 0
                for j in range(nq-i-1):
                    qtrans2[i+j+1, :] = orthogonalize(qtrans2[i+j+1, :], cons[i])
            return qtrans2
        # Here we build a charge constraint for each molecule.
        if any(len(r.adict) > 0 for r in self.Readers.values()):
            logger.info("Building charge constraints...\n")
            # Build a concatenated dictionary
            Adict = OrderedDict()
            # This is a loop over files
            for r in self.Readers.values():
                # This is a loop over molecules
                for k, v in r.adict.items():
                    Adict[k] = v
            nmol = 0
            for molname, molatoms in Adict.items():
                mol_charge_count = np.zeros(self.np)
                tq = 0
                qmap = []
                qid = []
                for i in range(self.np):
                    qct = 0
                    qidx = []
                    for imol, iatoms in self.patoms[i]:
                        for iatom in iatoms:
                            if imol == molname and iatom in molatoms:
                                qct += 1
                                tq += 1
                                qidx.append(molatoms.index(iatom))
                    if any([j in self.plist[i] for j in concern]) and qct > 0:
                        qmap.append(i)
                        qid.append(qidx)
                        logger.info("Parameter %i occurs %i times in molecule %s in locations %s (%s)\n" % (i, qct, molname, str(qidx), self.plist[i]))
                #Here is where we build the qtrans2 matrix.
                if len(qmap) > 0:
                    qtrans2 = build_qtrans2(tq, qid, qmap)
                    if self.constrain_charge:
                        insert_mat(qtrans2, qmap)
                if nmol == 0:
                    self.qid = qid
                    self.qmap = qmap
                # The warning about ESP fitting is not very helpful
                # else:
                #     logger.info("Note: ESP fitting will be performed assuming that molecule id %s is the FIRST molecule and the only one being fitted.\n" % molname)
                nmol += 1
        elif self.constrain_charge:
            warn_press_key("'adict' {molecule:atomnames} was not found.\n This isn't a big deal if we only have one molecule, but might cause problems if we want multiple charge neutrality constraints.")
            qnr = 0
            if any([self.Readers[i].pdict == "XML_Override" for i in self.fnms]):
                # Hack to count the number of atoms for each atomic charge parameter, when the force field is an XML file.
                # This needs to be changed to Chain or Molecule
                logger.info(str([determine_fftype(k) for k in self.ffdata]))
                ListOfAtoms = list(itertools.chain(*[[e.get('type') for e in self.ffdata[k].getroot().xpath('//Residue/Atom')] for k in self.ffdata if determine_fftype(k) == "openmm"]))
            for i in range(self.np):
                if any([j in self.plist[i] for j in concern]):
                    self.qmap.append(i)
                    if 'Multipole/c0' in self.plist[i] or 'Atom/charge' in self.plist[i]:
                        AType = self.plist[i].split('/')[-1].split('.')[0]
                        nq = count(ListOfAtoms,AType)
                    else:
                        thisq = []
                        for k in self.plist[i].split():
                            for j in concern:
                                if j in k:
                                    thisq.append(k.split('-')[-1])
                                    break
                        try:
                            self.qid2.append(np.array([self.atomnames.index(k) for k in thisq]))
                        except: pass
                        nq = sum(np.array([count(self.plist[i], j) for j in concern]))
                    self.qid.append(qnr+np.arange(nq))
                    qnr += nq
            if len(self.qid2) == 0:
                sys.stderr.write('Unable to match atom numbers up with atom names (minor issue, unless doing ESP fitting).  \nAre atom names implemented in the force field parser?\n')
            else:
                self.qid = self.qid2
            tq = qnr - 1
            #Here is where we build the qtrans2 matrix.
            if len(self.qmap) > 0:
                cons0 = np.ones((1,tq))
                qtrans2 = build_qtrans2(tq, self.qid, self.qmap)
                # Insert qtrans2 into qmat2.
                if self.constrain_charge:
                    insert_mat(qtrans2, self.qmap)

        ## Some customized constraints here.
        # Quadrupoles must be traceless
        if self.constrain_charge:
            MultipoleAtoms = set([p.split('/')[-1] for p in self.plist if 'Multipole' in p])
            QuadrupoleGrps = [[i for i, p in enumerate(self.plist) if 'Multipole' in p and p.split('/')[-1] == A and p.split('/')[1] in ['q11','q22','q33']] for A in MultipoleAtoms]
            for Grp in QuadrupoleGrps:
                qid = [np.array([i]) for i in range(3)]
                tq = 3
                qtrans2 = build_qtrans2(tq, qid, Grp)
                logger.info("Making sure that quadrupoles are traceless (for parameter IDs %s)\n" % str(Grp))
                insert_mat(qtrans2, Grp)

        #ListOfAtoms = list(itertools.chain(*[[e.get('type') for e in self.ffdata[k].getroot().xpath('//Multipole')] for k in self.ffdata]))

        # print "Charge parameter constraint matrix - feel free to check it"
        # for i in qmat2:
        #     for j in i:
        #         print "% .3f" % j,
        #     print
        # print

        # There is a bad bug here .. this matrix multiplication operation doesn't work!!
        # I will proceed using loops. This is unsettling.
        # Input matrices are qmat2 and self.rs (diagonal)
        transmat = np.matrix(qmat2) * np.matrix(np.diag(self.rs))
        transmat1 = np.zeros((self.np, self.np))
        for i in range(self.np):
            for k in range(self.np):
                transmat1[i,k] = qmat2[i,k] * self.rs[k]

        # This prints out the difference between the result of matrix multiplication
        # and the manual multiplication.
        #print transmat1
        #print transmat
        if len(transmat) > 0 and np.max(np.abs(transmat1 - transmat)) > 0.0:
            logger.warning('The difference between the numpy multiplication and the manual multiplication is \x1b[1;91m%f\x1b[0m, '
                           'but it should be zero.\n' % np.max(np.abs(transmat1 - transmat)))

            transmat = np.array(transmat1, copy=True)
        transmatNS = np.array(transmat,copy=True)
        self.excision = []
        for i in range(self.np):
            if abs(transmatNS[i, i]) < 1e-8:
                self.excision.append(i)
                transmatNS[i, i] += 1
        self.excision = list(set(self.excision))
        for i in self.excision:
            transmat[i, :] = np.zeros(self.np)
        self.tm = transmat
        self.tmI = transmat.T

    def list_map(self):
        """ Create the plist, which is like a reversed version of the parameter map.  More convenient for printing. """
        if len(self.map) == 0:
            warn_press_key('The parameter map has no elements (Okay if we are not actually tuning any parameters.)')
        else:
            self.plist = [[] for j in range(max([self.map[i] for i in self.map])+1)]
            for i in self.map:
                self.plist[self.map[i]].append(i)
            for i in range(self.np):
                self.plist[i] = ' '.join(natural_sort(self.plist[i]))

    def print_map(self,vals = None,precision=4):
        """Prints out the (physical or mathematical) parameter indices, IDs and values in a visually appealing way."""
        if vals is None:
            vals = self.pvals0
        logger.info('\n'.join(["%4i [ %s ]" % (self.plist.index(i), "%% .%ie" % precision % float(vals[self.plist.index(i)]) if isfloat(str(vals[self.plist.index(i)])) else (str(vals[self.plist.index(i)]))) + " : " + "%s" % i.split()[0] for i in self.plist]))
        logger.info('\n')

    def sprint_map(self,vals = None,precision=4):
        """Prints out the (physical or mathematical) parameter indices, IDs and values to a string."""
        if vals is None:
            vals = self.pvals0
        out = '\n'.join(["%4i [ %s ]" % (self.plist.index(i), "%% .%ie" % precision % float(vals[self.plist.index(i)]) if isfloat(str(vals[self.plist.index(i)])) else (str(vals[self.plist.index(i)]))) + " : " + "%s" % i.split()[0] for i in self.plist])
        return out

    def assign_p0(self,idx,val):
        """ Assign physical parameter values to the 'pvals0' array.

        @param[in] idx The index to which we assign the parameter value.
        @param[in] val The parameter value to be inserted.
        """
        if idx == len(self.pvals0):
            self.pvals0.append(val)
        else:
            self.pvals0[idx] = val

    def assign_field(self,idx,pid,fnm,ln,pfld,mult,cmd=None):
        """ Record the locations of a parameter in a txt file; [[file name, line number, field number, and multiplier]].

        Note that parameters can have multiple locations because of the repetition functionality.

        @param[in] idx  The (not necessarily unique) index of the parameter.
        @param[in] pid  The unique parameter name.
        @param[in] fnm  The file name of the parameter field.
        @param[in] ln   The line number within the file (or the node index in the flattened xml)
        @param[in] pfld The field within the line (or the name of the attribute in the xml)
        @param[in] mult The multiplier (this is usually 1.0)

        """
        self.pfields.append([pid,fnm,ln,pfld,mult,cmd])

    def __eq__(self, other):
        # check equality of forcefields using comparison of pfields and map
        if isinstance(other, FF):
            # list comprehension removes pid/filename element of pfields since we don't care about filename uniqueness
            self_pfields = [p[2:] for p in self.pfields]
            other_pfields= [p[2:] for p in other.pfields]

            return  self_pfields == other_pfields and\
                        self.map == other.map and\
                        (self.pvals0 == other.pvals0).all()

        # we only compare two forcefield objects
        else: return NotImplemented

def rs_override(rsfactors,termtype,Temperature=298.15):
    """ This function takes in a dictionary (rsfactors) and a string (termtype).

    If termtype matches any of the strings below, rsfactors[termtype] is assigned
    to one of the numbers below.

    This is LPW's attempt to simplify the rescaling factors.

    @param[out] rsfactors The computed rescaling factor.
    @param[in] termtype The interaction type (corresponding to a physical unit)
    @param[in] Temperature The temperature for computing the kT energy scale

    """
    if match('PIMPDIHS[1-6]K|PDIHMULS[1-6]K|PDIHS[1-6]K|RBDIHSK[1-5]|MORSEC',termtype):
        # eV or eV rad^-2
        rsfactors[termtype] = 96.4853
    elif match('UREY_BRADLEYK1|ANGLESK',termtype):
        rsfactors[termtype] = 96.4853 * 6.28
    elif match('COUL|VPAIR_BHAMC|QTPIEA',termtype):
        # elementary charge, or unitless, or already in atomic unit
        rsfactors[termtype] = 1.0
    elif match('QTPIEC|QTPIEH',termtype):
        # eV to atomic unit
        rsfactors[termtype] = 27.2114
    elif match('BONDSB|UREY_BRADLEYB|MORSEB|VDWS|VPAIRS|VSITE|VDW_BHAMA|VPAIR_BHAMA',termtype):
        # nm to atomic unit
        rsfactors[termtype] = 0.05291772
    elif match('BONDSK|UREY_BRADLEYK2',termtype):
        # au bohr^-2
        rsfactors[termtype] = 34455.5275 * 27.2114
    elif match('PDIHS[1-6]B|ANGLESB|UREY_BRADLEYB',termtype):
        # radian
        rsfactors[termtype] = 57.295779513
    elif match('VDWT|VDW_BHAMB|VPAIR_BHAMB',termtype):
        # VdW well depth; using kT.  This was a tough one because the energy scale is so darn small.
        rsfactors[termtype] = kb*Temperature
    elif match('MORSEE',termtype):
        rsfactors[termtype] = 18.897261
