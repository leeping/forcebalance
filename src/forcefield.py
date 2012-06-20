""" @package forcefield

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
from re import match, sub, split
import gmxio, qchemio, tinkerio, custom_io, openmmio, amberio
import basereader
from numpy import arange, array, diag, exp, eye, log, mat, mean, ones, vstack, zeros
from numpy.linalg import norm
from nifty import col, flat, invert_svd, isint, isfloat, kb, orthogonalize, pmat2d, printcool, row, warn_press_key
from string import count
from copy import deepcopy
try:
    from lxml import etree
except: pass
import itertools
from collections import OrderedDict


FF_Extensions = {"itp" : "gmx",
                 "in"  : "qchem",
                 "prm" : "tinker",
                 "gen" : "custom",
                 "xml" : "openmm",
                 "frcmod" : "frcmod",
                 "mol2" : "mol2"
                 }

""" Recognized force field formats. """
FF_IOModules = {"gmx": gmxio.ITP_Reader ,
                "qchem": qchemio.QCIn_Reader ,
                "tinker": tinkerio.Tinker_Reader ,
                "custom": custom_io.Gen_Reader , 
                "openmm" : openmmio.OpenMM_Reader,
                "frcmod" : amberio.FrcMod_Reader,
                "mol2" : amberio.Mol2_Reader
                }

def determine_fftype(ffname,verbose=False):
    """ Determine the type of a force field file.  It is possible to
    specify the file type explicitly in the input file using the
    syntax 'force_field.ext:type'.  Otherwise this function will try
    to determine the force field type by extension. """

    fsplit = ffname.split('/')[-1].split(':')
    fftype = None
    if verbose: print "Determining file type of %s ..." % fsplit[0],
    if len(fsplit) == 2:
        if fsplit[1] in FF_IOModules:
            if verbose: print "We're golden! (%s)" % fsplit[1]
            fftype = fsplit[1]
        else:
            if verbose: print "\x1b[91m Warning: \x1b[0m %s not in supported types (%s)!" % (fsplit[1],', '.join(FF_IOModules.keys()))
    elif len(fsplit) == 1:
        if verbose: print "Guessing from extension (you may specify type with filename:type) ...", 
        ffname = fsplit[0]
        ffext = ffname.split('.')[-1]
        if ffext in FF_Extensions:
            guesstype = FF_Extensions[ffext]
            if guesstype in FF_IOModules:
                if verbose: print "guessing %s -> %s!" % (ffext, guesstype)
                fftype = guesstype
            else:
                if verbose: print "\x1b[91m Warning: \x1b[0m %s not in supported types (%s)!" % (fsplit[0],', '.join(FF_IOModules.keys()))
        else:
            if verbose: print "\x1b[91m Warning: \x1b[0m %s not in supported extensions (%s)!" % (ffext,', '.join(FF_Extensions.keys()))
    if fftype == None:
        if verbose: print "Force field type not determined!"
        #sys.exit(1)
    return fftype

class FF(object):
    """ Force field class.

    This class contains all methods for force field manipulation.
    To create an instance of this class, an input file is required
    containing the list of force field file names.  Everything else
    inside this class pertaining to force field generation is self-contained.

    For details on force field parsing, see the detailed documentation for addff.
    
    """
    def __init__(self, options, verbose=True):

        """Instantiation of force field class.

        Many variables here are initialized to zero, but they are filled out by
        methods like addff, rsmake, and mktransmat.
        
        """
        #======================================#
        # Options that are given by the parser #
        #======================================#
        ## As these options proliferate, the force field class becomes less standalone.
        ## I need to think of a good solution here...
        ## The root directory of the project
        self.root        = os.getcwd()
        ## File names of force fields
        self.fnms        = options['forcefield']
        ## Directory containing force fields, relative to project directory
        self.ffdir       = options['ffdir']
        ## Priors given by the user :)
        self.priors      = options['priors']
        ## Whether to constrain the charges.
        self.constrain_charge  = options['constrain_charge']
        ## Whether to constrain the charges.
        self.logarithmic_map  = options['logarithmic_map']
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        # A lot of these variables are filled out by calling the methods.
        
        ## The content of all force field files are stored in memory
        self.ffdata       = {}        
        ## The mapping of interaction type -> parameter number
        self.map         = {}
        ## The listing of parameter number -> interaction types
        self.plist       = []
        ## A listing of parameter number -> atoms involved
        self.patoms      = []
        ## A list where pfields[pnum] = ['file',line,field,mult],
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
        self.R           = OrderedDict()
        ## A list of atom names (this is new, for ESP fitting)
        self.atomnames   = []
        
        # Read the force fields into memory.
        for fnm in self.fnms:
            if verbose:
                print "Reading force field from file: %s" % fnm
            self.addff(fnm)

        # Set the initial values of parameter arrays.
        ## Initial value of physical parameters
        self.pvals0 = array(self.pvals0)

        # Prepare various components of the class for first use.
        ## Creates plist from map.
        self.list_map()
        if verbose:
            ## Prints the plist to screen.
            bar = printcool("Starting parameter indices, physical values and IDs")
            self.print_map()                       
            print bar
        ## Make the rescaling factors.
        self.rsmake(printfacs=verbose)            
        ## Make the transformation matrix.
        self.mktransmat()                      
        
    def addff(self,ffname):
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

        In this example, the parameter identifier would look like <tt> AmoebaVdwForce.Vdw_74_epsilon </tt>.

        --- If GROMACS (.itp) or TINKER (.prm) : ---

        Follow the rules in the ITP_Reader or Tinker_Reader derived
        class.  Read the documentation in the class documentation or
        the 'feed' method to learn more.  In all cases the parameter
        is tagged using <tt> # PARM 3 </tt> (where # denotes a comment,
        the word PARM stays the same, and 3 is the field number starting
        from zero.)

        --- If normal text : ---
        
        The parameter identifier is simply built using the file name,
        line number, and field.  Thus, the identifier is unique but
        completely noninformative (which is not ideal for our
        purposes, but it works.)

        --- Endif ---

        @warning My program currently assumes that we are only using one MM program per job.
        If we use CHARMM and GROMACS to perform fitting simulations in the same job, we will
        get f-ed up.  Maybe this needs to be fixed in the future, with program prefixes to
        parameters like C_ , G_ .. or simply unit conversions, you get the idea.

        @warning I don't think the multiplier actually works for analytic derivatives unless
        the interaction calculator knows the multiplier as well.  I'm sure I can make this
        work in the future if necessary.

        @param[in] ffname Name of the force field file

        """
        fftype = determine_fftype(ffname)
        ffname = ffname.split(':')[0]

        # Determine the appropriate parser from the FF_IOModules dictionary.
        # If we can't figure it out, then use the base reader, it ain't so bad. :)
        Reader = FF_IOModules.get(fftype,basereader.BaseReader)

        # Open the force field using an absolute path and read its contents into memory.
        absff = os.path.join(self.root,self.ffdir,ffname)
        
        # Create an instance of the Reader.
        # The reader is essentially a finite state machine that allows us to 
        # build the pid.
        self.R[ffname] = Reader(ffname)
        if fftype == "openmm":
            ## Read in an XML force field file as an _ElementTree object
            self.ffdata[ffname] = etree.parse(absff)
            # Process the file
            self.addff_xml(ffname)
        else:
            ## Read in a text force field file as a list of lines
            self.ffdata[ffname] = [line.expandtabs() for line in open(absff).readlines()]
            # Process the file
            self.addff_txt(ffname, fftype)
        if hasattr(self.R[ffname], 'atomnames'):
            if len(self.atomnames) > 0:
                sys.stderr.write('Found more than one force field containing atom names; skipping the second one (%s)\n' % ffname)
            else:
                self.atomnames += self.R[ffname].atomnames

    def addff_txt(self, ffname, fftype):
        """ Parse a text force field and create several important instance variables.

        Each line is processed using the 'feed' method as implemented
        in the reader class.  This essentially allows us to create the
        correct parameter identifier (pid), because the pid comes from
        more than the current line, it also depends on the section that
        we're in.

        When 'PARM' or 'RPT' is encountered, we do several things:
        - Build the parameter identifier and insert it into the map
        - Point to the file name, line number, and field where the parameter may be modified
        
        Additionally, when 'PARM' is encountered:
        - Store the physical parameter value (this is permanent; it's the original value)
        - Increment the total number of parameters

        When 'RPT' is encountered we don't expand the parameter vector
        because this parameter is a copy of an existing one.  If the
        parameter identifier is preceded by MINUS_, we chop off the
        prefix but remember that the sign needs to be flipped.

        """
        
        for ln, line in enumerate(self.ffdata[ffname]):
            self.R[ffname].feed(line)
            sline = self.R[ffname].Split(line)
            if 'PARM' in sline:
                pmark = (array(sline) == 'PARM').argmax() # The position of the 'PARM' word
                pflds = [int(i) for i in sline[pmark+1:]] # The integers that specify the parameter word positions
                for pfld in pflds:
                    # For each of the fields that are to be parameterized (indicated by PARM #),
                    # assign a parameter type to it according to the Interaction Type -> Parameter Dictionary.
                    pid = self.R[ffname].build_pid(pfld)
                    # Add pid into the dictionary.
                    self.map[pid] = self.np
                    # This parameter ID has these atoms involved.
                    self.patoms.append([self.R[ffname].molatom])
                    # Also append pid to the parameter list
                    self.assign_p0(self.np,float(sline[pfld]))
                    self.assign_field(self.np,ffname,ln,pfld,1)
                    self.np += 1
            if "RPT" in sline:
                parse = (array(sline)=='RPT').argmax()+1 # The position of the 'RPT' word
                while parse < (array(sline)=='/RPT').argmax():
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
                        
                        raise Exception('Parameter repetition entry in force field file is incorrect (see above)')
                    pid = self.R[ffname].build_pid(pfld)
                    self.map[pid] = prep
                    # This repeated parameter ID also has these atoms involved.
                    self.patoms[prep].append(self.R[ffname].molatom)
                    self.assign_field(prep,ffname,ln,pfld,"MINUS_" in sline[parse+1] and -1 or 1)
                    parse += 2
    
    def addff_xml(self, ffname):
        """ Parse an XML force field file and create important instance variables.

        This was modeled after addff_txt, but XML and text files are
        fundamentally different, necessitating two different methods.

        We begin with an _ElementTree object.  We search through the tree
        for the 'parameterize' and 'param_repeat' keywords.  Each time
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
        
        fflist = list(self.ffdata[ffname].iter())
        for e in self.ffdata[ffname].getroot().xpath('//@parameterize/..'):
            parameters_to_optimize = sorted([i.strip() for i in e.get('parameterize').split(',')])
            for p in parameters_to_optimize:
                pid = self.R[ffname].build_pid(e, p)
                self.map[pid] = self.np
                self.assign_p0(self.np,float(e.get(p)))
                self.assign_field(self.np,ffname,fflist.index(e),p,1)
                self.np += 1

        for e in self.ffdata[ffname].getroot().xpath('//@param_repeat/..'):
            for field in e.get('param_repeat').split(','):
                dest = self.R[ffname].build_pid(e, field.strip().split('=')[0])
                src  = field.strip().split('=')[1]
                if src in self.map:
                    self.map[dest] = self.map[src]
                else:
                    warn_press_key(["Warning: You wanted to copy parameter from %s to %s, " % (src, dest), 
                                    "but the source parameter does not seem to exist!"])
                self.assign_field(self.map[dest],ffname,fflist.index(e),dest.split('/')[1],1)
            
    def make(self,vals,usepvals=False,printdir=None,precision=12):
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
        @param[in] usepvals Switch for whether to bypass the coordinate transformation
        and use physical parameters directly.
        
        """
        if usepvals:
            pvals = vals.copy()
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
        newffdata = deepcopy(self.ffdata)
        #============================#
        # Print the new force field. #
        #============================#
        for i in range(len(self.pfields)):
            pfld_list = self.pfields[i]
            for pfield in pfld_list:
                fnm,ln,fld,mult = pfield
                # XML force fields are easy to print.  
                # Our 'pointer' to where to replace the value
                # is given by the position of this line in the
                # iterable representation of the tree and the
                # field number.
                if type(newffdata[fnm]) is etree._ElementTree:
                    list(newffdata[fnm].iter())[ln].attrib[fld] = OMMFormat % (mult*pvals[i])
                # Text force fields are a bit harder.
                # Our pointer is given by the line and field number.
                # We take care to preserve whitespace in the printout
                # so that the new force field still has nicely formated
                # columns.
                else:
                    # Split the string into whitespace and data fields.
                    sline       = self.R[fnm].Split(newffdata[fnm][ln])
                    whites      = self.R[fnm].Whites(newffdata[fnm][ln])
                    # Align whitespaces and fields (it should go white, field, white, field)
                    if len(whites) == len(sline) - 1: 
                        whites = [''] + whites
                    # Subtract one whitespace, unless the line begins with a minus sign.
                    if not match('^-',sline[fld]) and len(whites[fld]) > 1:
                        whites[fld] = whites[fld][:-1]
                    # Subtract whitespace equal to (length of the data field minus two).
                    if len(whites[fld]) > len(sline[fld])+2:
                        whites[fld] = whites[fld][:len(sline[fld])+2]
                    # Actually replace the field with the physical parameter value.
                    if precision == 12:
                        sline[fld]  = "% 17.12e" % (mult*pvals[i])
                    else:
                        #sline[fld]  = TXTFormat % (mult*pvals[i])
                        sline[fld]  = TXTFormat(mult*pvals[i], precision)
                    # Replace the line in the new force field.
                    newffdata[fnm][ln] = ''.join([whites[j]+sline[j] for j in range(len(sline))])+'\n'

        if printdir != None:
            absprintdir = os.path.join(self.root,printdir)
        else:
            absprintdir = os.getcwd()

        if not os.path.exists(absprintdir):
            print 'Creating the directory %s to print the force field' % absprintdir
            os.makedirs(absprintdir)

        for fnm in newffdata:
            if type(newffdata[fnm]) is etree._ElementTree:
                with open(os.path.join(absprintdir,fnm),'w') as f: newffdata[fnm].write(f)
            else:
                with open(os.path.join(absprintdir,fnm),'w') as f: f.writelines(newffdata[fnm])
        return pvals
        
    def create_pvals(self,mvals):
        """Converts mathematical to physical parameters.

        First, mathematical parameters are rescaled and rotated by
        multiplying by the transformation matrix, followed by adding
        the original physical parameters.

        @param[in] mvals The mathematical parameters
        @return pvals The physical parameters
        
        """
        if self.logarithmic_map:
            pvals = exp(mvals) * self.pvals0
        else:
            pvals = flat(mat(self.tmI)*col(mvals)) + self.pvals0
        concern= ['polarizability','epsilon','VDWT']
        # Guard against certain types of parameters changing sign.
        for i in range(self.np):
            if any([j in self.plist[i] for j in concern]) and pvals[i] * self.pvals0[i] < 0:
                print "Parameter %s has changed sign but it's not allowed to! Setting to zero." % self.plist[i]
                pvals[i] = 0.0
        return pvals

    def create_mvals(self,pvals):
        """Converts physical to mathematical parameters.

        We create the inverse transformation matrix using SVD.

        @param[in] pvals The physical parameters
        @return mvals The mathematical parameters
        """
        if self.logarithmic_map:
            raise Exception('create_mvals has not been implemented for logarithmic_map')
        mvals = flat(invert_svd(self.tmI) * col(pvals - self.pvals0))
        return mvals
        
    def rsmake(self,printfacs=True):
        """Create the rescaling factors for the coordinate transformation in parameter space.

        The proper choice of rescaling factors (read: prior widths in maximum likelihood analysis)
        is still a black art.  This is a topic of current research.

        @todo Pass in rsfactors through the input file

        @param[in] printfacs List for printing out the resecaling factors

        """
        typevals = {}
        rsfactors = {}
        rsfac_list = []
        ## Takes the dictionary 'BONDS':{3:'B', 4:'K'}, 'VDW':{4:'S', 5:'T'},
        ## and turns it into a list of term types ['BONDSB','BONDSK','VDWS','VDWT']
        if any([self.R[i].pdict == "XML_Override" for i in self.fnms]):
            termtypelist = ['/'.join([i.split('/')[0],i.split('/')[1]]) for i in self.map]
        else:
            termtypelist = itertools.chain(*sum([[[i+self.R[f].pdict[i][j] for j in self.R[f].pdict[i] if isint(str(j))] for i in self.R[f].pdict] for f in self.fnms],[]))
            #termtypelist = sum([[i+self.R.pdict[i][j] for j in self.R.pdict[i] if isint(str(j))] for i in self.R.pdict],[])
        for termtype in termtypelist:
            for pid in self.map:
                if termtype in pid:
                    typevals.setdefault(termtype, []).append(self.pvals0[self.map[pid]])
        for termtype in typevals:
            # The old, horrendously complicated rule
            # rsfactors[termtype] = exp(mean(log(abs(array(typevals[termtype]))+(abs(array(typevals[termtype]))==0))))
            # The newer, maximum rule (thanks Baba)
            rsfactors[termtype] = max(abs(array(typevals[termtype])))
            rsfac_list.append(termtype)
            # Physically motivated overrides
            rs_override(rsfactors,termtype)
        # Overrides from input file
        for termtype in self.priors:
            rsfac_list.append(termtype)
            rsfactors[termtype] = self.priors[termtype]
    
        # for line in os.popen("awk '/rsfactor/ {print $2,$3}' %s" % pkg.options).readlines():
        #     rsfactors[line.split()[0]] = float(line.split()[1])
        if printfacs:
            bar = printcool("Rescaling Factors (Lower Takes Precedence):",color=1)
            print '\n'.join(["   %-35s  : %.5e" % (i, rsfactors[i]) for i in rsfac_list])
            print bar
        ## The array of rescaling factors
        self.rs = ones(len(self.pvals0))
        for pnum in range(len(self.pvals0)):
            for termtype in rsfac_list:
                if termtype in self.plist[pnum]:
                    self.rs[pnum] = rsfactors[termtype]
                    
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
        qmat2 = eye(self.np,dtype=float)

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
            nq = len(qmap)
            cons0 = ones((1,tq),dtype=float)
            cons = zeros((cons0.shape[0], nq), dtype=float)
            qtrans2 = eye(nq, dtype=float)
            for i in range(cons.shape[0]):
                for j in range(cons.shape[1]):
                    cons[i][j] = sum([cons0[i][k-1] for k in qid[j]])
                cons[i] /= norm(cons[i])
                for j in range(i):
                    cons[i] = orthogonalize(cons[i], cons[j])
                qtrans2[i,:] = 0
                for j in range(nq-i-1):
                    qtrans2[i+j+1, :] = orthogonalize(qtrans2[i+j+1, :], cons[i])
            return qtrans2
        # Here we build a charge constraint for each molecule.
        if any(len(r.adict) > 0 for r in self.R.values()):
            print "Building charge constraints..."
            # Build a concatenated dictionary
            Adict = OrderedDict()
            # This is a loop over files
            for r in self.R.values():
                # This is a loop over molecules
                for k, v in r.adict.items():
                    Adict[k] = v
            nmol = 0
            for molname, molatoms in Adict.items():
                mol_charge_count = zeros(self.np, dtype=float)
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
                        print "Parameter %i occurs %i times in molecule %s in locations %s (%s)" % (i, qct, molname, str(qidx), self.plist[i])
                #Here is where we build the qtrans2 matrix.
                if len(qmap) > 0:
                    qtrans2 = build_qtrans2(tq, qid, qmap)
                    if self.constrain_charge:
                        insert_mat(qtrans2, qmap)
                if nmol == 0:
                    self.qid = qid
                    self.qmap = qmap
                else:
                    print "Note: ESP fitting will be performed assuming that molecule id %s is the FIRST molecule and the only one being fitted." % molname
                nmol += 1
        elif self.constrain_charge:
            warn_press_key("'adict' {molecule:atomnames} was not found.\n This isn't a big deal if we only have one molecule, but might cause problems if we want multiple charge neutrality constraints.")
            qnr = 0
            if any([self.R[i].pdict == "XML_Override" for i in self.fnms]):
                # Hack to count the number of atoms for each atomic charge parameter, when the force field is an XML file.
                # This needs to be changed to Chain or Molecule
                ListOfAtoms = list(itertools.chain(*[[e.get('type') for e in self.ffdata[k].getroot().xpath('//Residue/Atom')] for k in self.ffdata]))
            for i in range(self.np):
                if any([j in self.plist[i] for j in concern]):
                    self.qmap.append(i)
                    if 'AmoebaMultipoleForce.Multipole/c0' in self.plist[i] or 'NonbondedForce.Atom/charge' in self.plist[i]:
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
                            self.qid2.append(array([self.atomnames.index(k) for k in thisq]))
                        except: pass
                        nq = sum(array([count(self.plist[i], j) for j in concern]))
                    self.qid.append(qnr+arange(nq))
                    qnr += nq
            if len(self.qid2) == 0:
                sys.stderr.write('Unable to match atom numbers up with atom names (minor issue, unless doing ESP fitting).  \nAre atom names implemented in the force field parser?\n')
            else:
                self.qid = self.qid2
            tq = qnr - 1
            #Here is where we build the qtrans2 matrix.
            if len(self.qmap) > 0:
                cons0 = ones((1,tq))
                qtrans2 = build_qtrans2(tq, self.qid, self.qmap)
                # Insert qtrans2 into qmat2.
                if self.constrain_charge:
                    insert_mat(qtrans2, self.qmap)

        # print "Charge parameter constraint matrix - feel free to check it"
        # for i in qmat2:
        #     for j in i:
        #         print "% .3f" % j,
        #     print
        # print

        transmat = mat(qmat2) * diag(self.rs)
        transmatNS = array(transmat,copy=True)
        self.excision = []
        for i in range(self.np):
            if abs(transmatNS[i, i]) < 1e-8:
                self.excision.append(i)
                transmatNS[i, i] += 1
        self.excision = list(set(self.excision))
        for i in self.excision:
            transmat[i, :] = zeros(self.np, dtype=float)
        self.tm = transmat
        self.tmI = transmat.T
        
    def list_map(self):
        """ Create the plist, which is like a reversed version of the parameter map.  More convenient for printing. """
        self.plist = [[] for j in range(max([self.map[i] for i in self.map])+1)]
        for i in self.map:
            self.plist[self.map[i]].append(i)
        for i in range(self.np):
            self.plist[i] = ' '.join(self.plist[i])
            
    def print_map(self,vals = None):
        """Prints out the (physical or mathematical) parameter indices, IDs and values in a visually appealing way."""
        if vals == None:
            vals = self.pvals0
        print '\n'.join(["%4i [ %s ]" % (self.plist.index(i), "% .4e" % float(vals[self.plist.index(i)]) if isfloat(str(vals[self.plist.index(i)])) else (str(vals[self.plist.index(i)]))) + " : " + "%s" % i for i in self.plist])
        
    def assign_p0(self,idx,val):
        """ Assign physical parameter values to the 'pvals0' array.

        @param[in] idx The index to which we assign the parameter value.
        @param[in] val The parameter value to be inserted.
        """
        if idx == len(self.pvals0):
            self.pvals0.append(val)
        else:
            self.pvals0[idx] = val
            
    def assign_field(self,idx,fnm,ln,pfld,mult):
        """ Record the locations of a parameter in a txt file; [[file name, line number, field number, and multiplier]].

        Note that parameters can have multiple locations because of the repetition functionality.

        @param[in] idx  The index of the parameter.
        @param[in] fnm  The file name of the parameter field.
        @param[in] ln   The line number within the file (or the node index in the flattened xml)
        @param[in] pfld The field within the line (or the name of the attribute in the xml)
        @param[in] mult The multiplier (this is usually 1.0)
        
        """
        if idx == len(self.pfields):
            self.pfields.append([[fnm,ln,pfld,mult]])
        else:
            self.pfields[idx].append([fnm,ln,pfld,mult])

def rs_override(rsfactors,termtype,Temperature=298.15):
    """ This function takes in a dictionary (rsfactors) and a string (termtype).
    
    If termtype matches any of the strings below, rsfactors[termtype] is assigned
    to one of the numbers below.

    This is LPW's attempt to simplify the rescaling factors.

    @param[out] rsfactors The computed rescaling factor.
    @param[in] termtype The interaction type (corresponding to a physical unit)
    @param[in] Temperature The temperature for computing the kT energy scale
    
    """
    if match('PDIHS[1-6]K|RBDIHSK[1-5]|MORSEC',termtype):
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
