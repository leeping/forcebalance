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
@date 12/2011

"""

import os
import sys
from re import match, sub, split
import gmxio
import qchemio
import custom_io
import basereader
from numpy import arange, array, diag, exp, eye, log, mat, mean, ones, vstack, zeros
from numpy.linalg import norm
from nifty import col, flat, invert_svd, kb, orthogonalize, pmat2d, printcool, row
from string import count

FF_Extensions = {"itp" : "gmx",
                 "in"  : "qchem",
                 "gen" : "custom"
                 }

""" Recognized force field formats. """
FF_IOModules = {"gmx": gmxio.ITP_Reader ,
                "qchem": qchemio.QCIn_Reader ,
                "custom": custom_io.Gen_Reader
                }

def determine_fftype(ffname):
    fsplit = ffname.split('/')[-1].split(':')
    fftype = None
    print "Determining file type of %s ..." % fsplit[0],
    if len(fsplit) == 2:
        if fsplit[1] in FF_IOModules:
            print "We're golden! (%s)" % fsplit[1]
            fftype = fsplit[1]
        else:
            print "\x1b[91m Warning: \x1b[0m %s not in supported types (%s)!" % (fsplit[1],', '.join(keys(FF_IOModules)))
    elif len(fsplit) == 1:
        print "Guessing from extension (you may specify type with filename:type) ...", 
        ffname = fsplit[0]
        ffext = ffname.split('.')[-1]
        if ffext in FF_Extensions:
            guesstype = FF_Extensions[ffext]
            if guesstype in FF_IOModules:
                print "guessing %s -> %s!" % (ffext, guesstype)
                fftype = guesstype
            else:
                print "\x1b[91m Warning: \x1b[0m %s not in supported types (%s)!" % (fsplit[1],', '.join(keys(FF_IOModules)))
        else:
            print "\x1b[91m Warning: \x1b[0m %s not in supported extensions (%s)!" % (ffext,', '.join(keys(FF_Extensions)))
    if fftype == None:
        print "Force field type not determined! Exiting..."
        sys.exit(1)
    return fftype

class FF(object):
    """ Force field class.

    This class contains all methods for force field manipulation.
    To create an instance of this class, an input file is required
    containing the list of force field file names.  Everything else
    inside this class pertaining to force field generation is self-contained.

    For details on force field parsing, see the detailed documentation for addff.
    
    """
    def __init__(self, options):

        """Instantiation of force field class.

        Many variables here are initialized to zero, but they are filled out by
        methods like addff, rsmake, and mktransmat.
        
        """
        #======================================#
        # Options that are given by the parser #
        #======================================#

        ## The root directory of the project
        self.root        = os.getcwd()
        ## File names of force fields
        self.fnms        = options['forcefield']
        ## Directory containing force fields, relative to project directory
        self.ffdir       = options['ffdir']
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        # A lot of these variables are filled out by calling the methods.
        
        ## The content of all force field files are stored in memory
        self.stuff       = {}        
        ## The mapping of interaction type -> parameter number
        self.map         = {}
        ## The listing of parameter number -> interaction types
        self.plist       = []
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
        ## The force field content, but with parameter fields replaced with new parameters.
        self.newstuff    = {}
        ## Initial value of physical parameters
        self.pvals0      = []
        
        # Read the force fields into memory.
        for fnm in self.fnms:
            print "Reading force field from file: %s" % fnm
            self.addff(fnm)

        # Set the initial values of parameter arrays.
        ## Initial value of physical parameters
        self.pvals0 = array(self.pvals0)

        # Prepare various components of the class for first use.
        ## Creates plist from map.
        self.list_map()
        ## Prints the plist to screen.
        bar = printcool("Starting parameter indices, physical values and IDs")
        self.print_map()                       
        print bar
        ## Make the rescaling factors.
        self.rsmake(printfacs=True)            
        ## Make the transformation matrix.
        self.mktransmat()                      
        
    def addff(self,ffname):
        """ Parse a force field file and add it to the class.

        First, we need to figure out the type of file file.  Currently this is done
        using the three-letter file extension ('.itp' = gmx); that can be improved.
        
        First we open the force field file and read all of its lines.  As we loop
        through the force field file, we look for two types of tags: (1) section
        markers, in GMX indicated by [ section_name ], which allows us to determine
        the section, and (2) parameter tags, indicated by the 'PARM' or 'RPT' keywords.

        As we go through the file, we figure out the atoms involved in the interaction
        described on each line.

        @todo This can be generalized to parameters that don't correspond to atoms.
        @todo Fix the special treatment of NDDO.

        When a 'PARM' keyword is indicated, it is followed by a number which is the field
        in the line to be modified, starting with zero.  Based on the field number and the
        section name, we can figure out the parameter type.  With the parameter type
        and the atoms in hand, we construct a 'parameter identifier' or pid which uniquely
        identifies that parameter.  We also store the physical parameter value in an array
        called 'pvals0' and the precise location of that parameter (by filename, line number,
        and field number) in a list called 'pfields'.

        An example: Suppose in 'my_ff.itp' I encounter the following on lines 146 and 147:
        
        @code
        [ angletypes ]
        CA   CB   O   1   109.47  350.00  ; PARM 4 5
        @endcode

        From reading <tt>[ angletypes ]</tt> I know I'm in the 'angletypes' section.

        On the next line, I notice two parameters on fields 4 and 5.

        From the atom types, section type and field number I know the parameter IDs are <tt>'ANGLESBCACBO'</tt> and <tt>'ANGLESKCACBO'</tt>.
        
        After building <tt>map={'ANGLESBCACBO':1,'ANGLESKCACBO':2}</tt>, I store the values in
        an array: <tt>pvals0=array([109.47,350.00])</tt>, and I put the parameter locations in
        pfields: <tt>pfields=[['my_ff.itp',147,4,1.0],['my_ff.itp',146,5,1.0]]</tt>.  The 1.0
        is a 'multiplier' and I will explain it below.
        
        Note that in the creation of parameter IDs, we run into the issue that the atoms
        involved in the interaction may be labeled in reverse order (e.g. <tt>OCACB</tt>).  Thus,
        we store both the normal and the reversed parameter ID in the map.

        Parameter repetition and multiplier:
        
        If <tt>'RPT'</tt> is encountered in the line, it is always in the syntax:
        <tt>'RPT 4 ANGLESBCACAH 5 MINUS_ANGLESKCACAH /RPT'</tt>.  In this case, field 4 is replaced by
        the stored parameter value corresponding to <tt>ANGLESBCACAH</tt> and field 5 is replaced by
        -1 times the stored value of <tt>ANGLESKCACAH</tt>.  Now I just picked this as an example,
        I don't think people actually want a negative angle force constant .. :) the <tt>MINUS</tt>
        keyword does come in handy for assigning atomic charges and virtual site positions.
        In order to achieve this, a multiplier of -1.0 is stored into pfields instead of 1.0.

        @todo Note that I can also create the opposite virtual site position by changing the atom
        labeling, woo!

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
        
        Reader = FF_IOModules.get(fftype,basereader.BaseReader)

        # Open the force field using an absolute path and read its contents into memory.
        absff = os.path.join(self.root,self.ffdir,ffname)
        self.stuff[ffname] = open(absff).readlines()
        
        R = Reader(ffname)
        for ln, line in enumerate(self.stuff[ffname]):
            R.feed(line)
            sline = line.split()
            if 'PARM' in sline:
                pmark = (array(sline) == 'PARM').argmax() # The position of the 'PARM' word
                pflds = [int(i) for i in sline[pmark+1:]] # The integers that specify the parameter word positions
                for pfld in pflds:
                    # For each of the fields that are to be parameterized (indicated by PARM #),
                    # assign a parameter type to it according to the Interaction Type -> Parameter Dictionary.
                    pid = R.build_pid(pfld)
                    # Add pid into the dictionary.
                    self.map[pid] = self.np
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
                    prep = self.map[sline[parse+1].replace('MINUS_','')]
                    pid = R.build_pid(pfld)
                    self.map[pid] = prep
                    self.assign_field(prep,ffname,ln,pfld,"MINUS_" in sline[parse+1] and -1 or 1)
                    parse += 2
        
    def make(self,printdir,vals,usepvals):
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
        self.replace_pvals(pvals)
        self.print_newstuff(printdir)
        return pvals
        
    def create_pvals(self,mvals):
        """Converts mathematical to physical parameters.

        First, mathematical parameters are rescaled and rotated by
        multiplying by the transformation matrix, followed by adding
        the original physical parameters.

        @param[in] mvals The mathematical parameters
        @return pvals The physical parameters
        
        """
        pvals = flat(mat(self.tmI)*col(mvals)) + self.pvals0
        return pvals

    def create_mvals(self,pvals):
        """Converts physical to mathematical parameters.

        We create the inverse transformation matrix using SVD.

        @param[in] pvals The physical parameters
        @return mvals The mathematical parameters
        """
        mvals = flat(invert_svd(self.tmI) * col(pvals - self.pvals0))
        return mvals
        
    def replace_pvals(self,pvals):
        """Replaces numerical fields in stored force field files with the stored physical parameter values.
        
        Unless you really know what you're doing, you probably shouldn't be calling this directly.

        """
        vals = list(pvals)
        for fnm in self.stuff:
            self.newstuff[fnm] = list(self.stuff[fnm])
        for i in range(len(self.pfields)):
            pfld_list = self.pfields[i]
            for pfield in pfld_list:
                fnm,ln,fld,mult = pfield
                sline       = self.newstuff[fnm][ln].split()
                whites      = split('[^ ]+',self.newstuff[fnm][ln])
                if not match('^-',sline[fld]) and len(whites[fld]) > 1:
                    whites[fld] = whites[fld][:-1]
                sline[fld]  = "% .12e" % (mult*vals[i])
                self.newstuff[fnm][ln] = ''.join([whites[j]+sline[j] for j in range(len(sline))])+'\n'
                
    def print_newstuff(self,printdir):
        """Prints out the new content of force fields to files in 'printdir'.

        @param[in] printdir The directory to which new force fields are printed.

        """
        if not os.path.exists(os.path.join(self.root,printdir)):
            os.makedirs(os.path.join(self.root,printdir))
        for fnm in self.newstuff:
            with open(os.path.join(self.root,printdir,fnm),'w') as f: f.writelines(self.newstuff[fnm])
            
    def rsmake(self,printfacs=True):
        """Create the rescaling factors for the coordinate transformation in parameter space.

        The proper choice of rescaling factors (read: prior widths in maximum likelihood analysis)
        is still a black art.  This is a topic of current research.

        @param[in] printfacs List for printing out the resecaling factors

        """
        typevals = {}
        tvgeomean = {}
        ## Takes the dictionary 'BONDS':{3:'B', 4:'K'}, 'VDW':{4:'S', 5:'T'},
        ## and turns it into a list of term types ['BONDSB','BONDSK','VDWS','VDWT']
        termtypelist = sum([[i+gmxio.pdict[i][j] for j in gmxio.pdict[i]] for i in gmxio.pdict],[])
        for termtype in termtypelist:
            for pid in self.map:
                if termtype in pid:
                    typevals.setdefault(termtype, []).append(self.pvals0[self.map[pid]])
        for termtype in typevals:
            # The old, horrendously complicated rule
            # tvgeomean[termtype] = exp(mean(log(abs(array(typevals[termtype]))+(abs(array(typevals[termtype]))==0))))
            # The newer, maximum rule (thanks Baba)
            tvgeomean[termtype] = max(abs(array(typevals[termtype])))
            # Physically motivated overrides
            rs_override(tvgeomean,termtype)
        # TODO: Pass in rsfactors through the input file
        # for line in os.popen("awk '/rsfactor/ {print $2,$3}' %s" % pkg.options).readlines():
        #     tvgeomean[line.split()[0]] = float(line.split()[1])
        if printfacs:
            bar = printcool("Rescaling Factors for Different Parameter Types:",color=1)
            print '\n'.join(sorted(["   %-15s  : %.5e" % (i, tvgeomean[i]) for i in tvgeomean]))
            print bar
        ## The array of rescaling factors
        self.rs = ones(len(self.pvals0))
        for pnum in range(len(self.pvals0)):
            for termtype in tvgeomean:
                if termtype in self.plist[pnum]:
                    self.rs[pnum] = tvgeomean[termtype]
                    
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
        """
        qmap   = []
        qid    = []
        qnr    = 1
        concern= ['COUL']
        for i in range(self.np):
            if any([j in self.plist[i] for j in concern]):
                qmap.append(i)
                nq = sum(array([count(self.plist[i], j) for j in concern]))
                qid.append(qnr+arange(nq))
                qnr += nq
        tq = qnr - 1
        cons0 = ones((1,tq))
        #print qmap
        #print qid
        #chargegrp = []
        # LPW Charge groups aren't implemented at this time
        ## chargegrp = [[1,3],[4,5],[6,7]]
        ## for group in chargegrp:
        ##     a = min(group[0]-1, group[1])
        ##     b = max(group[0]-1, group[1])
        ##     constemp = zeros(tq, dtype=float)
        ##     for i in range(constemp.shape[0]):
        ##         if i >= a and i < b:
        ##             constemp[i] += 1
        ##         else:
        ##             constemp[i] -= 1
        ##     cons0 = vstack((cons0, constemp))
        ## print cons0
        #Here is where we build the qtrans2 matrix.
        
        nq = len(qmap)
        if nq > 0:
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
            #print qtrans2
        qmat2 = eye(self.np,dtype=float)
        x = 0
        for i in range(self.np):
            if i in qmap:
                y = 0
                for j in qmap:
                    qmat2[i, j] = qtrans2[x, y]
                    y += 1
                x += 1
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
        print '\n'.join(["%4i [ % .4e ]" % (self.plist.index(i),vals[self.plist.index(i)]) + " : " + "%s" % i for i in self.plist])
        
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
        """ Record the locations of a parameter in a file; [[file name, line number, field number, and multiplier]].

        Note that parameters can have multiple locations because of the repetition functionality.

        @param[in] idx  The index of the parameter.
        @param[in] fnm  The file name of the parameter field.
        @param[in] ln   The line number within the file.
        @param[in] pfld The field within the line.
        @param[in] mult The multiplier (this is usually 1.0)
        
        """
        if idx == len(self.pfields):
            self.pfields.append([[fnm,ln,pfld,mult]])
        else:
            self.pfields[idx].append([fnm,ln,pfld,mult])

def rs_override(tvgeomean,termtype,Temperature=298.15):
    """ This function takes in a dictionary (tvgeomean) and a string (termtype).
    
    If termtype matches any of the strings below, tvgeomean[termtype] is assigned
    to one of the numbers below.

    This is LPW's attempt to simplify the rescaling factors.

    @param[out] tvgeomean The computed rescaling factor.
    @param[in] termtype The interaction type (corresponding to a physical unit)
    @param[in] Temperature The temperature for computing the kT energy scale
    
    """
    if match('PDIHS[1-6]K|RBDIHSK[1-5]|MORSEC',termtype):
        # eV or eV rad^-2
        tvgeomean[termtype] = 96.4853
    elif match('UREY_BRADLEYK1|ANGLESK',termtype):
        tvgeomean[termtype] = 96.4853 * 6.28
    elif match('COUL|VPAIR_BHAMC|QTPIEA',termtype):
        # elementary charge, or unitless, or already in atomic unit
        tvgeomean[termtype] = 1.0
    elif match('QTPIEC|QTPIEH',termtype):
        # eV to atomic unit
        tvgeomean[termtype] = 27.2114
    elif match('BONDSB|UREY_BRADLEYB|MORSEB|VDWS|VPAIRS|VSITE|VDW_BHAMA|VPAIR_BHAMA',termtype):
        # nm to atomic unit
        tvgeomean[termtype] = 0.05291772
    elif match('BONDSK|UREY_BRADLEYK2',termtype):
        # au bohr^-2
        tvgeomean[termtype] = 34455.5275 * 27.2114
    elif match('PDIHS[1-6]B|ANGLESB|UREY_BRADLEYB',termtype):
        # radian
        tvgeomean[termtype] = 57.295779513
    elif match('VDWT|VDW_BHAMB|VPAIR_BHAMB',termtype):
        # VdW well depth; using kT.  This was a tough one because the energy scale is so darn small.
        tvgeomean[termtype] = kb*Temperature
    elif match('MORSEE',termtype):
        tvgeomean[termtype] = 18.897261
