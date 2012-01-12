"""@package molecule

Lee-Ping's convenience library for parsing molecule file formats.
Motivation: I'm not quite sure what OpenBabel does, don't fully trust it
I need to be very careful with Q-Chem input files and use input templates
I'd like to replace the grordr module because it's so old.

Currently i can read xyz, gro, com, and arc
Currently i can write xyz, gro, qcin (with Q-Chem template file)

Not sure what's next on the chopping block, but please stay consistent when adding new methods.

@author Lee-Ping Wang
@date 12/2011
"""

from re import match, sub, split
from sys import argv, exit, stdout
from PT import PT
from nifty import isint, isfloat
from numpy import array

def format_xyz_coord(element,xyz,tinker=False):
    """ Print a line consisting of (element, x, y, z) in accordance with .xyz file format

    @param[in] element A chemical element of a single atom
    @param[in] xyz A 3-element array containing x, y, z coordinates of that atom

    """
    if tinker:
        return "%-3s% 12.6f% 12.6f% 12.6f" % (element,xyz[0],xyz[1],xyz[2])
    else:
        return "%-5s% 16.10f% 16.10f% 16.10f" % (element,xyz[0],xyz[1],xyz[2])

def format_gro_coord(resid, resname, aname, seqno, xyz):
    """ Print a line in accordance with .gro file format

    @param[in] resid The number of the residue that the atom belongs to
    @param[in] resname The name of the residue that the atom belongs to
    @param[in] aname The name of the atom
    @param[in] seqno The sequential number of the atom
    @param[in] xyz A 3-element array containing x, y, z coordinates of that atom
    
    """
    return "%5i%-5s%5s%5i% 11.6f% 11.6f% 11.6f" % (resid,resname,aname,seqno,xyz[0],xyz[1],xyz[2])

def format_gro_box(xyz):
    """ Print a line corresponding to the box vector in accordance with .gro file format

    @param[in] xyz A 3-element or 9-element array containing the box vectors
    
    """
    return ' '.join(["%11.6f" % (i/10) for i in xyz])

def is_gro_coord(line):
    """ Determines whether a line contains GROMACS data or not

    @param[in] line The line to be tested
    
    """
    sline = line.split()
    if len(sline) == 6:
        return all([isint(sline[2]),isfloat(sline[3]),isfloat(sline[4]),isfloat(sline[5])])
    elif len(sline) == 5:
        return all([isint(line[15:20]),isfloat(sline[2]),isfloat(sline[3]),isfloat(sline[4])])
    else:
        return 0

def is_charmm_coord(line):
    """ Determines whether a line contains CHARMM data or not

    @param[in] line The line to be tested
    
    """
    sline = line.split()
    if len(sline) >= 7:
        return all([isint(sline[0]), isint(sline[1]), isfloat(sline[4]), isfloat(sline[5]), isfloat(sline[6])])
    else:
        return 0

def is_gro_box(line):
    """ Determines whether a line contains a GROMACS box vector or not

    @param[in] line The line to be tested
    
    """
    sline = line.split()
    if len(sline) == 9 and all([isfloat(i) for i in sline]):
        return 1
    elif len(sline) == 3 and all([isfloat(i) for i in sline]):
        return 1
    else:
        return 0

class Molecule:
    """ The Molecule class contains information about a system of molecules.

    This is a centralized container of information that is useful for
    file conversion and processing.  It is quite flexible but not
    perfectly flexible.  For example, we can accommodate multiple sets
    of xyz coordinates (i.e. multiple frames in a trajectory), but we
    only allow a single set of atomic elements (i.e. can't have
    molecules changing identities across frames.)

    """
    def __init__(self, fnm = None, ftype = None):
        """ To instantiate the class we simply define the table of file reading/writing functions
        and read in a file if it is provided."""
        
        ## The table of file readers
        self.Read_Tab = {'com':self.read_com,
                         'gro':self.read_gro,
                         'crd':self.read_charmm,
                         'cor':self.read_charmm,
                         'xyz':self.read_xyz,
                         'arc':self.read_arc}
        ## The table of file writers
        self.Write_Tab = {'gro':self.write_gro,
                          'xyz':self.write_xyz,
                          'in':self.write_qcin,
                          'arc':self.write_arc}
        ## If supplied with a file name, read in stuff.
        if fnm != None:
            self.read(fnm, ftype)
            
        ## Initialize the index file here
        self.index = {}

    def read(self, fnm, ftype = None):
        """ Read in a file.  This populates the attributes of the class.  For now, I don't know
        what will happen if we read two files into the class.  Probably unexpected behavior will
        occur."""
        if ftype == None:
            ## File type can be determined from the file name using the extension.
            ftype = fnm.split('.')[-1]

        Answer = self.Read_Tab.get(ftype,self.oops)(fnm)

        if 'elem' in Answer:
            ## Absolutely required; a list of chemical elements (number of atoms)
            self.elem     = Answer['elem']
        else:
            print "No list of elements! We cannot continue, blargggggh"
            exit(1)

        if 'xyzs' in Answer:
            ## Absolutely required; a list of lists of xyz coordinates (number of frames x number of atoms)
            self.xyzs     = Answer['xyzs']
            ## The number of snapshots is determined by the length of self.xyzs
            self.ns       = len(self.xyzs)
        else:
            print "No list of xyz coordinates! We cannot continue, eurk"
            exit(1)

        if 'comms' in Answer:
            ## The comments that usually go somewhere into the output file
            self.comms    = Answer['comms']
        else:
            print "No list of comments, making defaults"
            self.comms = ['Generated by filecnv.py from %s: frame %i of %i' % (fnm, i+1, self.ns) for i in range(self.ns)]

        ## Optional variable: the charge
        self.charge = Answer['charge'] if 'charge' in Answer else None
        ## Optional variable: the multiplicity
        self.mult = Answer['mult'] if 'mult' in Answer else None
        ## Optional variable: the box vectors (there would be self.ns of these)
        self.boxes = Answer['boxes'] if 'boxes' in Answer else None
        ## Optional variable: the residue number
        self.resid = Answer['resid'] if 'resid' in Answer else None
        ## Optional variable: the residue name
        self.resname = Answer['resname'] if 'resname' in Answer else None
        ## Optional variable: the atom name, which defaults to the elements
        self.atomname = Answer['atomname'] if 'atomname' in Answer else Answer['elem']
        ## Optional variable: raw .arc file in Tinker (because Tinker files are too hard to interpret!)
        self.rawarcs = Answer['rawarcs'] if 'rawarcs' in Answer else None

    def na(self):
        return len(self.elem)

    def write(self,fnm=None,ftype=None,append=False,tempfnm=None,subset=None):
        if fnm == None and ftype == None:
            print "Without a file name or a file type, what do you expect me to do?"
            exit(1)
        elif ftype == None:
            ftype = fnm.split(".")[-1]

        ## Name of the template file
        self.tempfnm = tempfnm
        out_lines = self.Write_Tab.get(ftype,self.oops)(subset)

        if fnm == None:
            outfile = stdout
        elif append:
            outfile = open(fnm,'a')
        else:
            outfile = open(fnm,'w')
        for line in out_lines:
            print >> outfile,line
        outfile.close()
        
    def oops(self,fnm = None):
        print "Yargh: Crashed when trying to %s" % (fnm == None and "write" or "read %s" % fnm)
        exit(1)

    def read_xyz(self,fnm):
        """ Parse a .xyz file which contains several xyz coordinates, and return their elements.

        @param[in] fnm The input XYZ file name
        @return elem  A list of chemical elements in the XYZ file
        @return comms A list of comments.
        @return xyzs  A list of XYZ coordinates (number of snapshots times number of atoms)

        """
        xyz   = []
        xyzs  = []
        comms = []
        elem  = []
        an    = 0
        for line in open(fnm):
            strip = line.strip()
            sline = line.split()
            if match('^[0-9]+$',strip):
                na = int(strip)
            elif match('[A-Z][a-z]*( +[-+]?([0-9]*\.[0-9]+|[0-9]+)){3}$',strip):
                xyz.append([float(i) for i in sline[1:]])
                if len(elem) < na:
                    elem.append(sline[0])
                an += 1
                if an == na:
                    xyzs.append(array(xyz))
                    xyz = []
                    an  = 0
            elif an == 0:
                comms.append(strip)
        Answer = {'elem' : elem,
                  'xyzs' : xyzs,
                  'comms': comms}
        return Answer

    def read_com(self,fnm):
        """ Parse a Gaussian .com file and return a SINGLE-ELEMENT list of xyz coordinates (no multiple file support)

        """
        elem    = []
        xyz     = []
        ln      = 0
        comfile = open(fnm).readlines()
        inxyz = 0
        for line in comfile:
            # Everything after exclamation point is a comment
            sline = line.split('!')[0].split()
            if len(sline) == 2:
                if isint(sline[0]) and isint(sline[1]):
                    charge = int(sline[0])
                    mult = int(sline[1])
                    title_ln = ln - 2
            elif len(sline) == 4:
                inxyz = 1
                if sline[0].capitalize() in PT and isfloat(sline[1]) and isfloat(sline[2]) and isfloat(sline[3]):
                    elem.append(sline[0])
                    xyz.append(array([float(sline[1]),float(sline[2]),float(sline[3])]))
            elif inxyz:
                break
            ln += 1

        Answer = {'xyzs'   : [array(xyz)],
                  'elem'   : elem,
                  'comms'  : [comfile[title_ln].strip()],
                  'charge' : charge,
                  'mult'   : mult}
        return Answer

    def read_arc(self,fnm):
        """ Read a TINKER .arc file.
        
        """
        rawarcs  = []
        rawarc   = []
        suffix   = []
        xyzs  = []
        xyz   = []
        resid = []
        elem  = []
        comms = []
        thisres = set([])
        forwardres = set([])
        title = True
        nframes = 0
        thisresid   = 1
        for line in open(fnm):
            rawarc.append(line.strip())
            sline = line.split()
            # The first line always contains the number of atoms
            if title:
                na = int(sline[0])
                comms.append(' '.join(sline[1:]))
                title = False
            elif len(sline) >= 6:
                #print sline
                if isint(sline[0]) and isfloat(sline[2]) and isfloat(sline[3]) and isfloat(sline[4]): # A line of data better look like this
                    if nframes == 0:
                        elem.append(sline[1])
                        resid.append(thisresid)
                        whites      = split('[^ ]+',line)
                        if len(sline) > 5:
                            suffix.append(''.join([whites[j]+sline[j] for j in range(5,len(sline))]))
                        else:
                            suffix.append('')
                    thisatom = int(sline[0])
                    thisres.add(thisatom)
                    forwardres.add(thisatom)
                    if len(sline) >= 6:
                        forwardres.update([int(j) for j in sline[6:]])
                    if thisres == forwardres:
                        thisres = set([])
                        forwardres = set([])
                        thisresid += 1
                    xyz.append([float(sline[2]),float(sline[3]),float(sline[4])])
                    if thisatom == na:
                        nframes += 1
                        title = True
                        xyzs.append(array(xyz))
                        xyz = []
                        rawarcs.append(list(rawarc))
                        rawarc = []
                else:
                    print "What is this line?  It isn't data"
                    print line,
                    exit()
        Answer = {'xyzs'   : xyzs,
                  'resid'  : resid,
                  'elem'   : elem,
                  'comms'  : comms,
                  'rawarcs': rawarcs,
                  'suffix' : suffix}
        return Answer

    def read_ndx(self, fnm):
        """ Read an index.ndx file and add an entry to the dictionary
        {'index_name': [ num1, num2, num3 .. ]}
        """

        section = None
        for line in open(fnm):
            line = line.strip()
            s = line.split()
            if match('^\[.*\]',line):
                section = sub('[\[\] \n]','',line)
            elif all([isint(i) for i in s]):
                for j in s:
                    self.index.setdefault(section,[]).append(int(j)-1)

    def reorder(self, idx_name=None):
        """ Reorders an xyz file using data provided from an .ndx file. """
        
        if idx_name not in self.index:
            print "ARGHH, %s doesn't exist in the index" % idx_name
            exit(1)

        self.require_resid()
        self.require_resname()

        newatomname = list(self.atomname)
        newelem = list(self.elem)
        newresid = list(self.resid)
        newresname = list(self.resname)
        for an, assign in enumerate(self.index[idx_name]):
            newatomname[an] = self.atomname[assign]
            newelem[an] = self.elem[assign]
            newresid[an] = self.resid[assign]
            newresname[an] = self.resname[assign]
        self.atomname = list(newatomname)
        self.elem = list(newelem)
        self.resid = list(newresid)
        self.resname = list(newresname)

        for sn, xyz in enumerate(self.xyzs):
            newxyz = xyz.copy()
            for an, assign in enumerate(self.index[idx_name]):
                newxyz[an] = xyz[assign]
            self.xyzs[sn] = newxyz.copy()
            
            
    def read_gro(self, fnm):
        """ Read a GROMACS .gro file.

        """
        xyzs     = []
        elem     = [] # The element, most useful for quantum chemistry calculations
        atomname = [] # The atom name, for instance 'HW1'
        comms    = []
        resid    = []
        resname  = []
        boxes    = []
        xyz      = []
        ln       = 0
        frame    = 0
        for line in open(fnm):
            sline = line.split()
            if ln == 0:
                comms.append(line.strip())
            elif ln == 1:
                na = int(line.strip())
            elif is_gro_coord(line):
                if frame == 0: # Create the list of residues, atom names etc. only if it's the first frame.
                    # Name of the residue, for instance '153SOL1 -> SOL1' ; strips leading numbers
                    thisresname = sub('^[0-9]*','',sline[0])
                    resname.append(thisresname)
                    resid.append(int(sline[0].replace(thisresname,'')))
                    atomname.append(sline[1])
                    thiselem = sline[1]
                    if len(thiselem) > 1:
                        thiselem = thiselem[0] + sub('[A-Z0-9]','',thiselem[1:])
                    elem.append(thiselem)
                xyz.append([float(i) for i in sline[-3:]])
            elif is_gro_box(line) and ln == na + 2:
                boxes.append([float(i)*10 for i in sline])
                xyzs.append(array(xyz)*10)
                xyz = []
                ln = -1
                frame += 1
            else:
                print "AHOOOOOGA, I encountered a line I didn't expect!"
                print line
            ln += 1
        Answer = {'xyzs'     : xyzs,
                  'elem'     : elem,
                  'atomname' : atomname,
                  'resid'    : resid,
                  'resname'  : resname,
                  'boxes'    : boxes,
                  'comms'    : comms}
        return Answer

    def read_charmm(self, fnm):
        """ Read a CHARMM .cor (or .crd) file.

        """
        xyzs     = []
        elem     = [] # The element, most useful for quantum chemistry calculations
        atomname = [] # The atom name, for instance 'HW1'
        comms    = []
        resid    = []
        resname  = []
        xyz      = []
        thiscomm = []
        ln       = 0
        frame    = 0
        an       = 0
        for line in open(fnm):
            sline = line.split()
            if match('^\*',line):
                if len(sline) == 1:
                    comms.append(';'.join(list(thiscomm)))
                    thiscomm = []
                else:
                    thiscomm.append(' '.join(sline[1:]))
            elif match('^ *[0-9]+ +(EXT)?$',line):
                na = int(sline[0])
            elif is_charmm_coord(line):
                if frame == 0: # Create the list of residues, atom names etc. only if it's the first frame.
                    resid.append(sline[1])
                    resname.append(sline[2])
                    atomname.append(sline[3])
                    thiselem = sline[3]
                    if len(thiselem) > 1:
                        thiselem = thiselem[0] + sub('[A-Z0-9]','',thiselem[1:])
                    elem.append(thiselem)
                xyz.append([float(i) for i in sline[4:7]])
                an += 1
                if an == na:
                    xyzs.append(array(xyz))
                    xyz = []
                    an = 0
                    frame += 1
            ln += 1
        Answer = {'xyzs'     : xyzs,
                  'elem'     : elem,
                  'atomname' : atomname,
                  'resid'    : resid,
                  'resname'  : resname,
                  'comms'    : comms}
        return Answer

    def write_qcin(self, subset = None):
        """Write a Q-Chem input file from the template.

        """
        
        if self.tempfnm == None:
            self.tempfnm = "qtemp.in"

        got_mol   = False
        got_comm  = False
        active    = True
        bas_sect  = False

        out = []
        temp_lines = []
        ln = 0
        xyz_insert = 0
        comm_insert = 0
        for line in open(self.tempfnm).readlines():
            line = line.strip()
            sline = line.split()
            if match('^\$',line):
                wrd = sub('\$','',line)
                if wrd == 'end':
                    active = True
                    bas_sect = False
            elif wrd == 'molecule':
                if got_mol: pass
                got_mol = True
                if match("[+-]?[0-9]+ +[0-9]+$",line.split('!')[0].strip()):
                    if xyz_insert == 0:
                        xyz_insert = ln
                    if active and self.charge == None and self.mult == None:
                        self.charge = int(sline[0])
                        self.mult = int(sline[1])
                    active = False
            elif wrd == 'comment':
                if got_comm: pass
                got_comm = True
                if comm_insert == 0:
                    comm_insert = ln
            elif wrd == 'basis':
                bas_sect = True
            if active:
                if bas_sect:
                    line = line.split('!')[0]
                temp_lines.append(line)
                ln += 1
                
            if match('^@$', line):
                have_ats = 1
            elif len(sline) != 0:
                have_ats = 0

        for I, xyz in enumerate(self.xyzs):
            if subset != None and I not in subset: continue
            for ln, line in enumerate(temp_lines):
                if ln == xyz_insert:
                    out.append("%i %i" % (self.charge, self.mult))
                    for i in range(self.na()):
                        out.append(format_xyz_coord(self.elem[i],xyz[i]))
                elif ln == comm_insert:
                    out.append(self.comms[I])
                out.append(line)
            if I < len(self.xyzs) - 1 and have_ats == 0:
                out.append('')
                out.append('@@@@')
                out.append('')
            
        return out

    def write_xyz(self, subset = None):
        out = []
        for I, xyz in enumerate(self.xyzs):
            if subset != None and I not in subset: continue
            out.append("%-5i" % self.na())
            out.append(self.comms[I])
            for i in range(self.na()):
                out.append(format_xyz_coord(self.elem[i],xyz[i]))
        return out

    def write_arc(self, subset = None):
        out = []
        if self.rawarcs == None:
            if self.tempfnm != None:
                suffix = self.read_arc(self.tempfnm)['suffix']
                print "Taking topology from template .arc file"
            else:
                suffix = ['' for i in range(self.na())]
                print "Beware, this .arc file contains no atom type or topology info (to do that, you need an .arc file as the third argument"
            for I, xyz in enumerate(self.xyzs):
                if subset != None and I not in subset: continue
                out.append("%6i  %s" % (self.na(), self.comms[I]))
                for i in range(self.na()):
                    out.append("%6i  %s%s" % (i+1,format_xyz_coord(self.elem[i],xyz[i],tinker=True),suffix[i]))
        else:
            for I, rawarc in enumerate(self.rawarcs):
                if subset != None and I not in subset: continue
                out += rawarc
        return out

    def write_gro(self, subset = None):
        out = []
        self.require_resname()
        self.require_resid()
        self.require_boxes()
        for I, xyz in enumerate(self.xyzs):
            if subset != None and I not in subset: continue
            xyzwrite = xyz.copy()
            xyzwrite /= 10.0 # GROMACS uses nanometers
            out.append("Generated by filecnv.py : " + self.comms[I])
            out.append("%5i" % self.na())
            for an, line in enumerate(xyzwrite):
                out.append(format_gro_coord(self.resid[an],self.resname[an],self.atomname[an],an+1,xyzwrite[an]))
            out.append(format_gro_box(xyz = self.boxes[I]))
        return out

    def require_resid(self):
        if self.resid == None:
            na_res = int(raw_input("Enter how many atoms are in a residue -> "))
            self.resid = [1 + i/na_res for i in range(self.na())]
            
    def require_resname(self):
        if self.resname == None:
            resname = raw_input("Enter a residue name (3-letter like 'SOL') -> ")
            self.resname = [resname for i in range(self.na())]
            
    def require_boxes(self):
        if self.boxes == None:
            boxstr = raw_input("Enter 1 / 3 / 9 numbers for a cubic / orthorhombic / triclinic box, use Angstrom -> ")
            box    = [float(i) for i in boxstr.split()]
            if len(box) == 3 or len(box) == 9:
                self.boxes = [box for i in range(self.ns)]
            elif len(box) == 1:
                self.boxes = [[box[0],box[0],box[0]] for i in range(self.ns)]
            else:
                print "Not sure what to do since you gave me %i numbers" % len(box)
                exit(1)
                
