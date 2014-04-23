import os
import re
import csv
import copy
import errno
import shutil
import numpy as np
import pandas as pd
import itertools
import cStringIO

from forcebalance.molecule import Molecule
from forcebalance.observable import OMap
from forcebalance.target import Target
from forcebalance.finite_difference import in_fd
from forcebalance.nifty import flat, col, row, isint, isnpnan
from forcebalance.nifty import lp_dump, lp_load, wopen, _exec
from forcebalance.nifty import GoInto, LinkFile, link_dir_contents
from forcebalance.nifty import printcool, printcool_dictionary
from forcebalance.nifty import getWorkQueue

from collections import defaultdict, OrderedDict

import forcebalance
from forcebalance.output import *
logger = getLogger(__name__)
# print logger.parent.parent.handlers[0]
# logger.parent.parent.handlers = []

def getval(dframe, col):
    """ Extract the single non-NaN value from a column. """
    nnan = [i for i in dframe[col] if not isnpnan(i)]
    if len(nnan) != 1:
        logger.error('%i values in column %s are not NaN (expected only 1)' % (len(nnan), col))
        raise RuntimeError
    return nnan[0]

class TextParser(object):
    """ Parse a text file. """
    def __init__(self, fnm):
        self.fnm = fnm
        self.parse()

    def is_empty_line(self):
        return all([len(fld.strip()) == 0 for fld in self.fields])

    def is_comment_line(self):
        return re.match('^[\'"]?#',self.fields[0].strip())

    def process_header(self):
        """ Function for setting more attributes using the header line, if needed. """
        self.heading = [i.strip() for i in self.fields[:]]

    def process_data(self):
        """ Function for setting more attributes using the current line, if needed. """
        trow = []
        for ifld in range(len(self.heading)):
            if ifld < len(self.fields):
                trow.append(self.fields[ifld].strip())
            else:
                trow.append('')
        return trow

    def sanity_check(self):
        """ Extra sanity checks. """

    def parse(self):
        self.heading = []                 # Fields in header line
        meta = defaultdict(list)          # Dictionary of metadata
        found_header = 0                  # Whether we found the header line
        table = []                        # List of data records
        self.generate_splits()            # Generate a list of records for each line.
        self.ln = 0                       # Current line number
        for line, fields in zip(open(self.fnm).readlines(), self.splits):
            # Set attribute so methods can use it.
            self.fields = fields
            # Skip over empty lines or comment lines.
            if self.is_empty_line():
                logger.debug("\x1b[96mempt\x1b[0m %s\n" % line.replace('\n',''))
                self.ln += 1
                continue
            if self.is_comment_line():
                logger.debug("\x1b[96mcomm\x1b[0m %s\n" % line.replace('\n',''))
                self.ln += 1
                continue
            # Indicates metadata mode.
            is_meta = 0
            # Indicates whether this is the header line.
            is_header = 0
            # Split line by tabs.
            for ifld, fld in enumerate(fields):
                fld = fld.strip()
                # Stop parsing when we encounter a comment line.
                if re.match('^[\'"]?#',fld): break
                # The first word would contain the name of the metadata key.
                if ifld == 0:
                    mkey = fld
                # Check if the first field is an equals sign (turn on metadata mode).
                if ifld == 1:
                    # Activate metadata mode.
                    if fld == "=":
                        is_meta = 1
                    # Otherwise, this is the header line.
                    elif not found_header:
                        is_header = 1
                        found_header = 1
                # Read in metadata.
                if ifld > 1 and is_meta:
                    meta[mkey].append(fld)
            # Set field start, field end, and field content for the header.
            if is_header:
                logger.debug("\x1b[1;96mhead\x1b[0m %s\n" % line.replace('\n',''))
                self.process_header()
            elif is_meta:
                logger.debug("\x1b[96mmeta\x1b[0m %s\n" % line.replace('\n',''))
            else:
                # Build the row of data to be appended to the table.
                # Loop through the fields in the header and inserts fields
                # in the data line accordingly.  Ignores trailing tabs/spaces.
                logger.debug("\x1b[96mdata\x1b[0m %s\n" % line.replace('\n',''))
                table.append(self.process_data())
            self.ln += 1
        self.sanity_check()
        if logger.level == DEBUG:
            printcool("%s parsed as %s" % (self.fnm.replace(os.getcwd()+'/',''), self.format), color=6)
        self.metadata = meta
        self.table = table
        
class CSV_Parser(TextParser):
    
    """ 
    Parse a comma-separated file.  This class is for all
    source files that are .csv format (characterized by having the
    same number of comma-separated fields in each line).  Fields are
    separated by commas but they may contain commas as well.

    In contrast to the other formats, .csv MUST contain the same
    number of commas in each line.  .csv format is easily prepared
    using Excel.
    """
    
    def __init__(self, fnm):
        self.format = "comma-separated values (csv)"
        super(CSV_Parser, self).__init__(fnm)

    def generate_splits(self):
        with open(self.fnm, 'r') as f: self.splits = list(csv.reader(f))

class TAB_Parser(TextParser):
    
    """ 
    Parse a tab-delimited file.  This function is called for all
    source files that aren't csv and contain at least one tab.  
    Fields are separated by tabs and do not contain tabs.

    Tab-delimited format is easy to prepare using programs like Excel.
    It is easier to read than .csv but represented differently by
    different editors.  
    
    Empty fields must still exist (represented using multiple tabs).
    """
    
    def __init__(self, fnm):
        self.format = "tab-delimited text"
        super(TAB_Parser, self).__init__(fnm)

    def generate_splits(self):
        self.splits = [line.split('\t') for line in open(self.fnm).readlines()]

class FIX_Parser(TextParser):
    
    """ 
    Parse a fixed width format file.  This function is called for all
    source files that aren't csv and contain no tabs.

    Fixed width is harder to prepare by hand but easiest to read,
    because it looks the same in all text editors.  The field width is
    determined by the header line (first line in the data table),
    i.e. the first non-empty, non-comment, non-metadata line.

    Empty fields need to be filled with the correct number of spaces.
    All fields must have the same alignment (left or right).  The
    start and end of each field is determined from the header line and
    used to determine alignment. If the alignment cannot be determined
    then it will throw an error.

    Example of a left-aligned fixed width file:

    T           P (atm)     Al          Al_wt       Scd1_idx    Scd1        Scd2_idx    Scd2    
    323.15      1           0.631       1           C15                     C34                 
                                                    C17         0.198144    C36         0.198144
                                                    C18         0.198128    C37         0.198128
                                                    C19         0.198111    C38         0.198111
                                                    C20         0.198095    C39         0.198095
                                                    C21         0.198079    C40         0.198079
                                                    C22         0.197799    C41         0.197537
                                                    C23         0.198045    C42         0.198046
                                                    C24         0.178844    C43         0.178844
                                                    C25         0.167527    C44         0.178565
                                                    C26         0.148851    C45         0.16751
                                                    C27         0.134117    C46         0.148834
                                                    C28         0.119646    C47         0.1341
                                                    C29         0.100969    C48         0.110956
                                                    C30         0.07546     C49         0.087549
                                                    C31                     C50

    """

    def __init__(self, fnm):
        self.format = "fixed-width text"
        self.fbegs_dat = []
        self.fends_dat = []
        super(FIX_Parser, self).__init__(fnm)

    def generate_splits(self):
        # This regular expression splits a string looking like this:
        # "Density (kg m^-3) Hvap (kJ mol^-1) Alpha Kappa".  But I
        # don't want to split in these places: "Density_(kg_m^-3)
        # Hvap_(kJ_mol^-1) Alpha Kappa"
        allfields = [list(re.finditer('[^\s(]+(?:\s*\([^)]*\))?', line)) for line in open(self.fnm).readlines()]
        self.splits = []
        # Field start / end positions for each line in the file
        self.fbegs = []
        self.fends = []
        for line, fields in zip(open(self.fnm).readlines(), allfields):
            self.splits.append([fld.group(0) for fld in fields])
            self.fbegs.append([fld.start() for fld in fields])
            self.fends.append([fld.end() for fld in fields])
        
    def process_header(self):
        super(FIX_Parser, self).process_header()
        # Field start / end positions for the header line
        self.hbeg = self.fbegs[self.ln]
        self.hend = self.fends[self.ln]

    def process_data(self):
        trow = []
        hbeg = self.hbeg
        hend = self.hend
        fbeg = self.fbegs[self.ln]
        fend = self.fends[self.ln]
        fields = self.fields
        # Check alignment and throw an error if incorrectly formatted.
        if not ((set(fbeg).issubset(hbeg)) or (set(fend).issubset(hend))):
            logger.error("This \x1b[91mdata line\x1b[0m is not aligned with the \x1b[92mheader line\x1b[0m!\n")
            logger.error("\x1b[92m%s\x1b[0m\n" % header.replace('\n',''))
            logger.error("\x1b[91m%s\x1b[0m\n" % line.replace('\n',''))
            raise RuntimeError
        # Left-aligned case
        if set(fbeg).issubset(hbeg):
            for hpos in hbeg:
                if hpos in fbeg:
                    trow.append(fields[fbeg.index(hpos)])
                else:
                    trow.append('')
        # Right-aligned case
        if set(fend).issubset(hend):
            for hpos in hend:
                if hpos in fend:
                    trow.append(fields[fend.index(hpos)].strip())
                else:
                    trow.append('')
        # Field start / end positions for the line of data
        self.fbegs_dat.append(fbeg[:])
        self.fends_dat.append(fend[:])
        return trow

    def sanity_check(self):
        if set(self.hbeg).issuperset(set(itertools.chain(*self.fbegs_dat))):
            self.format = "left-aligned fixed width text"
        elif set(self.hend).issuperset(set(itertools.chain(*self.fends_dat))):
            self.format = "right-aligned fixed width text"
        else:
            # Sanity check - it should never get here unless the parser is incorrect.
            logger.error("Fixed-width format detected but columns are neither left-aligned nor right-aligned!\n")
            raise RuntimeError
    
def parse1(fnm):

    """Determine the format of the source file and call the
    appropriate parsing function."""

    # CSV files have the same number of comma separated fields in every line, they are the simplest to parse.
    with open(fnm, 'r') as f: csvf = list(csv.reader(f))
    if len(csvf[0]) > 1 and len(set([len(i) for i in csvf])) == 1:
        return CSV_Parser(fnm)

    # Strip away comments and empty lines.
    nclines = [re.sub('[ \t]*#.*$','',line) for line in open(fnm).readlines() 
               if not (line.strip().startswith("#") or not line.strip())]

    # Print the sanitized lines to a new file object.
    # Note the file object needs ot be rewound every time we read or write to it.
    fdat = cStringIO.StringIO()
    for line in nclines:
        print >> fdat, line,
    fdat.seek(0)
    
    # Now the file can either be tab-delimited or fixed width.
    # If ANY tabs are found in the sanitized lines, then it is taken to be
    # a tab-delimited file.
    have_tabs = any(['\t' in line for line in fdat.readlines()]) ; fdat.seek(0)
    if have_tabs:
        return TAB_Parser(fnm)
    else:
        return FIX_Parser(fnm)
    return

def fix_suffix(obs, head, suffixs, standard_suffix):

    """ Standardize the suffix in a column heading. """

    if head in suffixs:
        if obs == '': 
            logger.error('\x1b[91mEncountered heading %s but there is no observable to the left\x1b[0m\n' % head)
            raise RuntimeError
        return obs + '_' + standard_suffix, False
    elif len(head.split('_')) > 1 and head.split('_')[-1] in suffixs:
        newhl = head.split('_')
        newhl[-1] = standard_suffix
        return '_'.join(newhl), False
    else:
        return head, True

def stand_head(head, obs):

    """ 
    Standardize a column heading.  Does the following:

    1) Make lowercase
    2) Split off the physical unit
    3) If a weight, uncertainty or atom index, prepend the observable name
    4) Shorten temperature and pressure
    5) Determine if this is a new observable
    
    Parameters:
    head = Name of the heading
    obs = Name of the observable (e.g. from a previously read field)
    """

    head = head.lower()
    usplit = re.split(' *\(', head, maxsplit=1)
    punit = ''
    if len(usplit) > 1:
        hfirst = usplit[0]
        punit = re.sub('\)$','',usplit[1].strip())
        logger.debug("header %s split into %s, %s" % (head, hfirst, punit))
    else:
        hfirst = head
    newh = hfirst
    newh, o1 = fix_suffix(obs, newh, ['w', 'wt', 'wts', 'weight', 'weights'], 'wt')
    newh, o2 = fix_suffix(obs, newh, ['s', 'sig', 'sigma', 'sigmas'], 'sig')
    newh, o3 = fix_suffix(obs, newh, ['i', 'idx', 'index', 'indices'], 'idx')
    if newh in ['t', 'temp', 'temperature']: newh = 'temp'
    if newh in ['p', 'pres', 'pressure']: newh = 'pres'
    if all([o1, o2, o3]):
        obs = newh
    if newh != hfirst:
        logger.debug("header %s renamed to %s\n" % (hfirst, newh))
    return newh, punit, obs

def find_file(tgtdir, index, stype, sufs, iscrd, icn=0):
    """ 
    Search for a suitable file that matches the simulation index,
    type, suffix and IC number.  This can be used to search for
    initial coordinates, but also auxiliary files for the
    simulation (e.g. .top and .mdp files for a Gromacs simulation,
    or .key files for a Tinker simulation.)

    Generally, it is preferred to provide files where the base
    name matches the simulation type.  However, since it is also
    okay to put all files for a simulation type into a
    subdirectory, generic file names like 'topol' and 'conf' may
    be used.

    Initial condition files will be searched for in the following priority (suf stands for suffix)
    targets/target_name/index/stype/ICs/stype_#.suf
    targets/target_name/index/stype/ICs/stype#.suf
    targets/target_name/index/stype/ICs/#.suf
    targets/target_name/index/stype/ICs/stype.suf
    targets/target_name/index/stype/ICs/coords.suf
    targets/target_name/index/stype/ICs/conf.suf
    targets/target_name/index/stype/ICs/topol.suf
    targets/target_name/index/stype/ICs/grompp.suf
    targets/target_name/index/stype/ICs/input.suf
    targets/target_name/index/stype/ICs/tinker.suf
    targets/target_name/index/stype/stype.suf
    targets/target_name/index/stype/coords.suf
    targets/target_name/index/stype.suf
    targets/target_name/stype.suf

    @param[in] tgtdir Name of the target directory to look in
    @param[in] index Name of the index directory to look in (within tgtdir)
    @param[in] stype Name of the simulation type to look for
    @param[in] sufs List of suffixes to look for in order of priority
    @param[in] iscrd Whether the file is a coordinate file (false for auxiliary files like .mdp).
    @param[in] icn Initial coordinate number (will look for sequentially numbered file, or single file with multiple structures)
    """
    found = ''
    # The 2-tuple here corresponds to:
    # - Search path for the file
    # - Whether the file that we're looking for is 'numbered'
    #   (i.e. a different file for each structure); otherwise the
    #   single file may contain multiple structures
    pfxs = [stype, 'coords', 'conf', 'topol', 'grompp', 'input', 'tinker', '']
    
    basefnms = list(itertools.chain(*[[(os.path.join(index, stype, 'ICs', pfx+'_'+("%i" % icn)), True),
                                       (os.path.join(index, stype, 'ICs', pfx+("%i" % icn)), True),
                                       (os.path.join(index, stype, 'ICs', pfx), False),
                                       (os.path.join(index, stype, pfx), False),
                                       (os.path.join(index, pfx), False),
                                       (os.path.join(pfx), False)] for pfx in pfxs]))
    
    paths = OrderedDict()
    for fnm, numbered in basefnms:
        for suf in sufs:
            fpath = os.path.join(tgtdir, fnm+suf if suf.startswith('.') else fnm+'.'+suf)
            paths[fpath] = os.path.exists(fpath)
            if os.path.exists(fpath):
                if found != '':
                    logger.info('Target %s Index %s Simulation %s : '
                                '%s overrides %s\n' % (os.path.basename(tgtdir), index, stype, fpath))
                else:
                    if iscrd and not numbered:
                        M = Molecule(fpath)
                        if len(M) <= icn:
                            logger.error("Target %s Index %s Simulation %s : "
                                         "file %s doesn't have enough structures\n" % 
                                         (os.path.basename(tgtdir), index, stype, fpath))
                            raise RuntimeError
                    logger.info('Target %s Index %s Simulation %s : '
                                'found file %s\n' % (os.path.basename(tgtdir), index, stype, fpath))
                    found = os.path.abspath(fpath)
    if found == '':
        logger.error("Can't find a file for index %s, simulation %s, suffix %s in the search path" % (index, stype, '/'.join(sufs)))
        raise RuntimeError
    return found, 0 if numbered else icn

class Thermo(Target):
    """
    A target for fitting general experimental data sets. The source
    data is described in a text file formatted according to the
    Specification.

    """
    def __init__(self, options, tgt_opts, forcefield):
        ## Initialize base class
        super(Thermo, self).__init__(options, tgt_opts, forcefield)

        ## Parameters
        # Source data (experimental data, model parameters and weights)
        self.set_option(tgt_opts, "source", forceprint=True)
        # Observables to calculate
        self.set_option(tgt_opts, "observables", "user_observable_names", forceprint=True)
        # Length of simulation chain
        self.set_option(tgt_opts, "simulations", "user_simulation_names", forceprint=True)
        # Number of time steps in the equilibration run
        self.set_option(tgt_opts, "eq_steps", forceprint=True)
        # Number of time steps in the production run
        self.set_option(tgt_opts, "md_steps", forceprint=True)
        # Time step (in femtoseconds)
        self.set_option(tgt_opts, "timestep", forceprint=True)
        # Sampling interval (in picoseconds)
        self.set_option(tgt_opts, "interval", forceprint=True)
        # Save trajectories?
        self.set_option(tgt_opts, "save_traj", forceprint=True)

        ## Variables
        # Prefix names for simulation data
        self.simpfx    = "sim"
        # Data points for observables
        self.points    = []
        # Denominators for observables
        self.denoms    = {}
        # Weights for observables
        self.weights   = {}
        # The list of simulations that we'll be running.
        self.SimNames = [i.lower() for i in self.user_simulation_names]
        # Store the dictionary of allowed suffixes
        self.OptionDict['crdsfx'] = self.crdsfx

        ## Read source data and initialize points; creates self.Data, self.Indices and self.Columns objects.
        self.read_source(os.path.join(self.root, self.tgtdir, self.source))

        ## Set up self.Observables.
        self.initialize_observables()

        ## Set up self.Simulations.
        self.initialize_simulations()
        
        ## Copy run scripts from ForceBalance installation directory
        for f in self.scripts:
            LinkFile(os.path.join(os.path.split(__file__)[0], "data", f),
                     os.path.join(self.root, self.tempdir, f))

    def read_source(self, srcfnm):
        """Read and store source data.

        Parameters
        ----------
        srcfnm : string
            Read source data from this filename.

        Returns
        -------
        Nothing

        """
            
        logger.info('Parsing source file %s\n' % srcfnm)
        source = parse1(srcfnm)
        printcool_dictionary(source.metadata, title="Metadata")
        revhead = []
        col = ''
        colnames = []

        units = defaultdict(str)

        for i, head in enumerate(source.heading):
            if i == 0 and head.lower() == 'index': # Treat special case because index can also mean other things
                revhead.append('index')
                continue
            newh, punit, col = stand_head(head, col)
            if col not in colnames + ['temp', 'pres', 'n_ic']: colnames.append(col)
            revhead.append(newh)
            if punit != '':
                units[newh] = punit
        source.heading = revhead
 
        if len(set(revhead)) != len(revhead):
            logger.error('Column headings : ' + str(revhead) + '\n')
            logger.error('\x1b[91mColumn headings are not unique!\x1b[0m\n')
            raise RuntimeError

        if revhead[0] != 'index':
            logger.error('\x1b[91mIndex column heading is not present\x1b[0m\n(Add an Index column on the left!)\n')
            raise RuntimeError
            
        uqidx = []
        saveidx = ''
        index = []
        snum = 0
        drows = []
        # thisidx = Index that is built from the current row (may be empty)
        # saveidx = Index that may have been saved from a previous row
        # snum = Subindex number
        # List of (index, heading) tuples which contain file references.
        fref = OrderedDict()
        for rn, row in enumerate(source.table):
            this_insert = []
            thisidx = row[0]
            if thisidx != '': 
                saveidx = thisidx
                snum = 0
                if saveidx in uqidx: 
                    logger.error('Index %s is duplicated in data table\n' % i)
                    raise RuntimeError
                uqidx.append(saveidx)
            index.append((saveidx, snum))
            if saveidx == '':
                logger.error('Row of data : ' + str(row) + '\n')
                logger.error('\x1b[91mThis row does not have an index!\x1b[0m\n')
                raise RuntimeError
            snum += 1
            if any([':' in fld for fld in row[1:]]):
                # Here we read rows from another data table.  
                # Other files may be referenced in the cell of a primary
                # table using filename:column_number (numbered from 1).
                # Rules: (1) No matter where the filename appears in the column,
                # the column is inserted at the beginning of the system index.
                # (2) There can only be one file per system index / column.
                # (3) The column heading in the secondary file that's being
                # referenced must match that of the reference in the primary file.
                col2 = ''
                for cid_, fld in enumerate(row[1:]):
                    if ':' not in fld: continue
                    cid = cid_ + 1
                    def reffld_error(reason=''):
                        logger.error('Row: : ' + ' '.join(row) + '\n')
                        logger.error('Entry : ' + fld + '\n')
                        logger.error('This filename:column reference is not valid!%s' % 
                                     (' (%s)' % reason if reason != '' else ''))
                        raise RuntimeError
                    if len(fld.split(':')) != 2:
                        reffld_error('Wrong number of colon-separated fields')
                    if not isint(fld.split(':')[1]):
                        reffld_error('Must be an integer after the colon')
                    fnm = fld.split(':')[0]
                    fcol_ = int(fld.split(':')[1])
                    fpath = os.path.join(os.path.split(srcfnm)[0], fnm)
                    if not os.path.exists(fpath):
                        reffld_error('%s does not exist' % fpath)
                    if (saveidx, revhead[cid]) in fref:
                        reffld_error('%s already contains a file reference' % (saveidx, revhead[cid]))
                    subfile = parse1(fpath)
                    fcol = fcol_ - 1
                    head2, punit2, col2 = stand_head(subfile.heading[fcol], col2)
                    if revhead[cid] != head2:
                        reffld_error("Column heading of %s (%s) doesn't match original (%s)" % (fnm, head2, revhead[cid]))
                    fref[(saveidx, revhead[cid])] = [row2[fcol] for row2 in subfile.table]

        # Insert the file-referenced data tables appropriately into
        # our main data table.
        for (saveidx, head), newcol in fref.items():
            inum = 0
            for irow in range(len(source.table)):
                if index[irow][0] != saveidx: continue
                lrow = irow
                cidx = revhead.index(head)
                source.table[irow][cidx] = newcol[inum]
                inum += 1
                if inum >= len(newcol): break
            for inum1 in range(inum, len(newcol)):
                lrow += 1
                nrow = ['' for i in range(len(revhead))]
                nrow[cidx] = newcol[inum1]
                source.table.insert(lrow, nrow)
                index.insert(lrow, (saveidx, inum1))
                
        for rn, row in enumerate(source.table):
            drows.append([i if i != '' else np.nan for i in row[1:]])

        # Turn it into a pandas DataFrame.
        self.Data = pd.DataFrame(drows, columns=revhead[1:], index=pd.MultiIndex.from_tuples(index, names=['index', 'subindex']))

        def intcol(col):
            if col in self.Data.columns:
                for idx in self.Data.index:
                    if not isnpnan(self.Data[col][idx]):
                        self.Data[col][idx] = int(self.Data[col][idx])

        def floatcol(col):
            if col in self.Data.columns:
                self.Data[col] = self.Data[col].astype(float)

        intcol('n_ic')

        # A list of indices (i.e. top-level indices) which correspond
        # to sets of simulations that we'll be running.
        self.Indices = []
        for idx in self.Data.index:
            if idx[0] not in self.Indices:
                self.Indices.append(idx[0])

        # List of columns in the main data table.
        self.Columns = self.Data.columns

        # Certain things (e.g. run parameters like temp, pres) are keyed to the index only.
        chead = []
        crows = []
        for index in self.Indices:
            crow = []
            for head in ['temp', 'pres']:
                if head not in self.Data: continue
                if head not in chead: chead.append(head)
                rlist = list(set([i for i in self.Data.ix[index][head][:] if not isnpnan(i)]))
                if len(rlist) != 1:
                    logger.error('Heading %s should appear once for index %s (found %i)' % (head, index, len(rlist)))
                    raise RuntimeError
                crow.append(rlist[0])
            crows.append(crow[:])

        # Now create the mini data table.
        self.Data2 = pd.DataFrame(crows, columns=chead, index=self.Indices)

        return

    def initialize_observables(self):
        """ 
        Determine Observable objects to be created.  Checks to see
        whether simulations are consistent with observables (i.e. no
        missing simulations or ambiguities.)

        In order to implement a new observable, create a class in
        observable.py and add it to OMap.
        """
        self.Observables = OrderedDict()
        for oname in [stand_head(i, '')[2] for i in self.user_observable_names]:
            if oname in self.Observables:
                logger.error('%s was already specified as an observable' % (oname))
                raise RuntimeError
            self.Observables[oname] = OrderedDict()
            for index in self.Indices:
                if oname in OMap:
                    Objs = []
                    Reqs = []
                    for OClass in OMap[oname]:
                        OObj = OClass(self.Data)
                        Reqs.append(OObj.requires.keys())
                        if all([i in self.SimNames for i in OObj.requires.keys()]):
                            Objs.append(OObj)
                    if len(Objs) == 0:
                        logger.error('Observable %s is specified but required simulations are missing; choose %s' % (oname, ' or '.join([str(r) for r in Reqs])))
                        raise RuntimeError
                    if len(Objs) > 1:
                        logger.error("Observable %s not uniquely mapped to simulations (choose between %s)" % (oname, ' or '.join([o.name in Objs])))
                        raise RuntimeError
                    logger.info("Creating %s observable object for index %s\n" % (Objs[0].name, index))
                    self.Observables[oname][index] = Objs[0]
                else:
                    logger.error('%s is specified but there is no corresponding Observable class\n' % oname)
                    raise RuntimeError
        return

    def initialize_simulations(self):

        """ 

        Prepare simulations to be launched.  Set initial conditions
        and create directories.  This function is intended to be run
        at the start of each optimization cycle, so that initial
        conditions may be easily set.

        """
        # print narrow()
            
        self.Simulations = OrderedDict([(i, []) for i in self.Indices])
        # Dictionary of time series to extract from each simulation.
        SimTS = defaultdict(set)
        # Check to see whether each observable can be unambiguously calculated from the specified simulations
        for obsname in self.Observables.keys():
            sreq = self.Observables[obsname][self.Indices[0]].requires
            for i, j in sreq.items():
                SimTS[i].update(set(j))
            printcool_dictionary(sreq, title="Observable %s uses these simulations : timeseries" % obsname)
        printcool_dictionary({i:' '.join(sorted(list(SimTS[i]))) for i in sorted(SimTS.keys())}, 
                             title="Needed Simulations : Extracted Timeseries")
        unused = sorted(list(set(self.SimNames).difference(set(SimTS.keys()))))
        if len(unused) > 0:
            logger.error("Simulation %s is specified but it's never used to calculate any observables" % ', '.join(unused))
            raise RuntimeError

        for index in self.Indices:
            for stype, tsset in SimTS.items():
                if 'n_ic' in self.Data2.ix[index]:
                    n_ic = self.Data2.ix[index]['n_ic']
                    print n_ic
                    if n_ic < 1:
                        logger.error("n_ic must >= 1")
                        raise RuntimeError
                else:
                    n_ic = 1
                for icn in range(n_ic):
                    sname = "%s_%i" % (stype, icn) if n_ic > 1 else stype
                    self.Simulations[index].append(Simulation(self, self.Data.ix[index], sname, index, stype, icn, sorted(list(tsset))))
        return

    def submit_jobs(self, mvals, AGrad=True, AHess=True):
        """This routine is called by Objective.stage() and will run before "get".
        It submits the jobs (or runs them locally) and the stage() function will wait for jobs
        to complete.

        Parameters
        ----------
        mvals : list
            Mathematical parameter values.
        AGrad : Boolean
            Switch to turn on analytic gradient.
        AHess : Boolean
            Switch to turn on analytic Hessian.

        Returns
        -------
        Nothing.
        
        """

        printcool("Submitting jobs")
        cwd = os.getcwd()
        wq = getWorkQueue()
        for index in self.Indices:
            # if 'temp' in self.Data:
            #     tset = set([iself.Data['temp'].ix[index][:])
            temp = self.Data2['temp'].ix[index] if 'temp' in self.Data2 else None
            pres = self.Data2['pres'].ix[index] if 'pres' in self.Data2 else None
            for Sim in self.Simulations[index]:
                Sim.gradient = AGrad
                simd = os.path.join(os.getcwd(), index, Sim.name)
                GoInto(simd)
                # Submit or run the simulation if the result file does not exist.
                if not (os.path.exists('result.p') or os.path.exists('result.p.bz2')):
                    # Write coordinate file in the current location.
                    M = Molecule(os.path.join(self.root, Sim.initial))[Sim.iframe]
                    M.write(Sim.EngOpts['coords'])
                    # Copy auxiliary files to the current location.
                    for i, j in Sim.faux.values():
                        shutil.copy2(i, j)
                    # Write to disk: Force field object, current parameter values, target options
                    with wopen('forcebalance.p') as f: lp_dump((self.FF,mvals,Sim),f)
                    # Copy scripts to the current location.
                    for f in self.scripts:
                        LinkFile(os.path.join(os.path.split(__file__)[0], "data", f),
                                 os.path.join(os.getcwd(), f))
                    # Put together the command.
                    cmdlist = ['%s python md_one.py %s' % (self.mdpfx, Sim.type)]
                    if temp != None:
                        cmdlist.append('-T %g' % float(temp))
                    if pres != None:
                        cmdlist.append('-P %g' % float(pres))
                    # if AGrad or AHess:
                    #     cmdlist.append('-g')
                    # cmdlist.append('-o')
                    # cmdlist += Sim.timeseries.keys()
                    cmdstr = ' '.join(cmdlist)
                    print cmdstr
                    # # cmdstr = '%s python md1.py %s %.3f %.3f' % (self.runpfx, temperature, pressure)
                    # if wq == None:
                    #     logger.info("Running condensed phase simulation locally.\n")
                    #     logger.info("You may tail -f %s/npt.out in another terminal window\n" % os.getcwd())
                    _exec(cmdstr, copy_stderr=False, outfnm='md_one.out')
                    # else:
                    #     queue_up(wq, command = cmdstr+' &> npt.out',
                    #              input_files = self.nptfiles + self.scripts + ['forcebalance.p'],
                    #              output_files = ['npt_result.p.bz2', 'npt.out'] + self.extra_output, tgt=self)
                os.chdir(cwd)
        return

    def retrieve(self, dp):
        """Retrieve the molecular dynamics (MD) results and store the calculated
        observables in the Point object dp.

        Parameters
        ----------
        dp : Point
            Store the calculated observables in this point.

        Returns
        -------
        Nothing
        
        """
        abspath = os.path.join(os.getcwd(), '%d/md_result.p' % dp.idnr)

        if os.path.exists(abspath):
            logger.info('Reading data from ' + abspath + '.\n')

            vals, errs, grads = lp_load(open(abspath))

            dp.data["values"] = vals
            dp.data["errors"] = errs
            dp.data["grads"]  = grads

        else:
            msg = 'The file ' + abspath + ' does not exist so we cannot read it.\n'
            logger.warning(msg)

            dp.data["values"] = np.zeros((len(self.observables)))
            dp.data["errors"] = np.zeros((len(self.observables)))
            dp.data["grads"]  = np.zeros((len(self.observables), self.FF.np))
            
    def indicate(self):
        """Shows optimization state."""
        return
        AGrad     = hasattr(self, 'Gp')
        PrintDict = OrderedDict()
        
        def print_item(key, physunit):
            if self.Xp[key] > 0:
                the_title = ("%s %s (%s)\n" % (self.name, key.capitalize(), physunit) +
                             "No.   Temperature  Pressure  Reference  " +
                             "Calculated +- Stddev " +
                             "   Delta    Weight    Term   ")
                    
                printcool_dictionary(self.Pp[key], title=the_title, bold=True,
                                     color=4, keywidth=15)
                
                bar = printcool(("%s objective function: % .3f%s" %
                                 (key.capitalize(), self.Xp[key],
                                  ", Derivative:" if AGrad else "")))
                if AGrad:
                    self.FF.print_map(vals=self.Gp[key])
                    logger.info(bar)

                PrintDict[key] = (("% 10.5f % 8.3f % 14.5e" %
                                   (self.Xp[key], self.Wp[key],
                                    self.Xp[key]*self.Wp[key])))

        for i, q in enumerate(self.observables):
            print_item(q, self.points[0].ref["units"][i])

        PrintDict['Total'] = "% 10s % 8s % 14.5e" % ("","", self.Objective)

        Title = ("%s Thermodynamic Properties:\n %-20s %40s" %
                 (self.name, "Property", "Residual x Weight = Contribution"))
        printcool_dictionary(PrintDict, color=4, title=Title, keywidth=31)
        return

    def objective_term(self, observable):
        """Calculates the contribution to the objective function (the term) for a
        given observable.

        Parameters
        ----------
        observable : string
            Calculate the objective term for this observable.

        Returns
        -------
        term : dict
            `term` is a dict with keys `X`, `G`, `H` and `info`. The values of
            these keys are the objective term itself (`X`), its gradient (`G`),
            its Hessian (`H`), and an OrderedDict with print information on
            individiual data points (`info`).
            
        """
        Objective = 0.0
        Gradient  = np.zeros(self.FF.np)
        Hessian   = np.zeros((self.FF.np, self.FF.np))

        # Grab ref data for observable        
        qid       = self.observables.index(observable)
        Exp       = np.array([pt.ref["refs"][qid] for pt in self.points])
        Weights   = np.array([pt.ref["weights"][qid] for pt in self.points])
        Denom     = self.denoms[observable]
            
        # Renormalize weights
        Weights /= np.sum(Weights)
        logger.info("Renormalized weights to " + str(np.sum(Weights)) + "\n")
        logger.info(("Physical observable '%s' uses denominator = %g %s\n" %
                     (observable.capitalize(), Denom,
                      self.points[0].ref["units"][self.observables.index(observable)])))

        # Grab calculated values        
        values = np.array([pt.data["values"][qid] for pt in self.points])
        errors = np.array([pt.data["errors"][qid] for pt in self.points])
        grads  = np.array([pt.data["grads"][qid] for pt in self.points])

        # Calculate objective term using Least-squares function. Evaluate using
        # Einstein summation: W is N-array, Delta is N-array and grads is
        # NxM-array, where N is number of points and M is number of parameters.
        #
        #     X_i   = W_i * Delta2_i (no summed indices)
        #     G_ij  = W_i * Delta_i * grads_ij (no summed indices)
        #     H_ijm = W_i * gradsT_jk * grads_lm (sum over k and l)
        #
        # Result: X is N-array, G is NxM-array and H is NxMxM-array.
        #
        Deltas = values - Exp
        Objs   = np.einsum("i,i->i", Weights, Deltas**2) / Denom / Denom
        Grads  = 2.0*np.einsum("i,i,ij->ij", Weights, Deltas, grads) / Denom / Denom
        Hess   = 2.0*np.einsum("i,jk,lm->ijm", Weights, grads.T, grads) / Denom / Denom
        
        # Average over all points
        Objective += np.sum(Objs, axis=0)
        Gradient  += np.sum(Grads, axis=0)
        Hessian   += np.sum(Hess, axis=0)
        
        # Store gradients and setup print map 
        GradMapPrint = [["#Point"] + self.FF.plist]

        for pt in self.points:
            temp  = pt.temperature
            press = pt.pressure
            GradMapPrint.append([' %8.2f %8.1f' % (temp, press)] +
                                ["% 9.3e" % i for i in grads[pt.idnr-1]])

        o = wopen('gradient_%s.dat' % observable)
        for line in GradMapPrint:
            print >> o, ' '.join(line)
        o.close()
        
        printer = OrderedDict([("    %-5d %-12.2f %-8.1f"
                                % (pt.idnr, pt.temperature, pt.pressure),
                                ("% -10.3f % -10.3f  +- %-8.3f % -8.3f % -9.5f % -9.5f"
                                 % (Exp[pt.idnr-1], values[pt.idnr-1],
                                    errors[pt.idnr-1], Deltas[pt.idnr-1],
                                    Weights[pt.idnr-1], Objs[pt.idnr-1])))
                                    for pt in self.points])
                
        return { "X": Objective, "G": Gradient, "H": Hessian, "info": printer }

    def get(self, mvals, AGrad=True, AHess=True):
        """Return the contribution to the total objective function. This is a
        weighted average of the calculated observables.

        Parameters
        ----------
        mvals : list
            Mathematical parameter values.
        AGrad : Boolean
            Switch to turn on analytic gradient.
        AHess : Boolean
            Switch to turn on analytic Hessian.

        Returns
        -------
        Answer : dict
            Contribution to the objective function. `Answer` is a dict with keys
            `X` for the objective function, `G` for its gradient and `H` for its
            Hessian.
                    
        """
        Answer   = {}

        Objective = 0.0
        Gradient  = np.zeros(self.FF.np)
        Hessian   = np.zeros((self.FF.np, self.FF.np))
        return { "X": Objective, "G": Gradient, "H": Hessian} 

        for pt in self.points:
            # Update data point with MD results
            self.retrieve(pt)

        obj        = OrderedDict()
        reweighted = []
        for q in self.observables:
            # Returns dict with keys "X"=objective term value, "G"=the
            # gradient, "H"=the hessian, and "info"=printed info about points
            obj[q] = self.objective_term(q)
        
            # Apply weights for observables (normalized)
            if obj[q]["X"] == 0:
                self.weights[q] = 0.0

            # Store weights sorted in the order of self.observables
            reweighted.append(self.weights[q])
        
        # Normalize weights
        reweighted  = np.array(reweighted)
        wtot        = np.sum(reweighted)
        reweighted  = reweighted/wtot if wtot > 0 else reweighted
         
        # Picks out the "X", "G" and "H" keys for the observables sorted in the
        # order of self.observables. Xs is N-array, Gs is NxM-array and Hs is
        # NxMxM-array, where N is number of observables and M is number of
        # parameters.
        Xs = np.array([dic["X"] for dic in obj.values()])
        Gs = np.array([dic["G"] for dic in obj.values()])
        Hs = np.array([dic["H"] for dic in obj.values()])
                                
        # Target contribution is (normalized) weighted averages of the
        # individual observable terms.
        Objective    = np.average(Xs, weights=(None if np.all(reweighted == 0) else \
                                               reweighted), axis=0)
        if AGrad:
            Gradient = np.average(Gs, weights=(None if np.all(reweighted == 0) else \
                                               reweighted), axis=0)
        if AHess:
            Hessian  = np.average(Hs, weights=(None if np.all(reweighted == 0) else \
                                               reweighted), axis=0)

        if not in_fd():
            # Store results to show with indicator() function
            self.Xp = {q : dic["X"] for (q, dic) in obj.items()}
            self.Wp = {q : reweighted[self.observables.index(q)]
                       for (q, dic) in obj.items()}
            self.Pp = {q : dic["info"] for (q, dic) in obj.items()}

            if AGrad:
                self.Gp = {q : dic["G"] for (q, dic) in obj.items()}

            self.Objective = Objective
        
        Answer = { "X": Objective, "G": Gradient, "H": Hessian }
        return Answer
    
# class Point --- data container
class Point(object):
    def __init__(self, idnr, label=None, refs=None, weights=None, names=None,
                 units=None, temperature=None, pressure=None, data=None):
        self.idnr        = idnr
        self.ref         = { "label"  : label,                    
                             "refs"   : refs,
                             "weights": weights,
                             "names"  : names,
                             "units"  : units }
        self.temperature = temperature
        self.pressure    = pressure
        self.data        = data if data is not None else {}
        
    def __str__(self):
        msg = []
        if self.temperature is None:
            msg.append("State: Unknown.")
        elif self.pressure is None:
            msg.append("State: Point " + str(self.idnr) + " at " +
                       str(self.temperature) + " K.")
        else:
            msg.append("State: Point " + str(self.idnr) + " at " +
                       str(self.temperature) + " K and " +
                       str(self.pressure) + " bar.")

        msg.append("Point " + str(self.idnr) + " reference data " + "-"*30)
        for key in self.ref:
            msg.append("  " + key.strip() + " = " + str(self.ref[key]).strip())
            
        msg.append("Point " + str(self.idnr) + " calculated data " + "-"*30)
        for key in self.data:
            msg.append("  " + key.strip() + " = " + str(self.data[key]).strip())

        return "\n".join(msg)

class Simulation(object):

    """ 
    Data container for a MD simulation (specified by index, simulation
    type, initial condition).  These settings are written to a file
    then passed to md_one.py.

    The Simulation object is passed between the master ForceBalance
    process and the remote script (e.g. md_one.py).
    """

    type_settings = {'gas': {'pbc' : 0},
                     'liquid': {'pbc' : 1},
                     'solid': {'pbc' : 1, 'anisotropic_box' : 1},
                     'bilayer': {'pbc' : 1, 'anisotropic_box' : 1}}

    def __init__(self, tgt, data, name, index, stype, icn, tsnames):

        # The name of the simulation (refers to a directory under job.tmp/target/iter_x/index/name)
        self.name = name
        # The Index that the simulation belongs to.
        self.index = index
        # The type of simulation (liquid, gas, solid, bilayer...)
        if stype not in Simulation.type_settings.keys():
            logger.error('Simulation type %s is not supported at this time')
            raise RuntimeError
        # The reference data! May contain parameters for calculating observables.
        self.Data = copy.deepcopy(data)
        # Type of the simulation (map to simulation settings)
        self.type = stype
        # Locate the initial coordinate file and frame number.
        self.initial, self.iframe = find_file(os.path.join(tgt.root, tgt.tgtdir), index, stype, tgt.crdsfx, True, icn)
        # The time series for the simulation.
        self.timeseries = OrderedDict([(i, []) for i in tsnames])
        # The file extension that the coordinate file will be written with.
        self.fext = os.path.splitext(self.initial)[1]
        # Auxiliary files to be copied to the current location prior to running the simulation.
        self.faux = OrderedDict()
        for sfx in tgt.auxsfx:
            auxf = find_file(os.path.join(tgt.root, tgt.tgtdir), index, stype, sfx, False)[0]
            self.faux[os.path.splitext(auxf)[1]] = (auxf, "%s%s" % (self.type, os.path.splitext(auxf)[1]))
        # Name of the simulation engine
        self.engname = tgt.engname
        # Whether to use the CUDA platform (OpenMM only).
        self.force_cuda = tgt.OptionDict.get('force_cuda', 0)
        # Finite difference step size.
        self.h = tgt.h
        # Active parameters to differentiate over.
        self.pgrad = tgt.pgrad

        pbc = Simulation.type_settings[self.type]['pbc']

        #----
        # MD options, passed straight to the molecular_dynamics() method
        #----
        self.MDOpts = OrderedDict()
        # The time step in femtoseconds.
        self.MDOpts['timestep'] = tgt.timestep
        # The number of equilibration MD steps.
        self.MDOpts['nequil'] = tgt.eq_steps
        # The number of production MD steps.
        self.MDOpts['nsteps'] = tgt.md_steps
        # The number of MD steps between sampling.
        self.MDOpts['nsave'] = int(1000 * tgt.interval / self.MDOpts['timestep'])
        # Flag for minimizing the energy.
        self.MDOpts['minimize'] = tgt.OptionDict.get('minimize_energy', 0)
        # The number of threads for this simulation (no-PBC simulations are 1 thread).
        self.MDOpts['threads'] = tgt.OptionDict.get('md_threads', 1) if pbc else 1
        # Whether to use multiple timestep integrator.
        self.MDOpts['mts'] = tgt.OptionDict.get('mts_integrator', 0)
        # The number of beads in an RPMD simulation.
        self.MDOpts['rpmd_beads'] = tgt.OptionDict.get('rpmd_beads', 0)
        # Print out lots of information.
        self.MDOpts['verbose'] = True
        # Save trajectory to disk.
        self.MDOpts['save_traj'] = tgt.save_traj
        # Number of MD steps between successive calls to Monte Carlo barostat (OpenMM only).
        self.MDOpts['nbarostat'] = tgt.OptionDict.get('n_mcbarostat', 25)
        # Flag for anisotropic simulation cell (OpenMM only).
        self.MDOpts['anisotropic'] = tgt.OptionDict.get('anisotropic_box', 0)
        # The time step for the 'fast forces' in femtoseconds in MTS integrators.
        self.MDOpts['timestep'] = tgt.OptionDict.get('faststep', 0.25)
        # Simulation temperature in Kelvin.
        self.MDOpts['temperature'] = getval(self.Data, 'temp') if 'temp' in self.Data else None
        # Simulation pressure in bar.
        self.MDOpts['pressure'] = getval(self.Data, 'pres') if 'pres' in self.Data else None

        #----
        # Engine options, used in creating the Engine object
        #----
        self.EngOpts = OrderedDict()
        # Whether to use periodic boundary conditions.
        self.EngOpts['pbc'] = pbc
        # The name of the coordinate file to be written prior to running the simulation.
        self.EngOpts['coords'] = "%s%s" % (self.type, self.fext)
        # Software-specific options.
        if self.engname == 'openmm':
            self.EngOpts['platname'] = 'CUDA' if self.EngOpts['pbc'] else 'Reference'
        else:
            if self.force_cuda: 
                logger.error("force_cuda option is set, but has no effect on Gromacs engine.") ; raise RuntimeError
            if self.MDOpts['rpmd_beads'] > 0: 
                logger.error('Only the OpenMM engine can handle RPMD simulations.') ; raise RuntimeError
            if self.MDOpts['mts']: 
                logger.error('Only OpenMM is configured to use multiple timestep integrator.') ; raise RuntimeError
            if self.MDOpts['anisotropic']: 
                logger.error('Only OpenMM is configured to use anisotropic pressure coupling.') ; raise RuntimeError

        if self.engname == 'gromacs':
            self.EngOpts['gmxpath'] = tgt.gmxpath
            self.EngOpts['gmxsuffix'] = tgt.gmxsuffix
            self.EngOpts['gmx_top'] = self.faux['.top'][1]
            self.EngOpts['gmx_mdp'] = self.faux['.mdp'][1]

        if self.engname == 'tinker':
            self.EngOpts['tinkerpath'] = tgt.tinkerpath
            self.EngOpts['tinker_key'] = self.faux['.key'][1]

    def __str__(self):
        msg = []
        msg.append("Simulation: Name %s, Index %s, Type %s" % (self.name, self.index, self.type))
        msg.append("Initial Conditions: File %s Frame %i" % (self.initial, self.iframe))
        msg.append("Timeseries Names: %s" % (', '.join(self.timeseries.keys())))
        return "\n".join(msg)
