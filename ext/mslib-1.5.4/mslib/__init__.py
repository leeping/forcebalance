## Automatically adapted for numpy.oldnumeric Jul 30, 2007 by 

"""
Module mslib                                           Author: M. Sanner

This module is a high level interface tot he MSMS library developped by
M. Sanner at TSRI.

This module is built on top of the classes wrapping the MSMS C-structures
MOLSRF, RSR, RS, RSF, RSE, RSV, SESR, SES, SESF, SESE, SESV, TRI and TRIV.
These structures have been wrapped using SWIG. (see "HREF=./msms_wrap.html:: SWIG generated documentation of the wrapped mslib module/HREF")

This modules provides the Class MSMS derived from the MOSLRF C structure and
that implements methods to compute surfaces, find buried surfaces, update
surfaces after a subset of atoms have been assigned new components. The class
also offers support for writting out the surfaces in different formats.

examples:
    
    >>> # computing a surface with default paramters and getting the
          from a file coordinates
    >>> import mslib
    >>> m = mslib.MSMS(filename='/home/sanner/links/oldhome/data/1crn.xyzrn')
    >>> m.compute()
    >>> print m.info()
    
    >>> # computing a surface with scaled radii and a probe of 1.7 Angstroem
    >>> coords, names = mslib.readxyzr('/home/sanner/links/oldhome/data/1crn.xyzrn')
    >>> rad = coords[:,3]*0.9
    >>> m1 = mslib.MSMS(name='mysurf', coords=coords[:,:3], radii=rad,
                        atnames = names)
    >>> m1.compute(probe_radius=1.7)

"""


import numpy.oldnumeric as Numeric
#import msmsc
import msms

def readxyzr(filename):
    """coords, names <- readxyzr(filename)
    function to read an xrzr or xyzrn file. The returned coords will be a 4xN
    array of floating point values for describing x,y,z and rt for N atoms,
    names will be n strings. All names will be empty string if the file does
    not provide names"""
    return msms.MS_readxyzr(filename)


class MSMS(msms.MOLSRF):
    """High level molecule surface class.

    Such an object can be built by passing wither a filename, an array of Nx4
    coordinates or an array of Nx3 centers and an array of N radii.
    The class provide methods to compute: one or several reduced surface
    components (RS), the corresponding analytical models of the solvent
    excluded surface (SES) and triangulations of these models.
    In addition, methods to compute analytical and numerical surface areas for
    the SES and SAS are available.
    This object also provides methods to compute numerically patches of a SES
    component buried by a second set of spheres.
    Finally, methods allowing to assign new coordinates toa subset of atoms
    and re-compute the surface partially are available.
    """
    
    def __init__(self, coords=None, radii=None, atnames=None, filename=None,
                 name='msms', maxnat=0, surfflags=None, hdflags=None):
        """MSMS <- MSMS(coords=None, radii=None, atnames=None, filename=None,
                        name='msms', maxnat=0, surfflags=None, hdflags=None)

    This class can be instantiated int a number of ways:

    - no arguments:
         m = MSMS()
    - with an array of Nx4 coordinates:
         m = MSMS(coords = c)
    - with an array of Nx3 centers and an array of radii:
         m = MSMS(coords = c, radii = r)
    - by specifying a filename from wich to load the xyzr and possibly names
         m = MSMS(filename)

    additional parameters are:
    name:    string to name that surface
    atnames: array of N atom names
    maxnat:  maximum number of atoms in the molecule. This number needs to be
             larger than largest number of atoms that surface will ever have.
    """
        if filename:
            c, atnames, surfflags, hdflags = readxyzr(filename)
            nat = len(c)
            if name: self.surface_name = name
            else: self.surface_name = os.basename(filename)
        else:
            self.surface_name = name
            if coords:
                self.coords = Numeric.array(coords).astype('f')
                assert len(self.coords.shape)==2, "coordinates array of bad shape"
                if self.coords.shape[1] == 3:
                    self.radii = Numeric.array(radii).astype('f')
                    self.radii.shape = (self.radii.shape[0],1)
                    c = Numeric.concatenate( (self.coords, self.radii), 1 )
                else:
                    assert self.coords.shape[1]==4, "coordinates array of bad shape"
                    c = self.coords
                if not c.flags.contiguous or c.dtype.char!='f':
                    c = Numeric.array(c).astype('f')
                nat = len(c)
            else:
                c=None
                nat=0
        if surfflags:
            assert len(surfflags)==len(c)
        if hdflags:
            assert len(hdflags)==len(c)

        msms.MOLSRF.__init__(self, name=name, coords=c, nat=nat,
                             maxat=maxnat, names=atnames, surfflags=surfflags,
                             hdflags=hdflags)
        self.geometry = None


    def _getSESComp(self, num):
        """helper function to get an SES component from it's number"""
        comp = self.sesr.fst
        for i in range(num):
            comp = comp.nxt
        return comp


    def _getRSComp(self, num):
        """helper function to get an RS component from it's number"""
        comp = self.rsr.fst
        for i in range(num):
            comp = comp.nxt
        return comp


    def compute(self, probe_radius=1.5, density=1.0, hdensity=3.0):
        """err <- self.compute(probe_radius=1.5, density=1.0, seedAtoms=[])
    
    Compute the external component of the reduced surface with the
    specified probe_radius, then compute the solvent exclude surface
    and triangulate it with the specified density.
    Return MS_OK upon successfull completion or MS_ERR when it fails. In
    the latter case an error message is available in MS_ERR_MSG"""
        return msms.MS_compute_surface(self.this, probe_radius, density,
                                       hdensity)


    def compute_rs(self, probe_radius=1.5, allComponents=0,
                   atoms=None):
        """self.compute_rs(probe_radius=1.5, allComponents=0, atoms=None)

    Compute the external component of the reduced surface.
    probe_radius:  should be larger than 0.1.
    allComponents: can be either 0 or 1. When it is 1 all surface components
                   are looked for and the complexity becomes N^2
    atoms:         list of 3 0-based atom indices used to seed the RS
                   calculation. This is useful to compute a specific component
    """
        msms.MS_reset_msms_err()

        if probe_radius != self.rp:
            if ( probe_radius < 0.1 ):
                raise RuntimeError("Probe radius has to be > 0.1")
            if self.rsr.nb !=0: # new probe radius ==> delete RS, SES
                msms.MS_reset_RSR(self.this)
                s = self.sesr.fst
                while s:
                    msms.MS_free_triangulated_surface(s.this)
                    s=s.nxt
                msms.MS_reset_SESR(self.this)

                rsr = self.rsr  # initialize RSR
                rsr.fst = None
                rsr.lst = None
                rsr.nb = 0
                rsr.nb_free_vert = 0
                rsr.ffnba = 0
                rsr.set_ffat( -1, -1, -1 )
                rsr.all_comp_done = 0
                rsr.ext_comp_done = 0

                sesr = self.sesr # initialize SESR
                sesr.fst = None
                sesr.lst = None

                sesr.nb = 0

            self.rp = probe_radius
            self.rp1 = 1.0/probe_radius


        if allComponents != self.all_components:
            msg = "Only 0 and 1 are valid for allComponents"
            assert allComponents in (0,1), msg
            self.all_components = allComponents
            
        if atoms:
            self.all_components = 0;
            self.rsr.ffnba = 0;
            dum = [-1, -1, -1]
            i=0
            for j in (0,1,2):
                if atoms[j]!=-1:
                    self.rsr.ffnba = self.rsr.ffnba+1
                    dum[i] = atoms[j]
                    i = i + 1
            self.rsr.set_ffat( dum[0], dum[1], dum[2] )

        status = msms.MS_reduced_surface( self.this )
        if status != msms.MS_OK:
            raise RuntimeError("Could not construct the reduced surface.")


    def compute_ses(self, component=None):
        """self.compute_ses(component=None)

    Compute the analytical SES for the specified component correponding to a
    user specified component of the reduced surface.
    if component is None, all components are calculated. Components are
    specified using 0-based integers, 0 being the external component
    """
        msms.MS_reset_msms_err()
        if component==None:
            rs = self.rsr.fst
            while rs:
                msms.MS_solvent_excluded_surface(self.this, rs.this)
                rs=rs.nxt
        else:
            rs = self._getRSComp(component).this
            msms.MS_solvent_excluded_surface(self.this, rs)


    def triangulate_one_component(self, rs, density=1.0,  hdensity=3.0):
        """triangulate_one_component(rs, density=1.0)
        
    Triangulate a given solvent excluded component, with the specified
    density
        """
        ses = rs.ses
        if density != ses.density:
            msms.MS_free_triangulated_surface(ses.this)
            msms.MS_set_SES_density( ses.this, density,  hdensity, self.rp )

        if ses.nbtri:
            raise RuntimeError, "This SES component is already triangulated"
        print "making templates"
        status = msms.MS_make_templates( self.this, ses.this )
        if status != msms.MS_OK:
            raise RuntimeError, "Could not construct triangulation templates"

        status = msms.MS_triangulate_SES_component( self.this, rs.this )
        if status != msms.MS_OK:
            raise RuntimeError, "Could not triangulate SES component"


    def triangulate(self, density=1.0, component=None):
        """triangulate(self, density=1.0, component=None)

    Triangulate analytical SES models using a user specified density.
    If component is None, all components are calculated. Components are
    specified using 0-based integers, 0 being the external component
        """
        msms.MS_reset_msms_err()
        if component==None:
            rs = self.rsr.fst
            while rs:
                if rs.ses.this:
                    self.triangulate_one_component(rs, density)
                rs=rs.nxt

        else:
            rs = self._getRSComp(component)
            self.triangulate_one_component(rs, density=density)


    def write_triangulation(self, filename, no_header=1, component=None,
                            format=msms.MS_TSES_ASCII ):
        """write_triangulation(filename, no_header=1, component=None,
                               format=msms.MS_TSES_ASCII)

    Write the solvent excluded surface to a file.
    no_header allows to add/remove the header lines to these files
    If component is None, all components are calculated. Components are
    specified using 0-based integers, 0 being the external component
    format can be one of the following:
        MS_TSES_ASCII     : Triangulated surface in ASCII format
        MS_ASES_ASCII     : Analytical surface in ASCII format. This is
                            actually a discrete representation of the
                            analytical model.
        MS_TSES_ASCII_AVS : Triangulated surface in ASCII with AVS header
        MS_ASES_ASCII_AVS : Analytical surface in ASCII format with AVS header
    """
        f = msms.MS_write_triangulated_SES_component
        if component is None:
            rs = self.rsr.fst
            i=0
            while rs:
                if rs.ses.this and rs.ses.nbtri>0:
                    _filename = "%s_%d"%(filename, i)
                    f( _filename, self.this, rs.ses.this, no_header, format)
                    i = i + 1
                rs=rs.nxt
        else:
            comp = self._getSESComp(component).this
            f( filename, self.this, comp, no_header, format)
            

    def getTriangles(self, atomIndices=None, selnum=1, base=0, component=0, keepOriginalIndices=0):
        """vfloat, vint, tri <- getTriangles(atomIndices=None, base=0, component=0)
    Return numeric arrays for vertices floating point (vfloat) and integer
    (vint) data and triangles data (tri).
    For each vertex vfloat provides (x,y,z,nx,ny,nz,sesA,sasA)
    for each vertex vint provides (type, closestAtomindex, buried), where
    type can be -1 for SESV, -2 for SESE and the SESF number for vertices
    inside faces; closestAtomindex is the 1 based index of the atom closest
    to that vertex and buried is 0 or 1 if this vertex is buried.
    For each triangles tri provides (i, j, k, type, SESF_num), where type can
    be 1:contact, 2:reentrant, 3:toroidal, 4:full torus and
    5:full torus with radial singlarity.
    The triple (i,j,k) describing the triangle's connectivity can be 0-based
    or 1-based according to the value of base (0 or 1).
    Components are specified using 0-based integers, 0 being the external
    component
        """
        comp = self._getSESComp(component).this
        return msms.MS_get_triangles(self.this, comp, atomIndices,
                                      selnum, base, keepOriginalIndices)


    def compute_ses_area(self):
        """err <- compute_ses_area()

    Compute the surface area of all faces of the analytical SES model.
    after that computation, each SESF has its a_ses_area member set
    and the SES has the members a_reent_area, a_toric_area, a_contact_area and
    a_ses_area and a_sas_area set.
    """

        return msms.MS_compute_SES_area(self.this)


    def compute_numeric_area_vol(self, component=None,
                                 mode=msms.MS_SEMI_ANALYTICAL):
        """compute_numeric_area_vol(component=None,mode=msms.MS_SEMI_ANALYTICAL)

    compute the surface area using the triangulated surface.
    If component is None, all components are calculated. Components are
    specified using 0-based integers, 0 being the external component.
    mode can be one of the following:

    - MS_NUMERICAL:       use triangle surface area
    - MS_SEMI_ANALYTICAL: use spherical triangles surface areas for contact
                          and reentrant faces
    - MS_BOTH:            do both calculations
    """
        
        assert mode in (msms.MS_NUMERICAL, msms.MS_SEMI_ANALYTICAL,
                        msms.MS_BOTH,)
        
        if component is None:
            rs = self.rsr.fst
            i=0
            while rs:
                if rs.ses.this and rs.ses.nbtri>0:
                    msms.MS_compute_numerical_area_vol( self.this,
                                                         rs.ses.this, mode)
                    i = i + 1
                rs=rs.nxt
        else:
            comp = self._getSESComp(component).this
            msms.MS_compute_numerical_area_vol(self.this, comp, mode)


    def write_ses_area(self, filename, component=None):
        """write_ses_area(filename, component=None)

    Write the surface areas to an output files. Results are written by atom.
    If component is None, each component is wrtten to a file. Components are
    specified using 0-based integers, 0 being the external component.
    """
        if component is None:
            rs = self.rsr.fst
            i=0
            while rs:
                if rs.ses.this and rs.ses.nbtri>0:
                    _filename = "%s_%d"%(filename, i)
                    msms.MS_write_surface_areas(self.this, _filename, i)
                    i = i + 1
                rs=rs.nxt
        else:
            msms.MS_write_surface_areas(self.this, filename, component)


    def detailed_info(self):
        """ string <- detailed_info(()

    return a string describing this molecular surface
    """
        return msms.MS_detailed_info_molsrf(self.this)


    def info(self):
        """ string <- info(()

    return a string describing this molecular surface
    """

        return msms.MS_info_molsrf(self.this)


    def resetBuriedVertexFlags(self, component=None):
        """resetBuriedVertexFlags(component=None)

    Reset the buried flag on all SES triangulation vertices.
    If component is None, this operation is performed on all components.
    Components are specified using 0-based integers, 0 being the external one
    """
    
        if component is None:
            rs = self.rsr.fst
            i=0
            while rs:
                if rs.ses.this and rs.ses.nbtri>0:
                    msms.MS_resetBuriedVertexFlags(rs.ses.this)
                    i = i + 1
                rs=rs.nxt
        else:
            comp = self._getSESComp(component).this            
            msms.MS_resetBuriedVertexFlags(comp)

    def buriedVertices(self, coords, radii=None, component=None):
        """buriedVertices(coords, radii=None, component=None)

    tags all vertices of an SES trianuglated component that are buried by a
set of spheres.
- If radii is None, coords has to be an Mx4 array of floats describing the
centers and radii of the set of spheres used to check if the surface is
burried. Else coords has to be an Mx3 array describing the centers and the
radii have to be provided in the radii array.
- If component is None, this operation is performed on all components.
Components are specified using 0-based integers, 0 being the external one
"""
        if radii is not None:
            radii = Numeric.array(radii, 'f')
            radii.shape = (-1, 1)
            assert len(radii)==len(coords)
            coords = Numeric.array(coords, 'f')
            coords = Numeric.concatenate( (coords, radii), 1 )
        else:
            coords = Numeric.array(coords, 'f')
            
        assert len(coords.shape)==2 and coords.shape[1] == 4

        if component is None:
            self.resetBuriedVertexFlags()
            rs = self.rsr.fst
            i=0
            while rs:
                if rs.ses.this and rs.ses.nbtri>0:
                    self.findBuriedVertices( rs.ses.this, coords, len(coords))
                    i = i + 1
                rs=rs.nxt
        else:
            self.resetBuriedVertexFlags(component)
            comp = self._getSESComp(component).this
            self.findBuriedVertices(comp, coords, len(coords))            


    def resetBuriedVertexArea(self, component=None):
        """resetBuriedVertexArea(component=None)

    Reset the surface area assigned to each vertex in a triangulated SES
    component.
    If component is None, this operation is performed on all components.
    Components are specified using 0-based integers, 0 being the external one
    """
        if component is None:
            rs = self.rsr.fst
            i=0
            while rs:
                if rs.ses.this and rs.ses.nbtri>0:
                    msms.MS_resetBuriedVertexArea(rs.ses.this)
                    i = i + 1
                rs=rs.nxt
        else:
            comp = self._getSESComp(component).this            
            msms.MS_resetBuriedVertexArea(comp)


    def buriedSurfaceArea(self, component=None, mode=msms.MS_SEMI_ANALYTICAL):
        """buriedSurfaceArea(component=None,mode=msms.MS_SEMI_ANALYTICAL)

    Compute the surface area corresponding to the buried vertices.
    If component is None, this operation is performed on all components.
    Components are specified using 0-based integers, 0 being the external one
    mode can be one of the following:
 
    - MS_NUMERICAL:       use triangle surface area
    - MS_SEMI_ANALYTICAL: use spherical triangles surface areas for contact
                          and reentrant faces
    - MS_BOTH:            do both calculations

    Return: a dictionnary with 2 keys: 'ses' and 'sas'. The values are either
    lists of buried surface areas for all components considered.
    """
        assert mode in (msms.MS_NUMERICAL, msms.MS_SEMI_ANALYTICAL,
                        msms.MS_BOTH,)

        areas = {'ses':[], 'sas':[] }
        if component is None:
            self.resetBuriedVertexArea()
            self.compute_numeric_area_vol(mode=mode)
            rs = self.rsr.fst
            i=0
            while rs:
                if rs.ses.this and rs.ses.nbtri>0:
                    msms.MS_vertexBuriedSurfaceArea( rs.ses.this )
                    areas['ses'].append(rs.ses.n_buried_ses_area)
                    areas['sas'].append(rs.ses.n_buried_sas_area)
                    i = i + 1
                rs=rs.nxt
        else:
            comp = self._getSESComp(component)
            self.resetBuriedVertexArea(component)
            self.compute_numeric_area_vol(component, mode=mode)
            msms.MS_vertexBuriedSurfaceArea(comp.this)
            areas['ses'].append(comp.n_buried_ses_area)
            areas['sas'].append(comp.n_buried_sas_area)
        return areas
    

    def getBuriedSurfaceTriangles(self, atomIndices=None, component=0, selnum=1, negate=0):
        """vfloat, vint, tri = getBuriedSurfaceTriangles(atomIndices=None, component=0, selnum=1, negate=0)

    Return the triangles of the specified SES component for which at least
    'selnum' vertices are either buried (if negate=0) or not burried
    (if negate=1). 0 < selnum < 4.
    
    vfloat and vint hold the data for all vertices of the surface.
    tri contains the subset of the triangles that are buried.
    """

        assert selnum in (1,2,3)

        vfloat, vint, tri = self.getTriangles(atomIndices, component=component)
        buriedFlag = vint[:,2]
        if negate:
            buriedFlag = Numeric.logical_not(buriedFlag)
        triBuried = Numeric.choose(tri[:,:3], buriedFlag)
        sum = Numeric.sum(triBuried, 1)
        faceInd = Numeric.nonzero( Numeric.greater_equal(sum, selnum) )
        faces = Numeric.take(tri, faceInd)
        return vfloat, vint, faces
    

    def initUpdateSurface(self, cut=15.0, cleanup_ses=0):
        """initUpdateSurface(cut=15.0, cleanup_ses=0)

    Call the initialize the data structure before the first partial
    recomputation can be performed.
    cut: is used as a radius to select probes in fixed positions close to
         the moving atoms
    cleanup_ses: this flag has to be set to 1 in order for the genus of the
                 updated SES to make sense.
    """
        self.cleanup_ses = cleanup_ses
        msms.MS_reset_atom_update_flag( self.this )
        msms.MS_tagCloseProbes(self.this, self.rsr.fst, cut)


    def updateCoordsFromFile(self, filename, max=-1):
        """updateCoordsFromFile(filename, max=-1)

    Read a set of new coordinates from a file and assign them to the atoms.
    max can be -1 for all atoms specified in the file to be used of any
    integer smaller than the number of atoms in the file.
    """
        msms.MS_reset_atom_update_flag( self.this )
        msms.MS_get_xyzr_update(self.this, filename, max)


    def updateSpheres(self, coords, indices):
        """err <- updateSpheres(coords, indices)

    Not tested
    coords: Nx4 array of floats
    indiced: indices of moving atoms
        """
        msms.MS_reset_atom_update_flag( self.this )
        msms.MS_updateSpheres(self.this, len(coords), indices, coords)

        
    def updateSurface(self, mode=msms.FULL, density=1.0, update=0):
        """err <- updateSurface(mode=mslib.FULL, density=1.0, update=0)

    Recompute the surface after a subset of atoms assume new coordinates.
    mode can be one of the following:
        FULL             : the surface is rebuilt completely
        TORIC            : only toric faces are triangulated
        ALL_WITH_DENSITY : all rebuilt faces are triangulated with the
                           given density
        define NO_TRIANGULATION : no triangulation at all
    density: density of the reconstructed surface patches
    update: update number
    """
        assert mode in (msms.FULL, msms.TORIC, msms.ALL_WITH_DENSITY,
                        msms.NO_TRIANGULATION)

        stat = msms.MS_update_surface(self.this, self.rsr.fst, mode,
                                       density, update)

        return stat
#       if stat==msms.MS_ERR:
#	    print "ERROR while updating RS %d %s\n" % (update ,msms.MS_err_msg)


    def updateSurfaceArea(self):
        """updateSurfaceArea()

        Update the surface areas after the surface's geometry has been
        re-computed.
        """
        return msms.MS_update_SES_area(self.this, self.rsr.fst.ses)


    def update(self, mode=msms.FULL, density=None, nup=0):

        if density is None: density = self.density
        assert mode in (msms.FULL, msms.TORIC, msms.ALL_WITH_DENSITY,
                        msms.NO_TRIANGULATION)
        rs = self.rsr.fst
        rs.fstrfup = None
        rs.lstrfup = None
        rs.fstreup = None
        rs.lstreup = None
        rs.fstRSbfup = None
        rs.lstRSbfup = None
        rs.fstfup = None
        rs.lstfup = None
        rs.fsteup = None
        rs.lsteup = None
        
        su = rs.ses
        su.fsttfup = None
        su.lsttfup = None
        su.fsttorup = None
        su.lsttorup = None
        if self.try_num > 1:
            msms.MS_restore_radii(self.this, rs.this)
        i = msms.MS_update_reduced_surface(self.this,rs.this,nup)
        if i==msms.MS_ERR:
            raise RuntimeError, "ERROR while updating RS %s\n"%msms.MS_err_msg
        su.lsesv = None
        su.lsese = None
        su.lsesf = None
        i = msms.MS_update_ses(self.this,rs.this,nup)
        if i==msms.MS_ERR:
            raise RuntimeError, "ERROR while updating SES %s\n"%msms.MS_err_msg

        i = msms.MS_update_triangulation_SES_component(self.this, rs.this,
                                                        mode, density, nup)
        if i==msms.MS_ERR:
            raise RuntimeError, "ERROR while updating SES %s\n"%msms.MS_err_msg


    def getColorByType(self, component=None):
        """colors <- getColorByType(component=None)

        Compute the surface area corresponding to the buried vertices.
        If component is None, this operation is performed on all components.
        Components are specified using 0-based integers, 0 being the external one
        it returns a list of (r,g,b) tuples for each component
        """
	colors = ( (1.,0.,1.), (1., 0.5, 0.),
                   (0.,1.,0.), (1., 0., 0.), (0., 0., 1.) )
        col = []
        if component is None:
            rs = self.rsr.fst
            i=0
            while rs:
                if rs.ses.this and rs.ses.nbtri>0:
                    dum, dum, tri = self.getTriangles(component=i)
                    col.append( Numeric.take(colors, tri[:,3]-1) )
                    i = i + 1
                rs=rs.nxt
        else:
            comp = self._getSESComp(component).this            
            dum, dum, tri = self.getTriangles(component=component)
            col = Numeric.take(colors, tri[:,3]-1)
        return col


##      def getAnaFaceTriangles(self, num):
##  	"""returns the set of triangles of an analytical SES face"""
##  	fact = Numeric.equal( self.sesf[:,4]-1, num )
##  	find = Numeric.nonzero(fact)
##  	if len(find):
##  	    fac = Numeric.take( self.sesf, find )
##  	    vertsFlag = Numeric.zeros( (len(self.sesv),), 'i' )
##  	    for f in fac:
##  		vertsFlag[f[0]] = 1;
##  		vertsFlag[f[1]] = 1;
##  		vertsFlag[f[2]] = 1;
##  	    vind = Numeric.nonzero(vertsFlag)
##  	    vts = Numeric.take( self.sesv, vind )
##  	return vts, fac



##      def displayMsms(self, faces=None, only=1, analytical=0, viewer=None):
##  	"""Display SES triangles.

##  	obj.displayMsms( faces=None, only=1, analytical=0, viewer=None)

##  	faces: None for all triangles, [i,j,...,n] for triangles i,j,..,n
##  	only: define whether faces are added to or replace the current set
##  	analytical: when set, the indices in faces are analytical faces
##  	viewer: allows to specify a viewer in which to show the faces
##  	"""

##  	if self.viewer is None: self.getViewer(viewer)
##  	if self.msmsGeom is None:
##  	    from viewer.IndexedPolygons import IndexedPolygons
##  	    self.msmsGeom = IndexedPolygons('SES', vertices=self.sesv[:,:3] )
##  	    self.viewer.AddObject(self.msmsGeom)
##  	if faces is None:
##  	    fac = faces=self.sesf[:,:3]
##  	    mat = self.sesfColors
##  	elif faces == 'toric':
##  	    fact = Numeric.equal( self.sesf[:,3], 3 )
##  	    ind = Numeric.nonzero(fact)
##  	    fac = Numeric.take( self.sesf[:,:3], ind )
##  	    mat = Numeric.take( self.sesfColors, ind )
##  	elif faces == 'reent':
##  	    fact = Numeric.equal( self.sesf[:,3], 2 )
##  	    ind = Numeric.nonzero(fact)
##  	    fac = Numeric.take( self.sesf[:,:3], ind )
##  	    mat = Numeric.take( self.sesfColors, ind )
##  	elif faces == 'contact':
##  	    fact = Numeric.equal( self.sesf[:,3], 1 )
##  	    ind = Numeric.nonzero(fact)
##  	    fac = Numeric.take( self.sesf[:,:3], ind )
##  	    mat = Numeric.take( self.sesfColors, ind )
##  	else:
##  	    assert len(faces) > 0
##  	    if analytical == 0 :
##  		fac = Numeric.take( self.sesf[:,:3], list(faces) )
##  		mat = Numeric.take( self.sesfColors, list(faces) )
##  	    else:
##  		fact = Numeric.equal( self.sesf[:,4], faces[0] )
##  		for fi in faces[1:]:
##  		    fact = fact + Numeric.equal( self.sesf[:,4], fi )
##  		ind = Numeric.nonzero(fact)
##  		fac = Numeric.take( self.sesf[:,:3], ind )
##  		mat = Numeric.take( self.sesfColors, ind )
		
##  	if only:
##  	    self.msmsGeom.Set( faces=fac, materials=mat )
##  	else:
##  	    self.msmsGeom.Add( faces=fac, materials=mat )

##      def labelTriangles(self, triang=None, only=1, viewer=None):
##  	"""Label SES triangles.

##  	obj.labelTriangles(self, triang=None, only=1, viewer=None)

##  	triang: None for all triangles or list of triangle indices
##  	only: define whether triangles are added to, or replace the current set
##  	viewer: allows to specify a viewer in which to show the faces
##  	"""

##  	if self.viewer is None: self.getViewer(viewer)
##  	if self.triangLabelsGeom is None:
##  	    from viewer.Labels import Labels
##  	    self.triangLabelsGeom = Labels( 'TriangLab', shape=(0,3) )
##  	    self.viewer.AddObject(self.triangLabelsGeom)

##  	triCenters = Numeric.take( self.sesv[:,:3], self.sesf[:,:3] )
##  	triCenters = Numeric.sum(triCenters, 1)/3.0
##  	atlabs = map(str, range(len(self.sesf)))

##  	if triang is None:
##  	    labs = atlabs
##  	    pos = triCenters
##  	else:
##  	    assert len(triang) > 0
##  	    labs = Numeric.take( atlabs, list(triang) )
##  	    pos = Numeric.take( triCenters, list(triang) )

##  	if only:
##  	    self.triangLabelsGeom.Set( labels=labs, vertices=pos)
##  	else:
##  	    self.triangLabelsGeom.Add( labels=labs, vertices=pos)


##      def labelFaces(self, faces=None, only=1, viewer=None):
##  	"""Label SES Analytical faces.

##  	obj.Faces(self, faces=None, only=1, viewer=None)

##  	faces: None for all faces or list of faces indices
##  	only: define whether faces are added to, or replace the current set
##  	viewer: allows to specify a viewer in which to show the faces
##  	"""

##  	if self.viewer is None: self.getViewer(viewer)
##  	if self.facesLabelsGeom is None:
##  	    from viewer.Labels import Labels
##  	    self.facesLabelsGeom = Labels( 'FacesLab', shape=(0,3) )
##  	    self.viewer.AddObject(self.facesLabelsGeom)

##  	if faces is None: faces = range(self.sesf[:,4][-1])
##  	else: assert len(faces) > 0

##  	facesCenters = []
##  	colors = []
##  	rem = []
##  	triangs = self.sesf[:,:3]
##  	verts = self.sesv[:,:3]
##  	anaFacesNum = self.sesf[:,4]
##  	for fi in faces:
##  	    fact = Numeric.equal( anaFacesNum, fi )
##  	    ind = Numeric.nonzero(fact)
##  	    if len(ind):
##  		fac = Numeric.take( triangs, ind )
##  		facCenter = Numeric.take( verts, fac )
##  		facCenter = Numeric.sum(facCenter, 1)/3.0
##  		l = len(facCenter)
##  		facCenter = Numeric.sum(facCenter)/ l
##  		facesCenters.append( Numeric.reshape(facCenter, (3,)) )
##  		if self.sesfColors:
##  		    colors.append( self.sesfColors[ind[0]] )
##  	    else: rem.append(fi)

##  	for f in rem: faces.remove(f)
##  	atlabs = map(str, faces)
##  	pos = Numeric.array(facesCenters)
##  	if len(colors)==0: colors = ( (1.,1.,1.), )

##  	if only:
##  	    self.facesLabelsGeom.Set( labels=atlabs, vertices=pos,
##  				      materials=colors)
##  	else:
##  	    self.facesLabelsGeom.Add( labels=atlabs, vertices=pos,
##  				      materials=colors)


##      def labelVertices(self, vertices=None, analytical=0, only=1, viewer=None):
##  	"""Label SES vertices.

##  	obj.labelVertices(self, triang=None, only=1, viewer=None)

##  	vertices: None for all vertices or list of vertex indices or
##  	          list of analytical face numbers
##  	analytical: when set, the indices in faces are analytical faces
##  	only: define whether vertices are added to, or replace the current set
##  	viewer: allows to specify a viewer in which to show the faces
##  	"""

##  	if self.viewer is None: self.getViewer(viewer)
##  	if self.vertLabelsGeom is None:
##  	    from viewer.Labels import Labels
##  	    self.vertLabelsGeom = Labels( 'vertLab', shape=(0,3) )
##  	    self.viewer.AddObject(self.vertLabelsGeom)

##  	if vertices is None: vertices = range( len(self.sesv) )
##  	labs = map(str, vertices)

##  	if vertices is None:
##  	    labs = atlabs
##  	    pos = self.sesv[:,:3]
##  	else:
##  	    assert len(vertices) > 0
##  	    if analytical == 0 :
##  		pos = Numeric.take( self.sesv[:,:3], list(vertices) )
##  	    else:
##  		fact = Numeric.equal( self.sesf[:,4], vertices[0] )
##  		for fi in vertices[1:]:
##  		    fact = fact + Numeric.equal( self.sesf[:,4], fi )
##  		ind = Numeric.nonzero(fact)
##  		fac = Numeric.take( self.sesf[:,:3], ind )
##  		vertsFlag = Numeric.zeros( (len(self.sesv),) )
##  		for f in fac:
##  		    vertsFlag[f[0]] = 1;
##  		    vertsFlag[f[1]] = 1;
##  		    vertsFlag[f[2]] = 1;
##  		vind = Numeric.nonzero(vertsFlag)
##  		pos = Numeric.take( self.sesv[:,:3], vind )
##  		labs = map(str, vind)

##  	if only:
##  	    self.vertLabelsGeom.Set( labels=labs, vertices=pos)
##  	else:
##  	    self.vertLabelsGeom.Add( labels=labs, vertices=pos)


##  FUNCTIONS REMAINING TO BE EXPOSED IN MSMS CLASS
##
##  def MS_find_interface_atoms(*_args, **_kwargs):
##

##  def MS_set_SES_density(*_args, **_kwargs):
##  def MS_printInfo(*_args, **_kwargs):
##  def MS_genus(*_args, **_kwargs):

##  def MS_write_coordinates(*_args, **_kwargs):
##  def MS_write_rs_component(*_args, **_kwargs):
##  def MS_writeSolventAccessibleatoms(*_args, **_kwargs):

##  FUNCTIONS THAT WE WILL PROBABLY NOT EXPOSE IN MSMS CLASS
##
##  def MS_free_reduced_surface(*_args, **_kwargs):
##  def MS_free_SES_comp(*_args, **_kwargs):
##  def MS_free_SES(*_args, **_kwargs):
##  def MS_free_templates(*_args, **_kwargs):
##  def MS_free_triangulated_surface(*_args, **_kwargs):
##  def MS_tagCloseProbes(*_args, **_kwargs):
##  def MS_restore_radii(*_args, **_kwargs):


####################################################################

##      def display(self, atomIndices=None, component=0, only=1):
##          assert component < len(self.this.rs)
##          self.viewer = self.GetViewer()
##          s = self.this.rs[component].ses
##          vertFloat, vertInt, faces = self.getTriangles(atomIndices)
##  	if not self.geometry:
##  	    from DejaVu.IndexedPolygons import IndexedPolygons
##  	    self.geometry = IndexedPolygons(self.name)
##  	    self.viewer.AddObject(self.geometry)

##  	if only:
##  	    self.geometry.Set( vertices = vertFloat[:,:3],
##                               vnormals = vertFloat[:,3:6],
##                               faces = faces[:, :3] )
##  	else:
##  	    self.geometry.Add( vertices = vertFloat[:,:3],
##                               vnormals = vertFloat[:,3:6],
##                              faces = faces[:, :3]+len(self.geometry.vertexSet) )

##  	self.viewer.Redraw()
__MGLTOOLSVersion__ = '1-4alpha3'
CRITICAL_DEPENDENCIES = ['Numeric', 'mglutil']
NONCRITICAL_DEPENDENCIES = ['MolKit','Pmv','DejaVu']

