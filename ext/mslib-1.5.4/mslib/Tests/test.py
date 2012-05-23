## Automatically adapted for numpy.oldnumeric Jul 30, 2007 by 

# test.py : tests msms library.  Run from ../../ :
#    tester mslib

import sys, os
import numpy.oldnumeric as Numeric
from mglutil.regression import testplus

from MolKit import Read
coords = None
rad = None


def setUpSuite():
    from mslib import msms
    print "msms imported from: ", msms.__file__
    newmol = Read("Data/1crn.pdb")
    allatoms = newmol.chains.residues.atoms
    global coords
    coords = allatoms.coords
    global rad
    rad = newmol[0].defaultRadii(united=1)

def tearDownSuite():
    pass
    
    
## def test_01SurfaceNoArguments():
##     print "___________________"
##     print "test_01SurfaceNoArguments"
##     print "___________________"
##     from mslib import MSMS, msms
##     m = MSMS()

def test_02SurfaceCoordsRadii():
    print "___________________"
    print "test_02SurfaceCoordsRadii"
    print "___________________"
    from mslib import MSMS
    m = MSMS(coords=coords, radii=rad)

def test_03SurfaceFromFile():
    print "___________________"
    print "test_03SurfacefroFile"
    print "___________________"
    from mslib import MSMS
    #print "current dir:", os.getcwd() 
    m = MSMS(filename='Data/1crn.xyzrn')

def test_04RScalculations():
    print "___________________"
    print "test_04RScalculations"
    print "___________________"
    from mslib import MSMS, readxyzr, msms
    # compute all components with default probe radius 1.5
    global coords, rad
    m1 = MSMS(coords=coords, radii=rad)
    m1.compute_rs()
    ar = m1.rsr.fst.far0
    print "printing -n, -ct..."
    print ar._n()
    print ar._ct()
    print ar._cp()
    print ar._cc()
    print ar._rcc()
    print ar._ps()
    print ar._s()
    # recompute reduced surface with probe sphere radius = 2.0
    m1.compute_rs(probe_radius=2.0)
    # Multiple components
    output = readxyzr('Data/1crn.xyzr')
    new_coords = output[0]
    names = output[1]
    #new_coords, names = readxyzr('Data/1crn.xyzr')
    coords = new_coords[:,:3]
    rad = new_coords[:,3]

    m2 = MSMS(coords=coords, radii = rad)
    # "***** COMPUTE EXTERNAL COMPONENT"
    m2.compute_rs()
    print "Info1:"
    print m2.info()
    
    # "      ADD OTHER COMPONENTS"
    m2.compute_rs( allComponents=1 )
    print "Info2:"
    print m2.info()

    # "      CLEAR RSR"
    # "      COMPUTE ONE CAVITY"
    msms.MS_reset_RSR(m2.this)
    m2.compute_rs( atoms = (11, -1, -1) )
    print "Info3:"
    print m2.info()

    msms.MS_reset_RSR(m2.this)
    m2.compute_rs( atoms = (13, -1, -1) )
    print "Info4:"
    print m2.info()

    msms.MS_reset_RSR(m2.this)
    m2.compute_rs( atoms = (93, -1, -1) )
    print "Info5:"
    print m2.info()

    msms.MS_reset_RSR(m2.this)
    m2.compute_rs( atoms = (11, 13, -1) )
    print "Info6:"
    print m2.info()

    msms.MS_reset_RSR(m2.this)
    m2.compute_rs( atoms = (13, 11, -1) )
    print "Info7:"
    print m2.info()

    msms.MS_reset_RSR(m2.this)
    m2.compute_rs( atoms = (13, 93, -1) )
    print "Info8:"
    print m2.info()

    msms.MS_reset_RSR(m2.this)
    m2.compute_rs( atoms = (13, 93, 11) )
    print "Info9:"
    print m2.info()

def test_05SEScalculations():
    print "___________________"
    print "test_05SEScalculations"
    print "___________________"
    from mslib import MSMS, msms
    m = MSMS(coords=coords, radii = rad)
    m.compute_rs( allComponents=1 )
    m.compute_ses()
    print m.info()

    msms.MS_reset_RSR(m.this)
    m.compute_rs( allComponents=1 )
    m.compute_ses( 0 )
    m.compute_ses( 1 )
    print m.info()

def test_06Triangulation():
    print "___________________"
    print "test_06Triangulation"
    print "___________________"
    from mslib import MSMS, msms
    # triangulate with default parameters, component = 0

    m = MSMS(coords=coords, radii = rad)
    m.compute_rs(allComponents=1)
    m.compute_ses(0 )
    m.compute_ses(1)
    print "trinagulate 1"
    m.triangulate( component=0 )
    print m.info()
    # triangulate with default parameters, component = 1
    print "trinagulate 2"
    m.triangulate( component=1 )
    print m.info()

    # re-triangulate at density = 3.0
    print "trinagulate 3"
    m.triangulate( component=0, density=3.0 )
    print m.info()

    # triangulate all components
    print "trinagulate 4"
    m.triangulate( density=1.1 )

    # write triangulation
    #m.write_triangulation('testAll')
    #m.write_triangulation('testAllnoh', no_header=1)
    #m.write_triangulation('testASES', component=1, format=msms.MS_ASES_ASCII)
    #m.write_triangulation('testTAVS', component=0, format=msms.MS_TSES_ASCII_AVS)
    #m.write_triangulation('testAAVS', component=0, format=msms.MS_ASES_ASCII_AVS)
    vfloat, vint, tri = m.getTriangles()
    print "getTriangles(): len(vfloat): %d, len(vint): %d, len(tri): %d" % (len(vfloat), len(vint), len(tri))
    atomindices = [131, 120, 143, 147, 128, 146, 142, 137, 139, 130, 135,
                   136, 134, 129, 145, 119, 133, 132, 141, 144, 138, 140]
    vfloat, vint, tri = m.getTriangles(atomIndices=atomindices)
    print "getTriangles(atomIndices=atomindices): len(vfloat): %d, len(vint): %d, len(tri): %d" % (len(vfloat), len(vint), len(tri))
    # reset all after triangulation
    msms.MS_reset_RSR(m.this)
    print m.info()

def test_07SurfaceAreaCalculations() :
    print "____________________________"
    print "test_07SurfaceAreaCalculations"
    print "____________________________"
    from mslib import MSMS, msms
    m = MSMS(filename='Data/1crn.xyzrn')
    m.compute()
    m.compute_ses_area()
    
    print m.sesr.fst.a_reent_area
    print m.sesr.fst.a_toric_area
    print m.sesr.fst.a_contact_area
    print m.sesr.fst.a_ses_area
    print m.sesr.fst.a_sas_area

    #m.write_ses_area(filename='testAllComp.area', component=None)

    #m.write_ses_area(filename='testAllComp.area', component=0)

    m.compute_numeric_area_vol()
    m.compute_numeric_area_vol(component=0)

    m.compute_numeric_area_vol(component=0, mode=msms.MS_NUMERICAL)
    print m.sesr.fst.n_ses_volume

    m.compute_numeric_area_vol(component=0, mode=msms.MS_SEMI_ANALYTICAL)
    print m.sesr.fst.n_ses_volume

    print m.sesr.fst.n_sas_volume
    print m.sesr.fst.n_ses_area
    print m.sesr.fst.n_sas_area
    print m.detailed_info()
    print m.info()
    vfloat, vint, tri = m.getTriangles()
    print "triangles", len(vfloat), len(vint), len(tri)

def test_08BuriedSurfaceCalculations():
    print "______________________________"
    print "test_08BuriedSurfaceCalculations"
    print "______________________________"
    from mslib import MSMS, readxyzr, msms

    output1 = readxyzr('Data/1tpa_e.xyzr')
    coords_e=output1[0]
    names = output1[1]

    #coords_e, names = readxyzr('Data/1tpa_e.xyzr')
    output2 = readxyzr('Data/1tpa_i.xyzr')
    coords_i=output2[0]
    names = output2[1]
    #coords_i, names = readxyzr('Data/1tpa_i.xyzr')

    coords = coords_e[:,:3]
    rad = coords_e[:,3]
    m = MSMS(coords=coords, radii = rad)
    m.compute()
    m.buriedVertices(coords_i)
    vfloat, vint, tri = m.getTriangles()
    indBuried = Numeric.nonzero(vint[:,2])
    print len(indBuried)," vertices buried"

    m.resetBuriedVertexArea()
    m.compute_numeric_area_vol(component=0, mode=msms.MS_SEMI_ANALYTICAL)
    print m.sesr.fst.n_ses_volume
    print m.sesr.fst.n_ses_area
    m.buriedSurfaceArea()

    print m.sesr.fst.n_buried_ses_area
    print m.sesr.fst.n_buried_sas_area

    vfloat, vint, tri = m.getBuriedSurfaceTriangles(selnum=1)
    print "getBuriedSurfaceTriangles(selnum=1): len(vfloat): %d, len(vint): %d, len(tri): %d" % (len(vfloat), len(vint), len(tri))
    vfloat, vint, tri = m.getBuriedSurfaceTriangles(selnum=2)
    print "getBuriedSurfaceTriangles(selnum=2): len(vfloat): %d, len(vint): %d, len(tri): %d" % (len(vfloat), len(vint), len(tri))
    vfloat, vint, tri = m.getBuriedSurfaceTriangles(selnum=3)
    print "getBuriedSurfaceTriangles(selnum=3): len(vfloat): %d, len(vint): %d, len(tri): %d" % (len(vfloat), len(vint), len(tri))


##  harness = testplus.TestHarness( __name__,
##                                  connect = importLib,
##                                  funs = testplus.testcollect( globals()),
##                                  )

if __name__ == '__main__':
    testplus.chdir()
    args = ()
    harness = testplus.TestHarness( __name__,
                                    connect = (setUpSuite, args, {}),
                                    funs = testplus.testcollect( globals()),
                                    disconnect = tearDownSuite
                                    )
    print harness
    sys.exit( len( harness))
