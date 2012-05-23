## Automatically adapted for numpy.oldnumeric Jul 30, 2007 by 

#########################################################################
#
# Date: June 2003 Authors: Sophie Coon, Michel Sanner
#
#    sanner@scripps.edu
#    sophiec@scripps.edu
#
# Copyright: TSRI, Michel Sanner and Sophie Coon
#
#########################################################################

# $Header: /opt/cvs/mslibDIST/mslib/msmsParser.py,v 1.3 2007/10/03 18:39:50 vareille Exp $
#
# $Id: msmsParser.py,v 1.3 2007/10/03 18:39:50 vareille Exp $
#
import os
import numpy.oldnumeric as Numeric
class MSMSParser:
    """
    This class will parse a .vert file and a .face file.
    This class needs to move into mslib
    """
    def parse(self, vertFilename, faceFilename):
        assert os.path.exists(vertFilename)
        assert os.path.exists(faceFilename)
        vertf = open(vertFilename)
        vertDescr = vertf.readlines()
        vertf.close()
        facef = open(faceFilename)
        faceDescr = facef.readlines()
        facef.close()
        self.getFaces(faceDescr)
        self.getVert(vertDescr)
        return self.vertices, self.faces


    def getFaces(self, faceDescr):
        ################################################################
        # This function parses the face description and creates an array
        # of faces.
        # If the first line starts by a # then the 3 following are comments
        # lines
        # Comment #1: Comment + filename of the sphere set
        # Comment #2: Comment on content of comment #3
        # Comment #3: Nb triangles, nb of spheres, triangulation density,
        #             probe sphere radius.
        # vertInd1, vertInd2, vertInd3, Flag, FaceInd
        #
        # All the indices are 1 based !!!
        ################################################################
        # faceComments are all the lines with a # in front and the line after
        if faceDescr[0][0]=='#':
            facesComments = faceDescr[:3]
            facesLines = faceDescr[3:]
        else:
            facesLines = faceDescr
            facesComments = []
        facesInfo = map(lambda x: x.split(), facesLines)
        self.faces = Numeric.array(map(lambda x: [int(x[0])-1, int(x[1])-1,
                                                  int(x[2])-1],
                                       facesInfo)).astype('i')
        self.facesProp = Numeric.array(
            map(lambda x: [int(x[3]), int(x[4])], facesInfo)).astype('i')
        
    def getVert(self, vertDescr):
        ################################################################
        # This function parses the vertices description and creates an array
        # of vertices.
        # Comment #1: Comment + filename of the sphere set
        # Comment #2: Comment on content of comment #3
        # Comment #3: Nb triangles, nb of spheres, triangulation density,
        #             probe sphere radius.
        # x y z nx ny nz fInd sphInd flag AtmName
        #
        # All the indices are 1 based !!!
        ################################################################
        if vertDescr[0][0]=='#':
            vertComments = vertDescr[:3]
            vertLines = vertDescr[3:]
        else:
            vertComments = []
            vertLines = vertDescr
        vertInfo = map(lambda x: x.split(), vertLines)
        self.vertices = Numeric.array(
            map(lambda x: [float(x[0]), float(x[1]), float(x[2])],
                vertInfo)).astype('f')
        self.normals = Numeric.array(map(
            lambda x: [float(x[3]), float(x[4]), float(x[5])], vertInfo)
                                     ).astype('f')
        self.vertProp = Numeric.array(map(
            lambda x: [int(x[6]), int(x[7]), int(x[8])], vertInfo)).astype('i')
        
        
        


def testMSMSParser():
    from Pmv.msmsParser import MSMSParser
    msmsParser = MSMSParser()
    msmsParser.parse("./gc2.vert", "./gc2.face")

    msmsParser.parse("./test_0.vert", "./test_0.face")
    from DejaVu import Viewer
    vi = Viewer()

    from DejaVu.IndexedPolygons import IndexedPolygons
    visrf = IndexedPolygons('test', protected=True)
    visrf.Set(vertices=msmsParser.vertices,
              faces=msmsParser.faces,
              vnormals=msmsParser.normals, tagModified=False)
    
    vi.AddObject(visrf)
