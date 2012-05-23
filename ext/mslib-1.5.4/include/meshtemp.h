/* $Header: /opt/cvs/mslibDIST/include/meshtemp.h,v 1.2 2006/04/21 21:11:12 sanner Exp $
 *
 * $Id: meshtemp.h,v 1.2 2006/04/21 21:11:12 sanner Exp $
 *
 * $Log: meshtemp.h,v $
 * Revision 1.2  2006/04/21 21:11:12  sanner
 * - moved to lib1.4, updated the include files, the wrapper and the __init__.py
 *
 * Revision 1.1  2003/07/02 18:07:07  sanner
 * - added files in src/lib and src/tools
 *
 * Revision 1.1.1.1  2002/03/29 19:43:17  sanner
 * mslib1.3 initial checking
 *
 * Revision 0.1  2000/03/08 18:56:23  sanner
 * *** empty log message ***
 *
 * Revision 0.0  1999/10/27 23:02:11  sanner
 * *** empty log message ***
 *
 * Revision 0.1  1998/03/19  18:35:02  sanner
 * *** empty log message ***
 *
 */
#ifndef MESHTEMP
#define MESHTEMP

#define MxTmpF 10

struct MDOT__ {
    float x[3];
    int   v[MxTmpF];
    int   f[MxTmpF];
    int   at;		 /* atom to which this dot belongs */
    int   rn;		 /* relative number of dot */
    unsigned char nbv;
    unsigned char nbf;
    unsigned char cour;  /* numero de la couronne a laquelle il appartient */
    unsigned char mdfl;
};
typedef struct MDOT__ MDOT;

typedef struct {
    int s[2];
    int f[2];
    unsigned char marfl;
} MAR;

typedef struct {
    int s[3];
    int v[3];
    int a[3];
    float n[3];
    unsigned char mffl;
    unsigned char nbv;
    unsigned char nba;
} MFAC;


struct TM__ {  /* triangulated sphere template */
    MDOT *d;
    MFAC *f;
    MAR  *a;
    float r;
    float dm;  /* distance moyenne */
    float density;
    float real_density;
    int nbf;
    int nbd;
    int mxf;
    int nba;
};
typedef struct TM__ TM;

#endif
