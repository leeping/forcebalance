/*"MSMS library Module" */
%module msms
%feature ("kwargs"); /* enables python keyword arguments swig1.3.20 */
                     
%include numarr.i
%include Misc.i          /*typemaps for char ** */
#include <stdio.h>

#ifdef _MSC_VER
#include <windows.h>
#define WinVerMajor() LOBYTE(LOWORD(GetVersion()))
#endif

%typemap(in) float *coords (PyArrayObject *array)

 {
  array = NULL;
  if ($input == Py_None) {
    $1 = NULL;
  } else {
    array = contiguous_typed_array($input, PyArray_FLOAT, 2, NULL,$argnum);
    if (! array) return NULL;
    $1 = (float *)array->data;
  }
}
%typemap(freearg) float *coords
%{
   if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
%}



/************ Description ************************************************

This module wraps the fast molecular surface calculation library developped 
by Michel Sanner at the Molecular Graphics lab. (TSRI)

The basic functionnality of this library is to create compute solvent excluded
surfaces of molecules. The input is a set of spheres for which a 3D position
can be specified (x,y,z) along with a radius (r) as well as an optional name
(n). In a first step, a probe sphere representing a solvent molecule is rolled
over the set of spheres, defining the Reduced Surface (RS) (see reference 
below). The radius of the probe sphere is a user specified parameter. From 
the reduced surface the analytical model of the Solvent Excluded Surface (SES)
is computed. This analytical surface can be triangulated with a user specified
vertex density.
In addition, this library allows to:
    - compute and report surface areas that can be computed analytically or 
      numerically. 
    - compute the surface patches buried by a second set of spheres
    - upadte the surface after a subset of atoms assume new positions or radii
    - add/remove ?  atoms to the molecule and recompute the surface partially

example:

      import msms

References:

    - Sanner, M.F., Spehner, J.-C., and Olson, A.J. (1996) Reduced surface: 
      an efficient way to compute molecular surfaces. Biopolymers, Vol. 38, 
      (3), 305-320.
    - Michel F. Sanner and Arthur J. Olson. (1997) Real Time Surface 
      Reconstruction For moving Molecular Fragments. Proc. Second Pacific 
      Symposium in Biocomputing.
************************************************************************/

/*------------------------------------------------------------------------
 *
 * contour.h - contour library include file
 *
 * Copyright (c) 1999 Emilio Camahort
 *
 *----------------------------------------------------------------------*/

/* $Id: msms.i,v 1.10 2008/05/23 21:19:05 vareille Exp $ */

%{
#ifndef _WIN32
#include <unistd.h>
#endif
#include "mslib.h"
#include "msmsAllProto.h"
static MOLSRF_ROOT *msr;
static char buf[10000];
%}

%typemap (arginit) MOLSRF *ms, SES *su,  SES *ses, int c, int i, double OUT_VECTOR[3], double OUT_VECTOR[2], double OUT_VECTOR[6], double OUT_ARRAY2D[2][3], float OUT_VECTOR[3], int at1, char *name  %{
	if (MS_setjmp()) {
            PyErr_SetString(PyExc_RuntimeError, MS_err_msg);
            return NULL;
          }
%}

%init %{
  import_array(); /* load the Numeric PyCObjects */
  MS_init_msmslib();
  msr = MS_init_molsrf_root();
%}
%native(MS_get_triangles) extern PyObject *_wrap_MS_get_triangles (PyObject *self, PyObject *args, PyObject *kwargs);

/*"Constants"*/
/*------------------------------------------------------------------------
 * 
 *   constant definitions
 * 
 *----------------------------------------------------------------------*/

/*"reduced surface output formats"*/
#define MS_RS_SIMPLE_ASCII1 0
#define MS_RS_SIMPLE_ASCII2 1
#define MS_RS_SIMPLE_ASCII1_NORMALS 2

/*"restart modes" */
#define MS_NON_PARTIAL 0
#define MS_PARTIAL 1

/*"triangulation surface output formats"*/
#define MS_TSES_ASCII 0   /* whole triangulated surface */
#define MS_ASES_ASCII 1   /* discrete representation of the Analytical surf. */
#define MS_TSES_ASCII_AVS 10  /* same as MS_TSES_ASCII and AVS headers */
#define MS_ASES_ASCII_AVS 11  /* same as MS_ASES_ASCII and AVS headers */

/*"buried surface computation modes"*/
#define MS_NUMERICAL 1         /* numerical calculation using flat triangles */
#define MS_SEMI_ANALYTICAL 2   /* numerical calculation using spherical tri. */
#define MS_BOTH 3

/*"SESF buried flag"*/
#define MS_NOT_BURIED 0
#define MS_BURIED 1
#define MS_PARTIALLY_BURIED 2

/*"analytical SES edges types"*/
#define SAILLANT 1
#define RENTRANT 2
#define TORIQUE 3
#define SINGULIER 4
#define SINGULIER_TRAITE 8
#define SELECT 16
#define FULL_EATEN 32
#define RENT_FUS 64

/*"toroidal face types"*/
#define FULL_TORUS1 4  /* SES face type for full torus or first cone */
#define FULL_TORUS2 5  /* SES face type for second cone of full torus */

/*"error handling stuff"*/
#define MS_OK 0
#define MS_ERR 1
#define MS_ErrMsgMaxLen 512

/*"surface update triangulation modes"*/
#define FULL 0   /* the surface is rebuilt complete */
#define TORIC 1  /* only toric faces are triangulated */
#define ALL_WITH_DENSITY 2 /* all rebuilt faces are triangulated with the given density */
#define NO_TRIANGULATION 3   /* no triangulation at all */

/*"GLOBALS"*/
char *MS_VERSION;
int MS_err_flag;
char MS_err_msg[MS_ErrMsgMaxLen];


/*"Data Structures"
"Reduced Surface Data Structures"*/

/*------------------------------------------------------------------------
 * 
 *   data structures
 * 
 *----------------------------------------------------------------------*/

/*------------------------------------------------------------------------
 *   data structures for Reduced Surface
 *----------------------------------------------------------------------*/

%immutable;
/* RS edges */
%name (RSE) struct RSE__ {
    struct RSE__  *nxt;
    struct RSE__  *prv;
    struct RSF__  *f[2];
    struct SESF__ *rsedual[2]; /* pointeur vers la ou les faces toric dual */
    struct RSE__  *nxtUp;    /* chain of deleted or reconstructed edges */
    void   *data;           /* structure qcq */
    double n[3];	    /* vecteur unitaire de l'arete */
    double ct[3];	    /* centre du tore */
    double rt;		    /* rayon du tore */
    double cp[3];	    /* centre de la sphere temoin (RS  free edge) */
    double angt;	    /* ang entre z1,ct,z2 ou z1 et z2 sont les
			      positions initiale et finale de la sph temoin */
    double cc[2][3];	    /* centre des cercles de contact avec le premier
			       atome de l'arete */
    double rcc[2];	    /* rayon des cercles de contact */
    float  ps[2][3];	    /* coord. of singular points */
    int    s[2];	    /* indice des atomes sommets de l'arete */
    int	   num;		    /* numero de l'arete */
    unsigned char sing;	    /* # of singular points on each edge */
    unsigned char comp_num; /* numero de la composante connexe */
    unsigned char type;     /* FREE or REGULAR */
};
typedef struct RSE__ RSE;

%extend RSE__ {
  RSF  * _rsf0() { return self->f[0]; }
  RSF  * _rsf1() { return self->f[1]; }
  SESF * _rsedual0() { return self->rsedual[0]; }
  SESF * _rsedual1() { return self->rsedual[1]; }
  void _n( double OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->n[0];
    OUT_VECTOR[1] = self->n[1];
    OUT_VECTOR[2] = self->n[2];
  }
  void _ct( double OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->ct[0];
    OUT_VECTOR[1] = self->ct[1];
    OUT_VECTOR[2] = self->ct[2];
  }
  void _cp( double OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->cp[0];
    OUT_VECTOR[1] = self->cp[1];
    OUT_VECTOR[2] = self->cp[2];
  }
  void _cc( double OUT_ARRAY2D[2][3] ) {
/*
    OUT_ARRAY2D[0] = self->cc[0][0];
    OUT_ARRAY2D[1] = self->cc[0][1];
    OUT_ARRAY2D[2] = self->cc[0][2];
    OUT_ARRAY2D[3] = self->cc[1][0];
    OUT_ARRAY2D[4] = self->cc[1][1];
    OUT_ARRAY2D[5] = self->cc[1][2];
*/
    OUT_ARRAY2D[0][0] = self->cc[0][0];
    OUT_ARRAY2D[0][1] = self->cc[0][1];
    OUT_ARRAY2D[0][2] = self->cc[0][2];
    OUT_ARRAY2D[1][0] = self->cc[1][0];
    OUT_ARRAY2D[1][1] = self->cc[1][1];
    OUT_ARRAY2D[1][2] = self->cc[1][2]; 
  }
  void _rcc( double OUT_VECTOR[2] ) {
    OUT_VECTOR[0] = self->rcc[0];
    OUT_VECTOR[1] = self->rcc[1];
  }
  void _ps( double OUT_ARRAY2D[2][3] ) {
/*
    OUT_ARRAY2D[0] = self->ps[0][0];
    OUT_ARRAY2D[1] = self->ps[0][1];
    OUT_ARRAY2D[2] = self->ps[0][2];
    OUT_ARRAY2D[3] = self->ps[1][0];
    OUT_ARRAY2D[4] = self->ps[1][1];
    OUT_ARRAY2D[5] = self->ps[1][2];
*/
    OUT_ARRAY2D[0][0] = self->ps[0][0];
    OUT_ARRAY2D[0][1] = self->ps[0][1];
    OUT_ARRAY2D[0][2] = self->ps[0][2];
    OUT_ARRAY2D[1][0] = self->ps[1][0];
    OUT_ARRAY2D[1][1] = self->ps[1][1];
    OUT_ARRAY2D[1][2] = self->ps[1][2];
  }
  void _s( int OUT_VECTOR[2] ) {
    OUT_VECTOR[0] = self->s[0];
    OUT_VECTOR[1] = self->s[1];
  }
}

/* RS vertex */
%name (RSV) struct RSV__ {
    struct RSE__ **a;
    struct RS__  *rs;
    struct SESF__ **rsvdual;        /* array of pointer to contact faces */ 
    void   *data;
    double *ses_area;
    double *sas_area;
    float  x[3];
    float  r;                /* current atom radius, may change for restart */
    float  orig_r;           /* original atom radius */
    int    bhtind;	     /* index of atm in rsr->atmBHT array of BHPoints*/
    int	   num;		     /* numero de l'atome correspondant au sommet */
    int	   nxt;		     /* numero du sommet prochain sommet duplique */
    int    nxtup;            /* next updated RS vertex */
    short  type;	     /* type de l'atome, sert a trouver la sphere
				triangulee */
    short  mnba;	     /* nombre d'aretes allouees */
    short  nba;		     /* nombre d'aretes incidentes */
    short  nbf;		     /* nombre de faces de contact pour cet atome */
    unsigned char pb;        /* used to mark atoms in pb faces */
    unsigned char surf;	     /* = 1 surface is computed for this atom
			        = 0 atom is used for collision only
			        = 2 if it has been found that this atom
			          cannot create surface
				= 4 this atom has changed position or radius
				= 8 this atom has a new radius because of a
				  restart during an update */
    unsigned char comp_num; /* numero de la composante connexe */
    unsigned char free_edge_checked; /* =1 when has been checked for free edge */
    unsigned char hdensity;
    unsigned char split;     /* =1 when has been checked for free edge */
    char *name;              /* atom name */
};
typedef struct RSV__ RSV;

%extend RSV__ {
  RSE * get_a(int i) {
    if (i>=0 && i<self->nba) return self->a[i];
    else return NLRSE;
  }
  SESF *get_rsvdual(int i) {
    if (i>=0 && i<self->nbf) return self->rsvdual[i];
    else return NLSESF;
  }
  void _x( double OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->x[0];
    OUT_VECTOR[1] = self->x[1];
    OUT_VECTOR[2] = self->x[2];
  }
  double get_ses_area(int i) {
    return self->ses_area[i];
  }
  double get_sas_area(int i) {
    return self->sas_area[i];
  }
}

/* RS face */
%name(RSF) struct RSF__ {
    struct RSF__ *nxt;
    struct RSF__ *prv;
    struct RSE__ *a[3];	    /* list des aretes de la face */
    struct SESF__ *rsfdual[2]; /* pointeur vers 1 ou 2 faces de SES (2 si
				  singuliere coupee) de SES */
    struct RSF__  *fs  ;    /* pointeur vers la face qui cree une lentille
			       entiere ou partielle */
    struct RSF__ *nxtUp;    /* chain of deleted or reconstructed face */
    struct RSF__ *nxtbUp;   /* chain of RSF forming the boundary of the hole
			       created when RSF are deleted during update */
    void *data;

    double v[3];	    /* vect normal oriente exterieur */
    double c[3];	    /* centre of the probe sphere */

    int	   s[3];	    /* indexe des sommets dans le tableau atm */
    int    num;		    /* face number for debugging purpose */
    int    uind;	    /* index of RSF in rsr->uprbBHP array of BHPoints*/
    int    sind;	    /* index of RSF in rsr->sprbBHP array of BHPoints*/
    unsigned char comp_num; /* numero de la composante connexe */
    unsigned char cp;	    /* =1 si la sph. temoin coupe le plan de la face */
    unsigned char mfl;	    /* utilise pour marquer les cotes traites 1,2,4
			       8 ==> bad face (atom moved or changed rad)
			       16 ==> set when close to moving atoms 
			       32, 64, 128 used by walk_over_rs */
    unsigned char dir[3];   /* 1 quand l'arete est parcourue positivement
			       0 sinon */
};
typedef struct RSF__ RSF;

%extend RSF__ {
  RSE *_a0() { return self->a[0]; }
  RSE *_a1() { return self->a[1]; }
  RSE *_a2() { return self->a[2]; }
  SESF *_rsfdual0() { return self->rsfdual[0]; }
  SESF *_rsfdual1() { return self->rsfdual[1]; }
  void _v( double OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->v[0];
    OUT_VECTOR[1] = self->v[1];
    OUT_VECTOR[2] = self->v[2];
  }
  void _c( double OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->c[0];
    OUT_VECTOR[1] = self->c[1];
    OUT_VECTOR[2] = self->c[2];
  }
  void _s( int OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->s[0];
    OUT_VECTOR[1] = self->s[1];
    OUT_VECTOR[2] = self->s[2];
  }
  void _dir( int OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = (int)self->dir[0];
    OUT_VECTOR[1] = (int)self->dir[1];
    OUT_VECTOR[2] = (int)self->dir[2];
  }
}

/* RS header */
%name(RS) struct RS__ {
  struct RS__ *nxt;
  struct RS__ *prv;
  struct SES__ *ses;
  struct RSF__ *ffa;
  struct RSF__ *lfa;
  struct RSF__ *lorsf;    /* last kept RSF during update */
  struct RSE__ *far0;
  struct RSE__ *lar;
  struct RSE__ *ffar;
  struct RSE__ *lfar;
%mutable;
  struct RSF__ *fstrfup;  /* used to update surface: fst deleted RSF */
  struct RSF__ *lstrfup; /* last RSF to be removed in list built for update */

  struct RSE__ *fstreup;  /* used to update surface: fst deleted RSE */
  struct RSE__ *lstreup;  /* last RSE to be removed in list built for update */

  struct RSF__ *fstRSbfup; /* list of RSF bounding the holes after delete */
  struct RSF__ *lstRSbfup; /* last item in list  */

  struct SESF__ *fstfup;  /* list of SESF to be deleted */
  struct SESF__ *lstfup;  /* last entry in list */

  struct SESE__ *fsteup;  /* list of SESE to look at during update */
  struct SESE__ *lsteup;  /* last entry in list */
%immutable;
  TRBHTree *uprbRBHT; /* growable and movable BHtree for all probes concerned
			 by the update */
  TRBHTree *sprbRBHT; /* dynamic size, movable potentially singular probes
			 BHtree, used surf.c to handle singularities */
  void *data;
  int nbfcp;
  int nbfco;  /* nombre de face de RS collees avec plus d'un cote coupe */
  int nbf;
  int nba;
  int nbaf;
  int nbs;
  int num;
  int ffat[3];        /* atom indexes for first RS face */
  int needFreeEdgeCheck;
  unsigned char split_rsv;
};
typedef struct RS__ RS;

%extend RS__ {
  void _dir( int OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->ffat[0];
    OUT_VECTOR[1] = self->ffat[1];
    OUT_VECTOR[2] = self->ffat[2];
  }  
}

/* RS tree root */
%name(RSR) struct RSR__ {
%mutable;
  struct RS__ *fst;
  struct RS__ *lst;
%immutable;
  struct RSV__ *atm;
  TRBHTree *atmBHT;   /* growable and movable atoms BHtree */
%mutable;
  int nb;           /* number of RS components */
  int nb_free_vert; /* number of free vertices in the Reduced Surface */
  int ffnba;        /* used to compute only one component */
%immutable;
  int ffat[3];      /* atom indexes for first RS face */
%mutable;
  short all_comp_done;
  short ext_comp_done;
%immutable;
};
typedef struct RSR__ RSR;

%extend RSR__ {
  void set_ffat(int at1, int at2, int at3) {
    self->ffat[0] = at1;
    self->ffat[1] = at2;
    self->ffat[2] = at3;    
  }
  void _ffat( int OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->ffat[0];
    OUT_VECTOR[1] = self->ffat[1];
    OUT_VECTOR[2] = self->ffat[2];
  }
}

/*"Triangulation"*/
/*------------------------------------------------------------------------
 *   data structures for triangulation
 *----------------------------------------------------------------------*/

%name(TRIV) struct TRIV__ {
  float c[3];	  /* coord. de ce sommet */
  float n[3];	  /* vecteur normal a la surface */
  float uvw[3];   /* array of texture uvw's */
  float sesArea;  /* per vertex SES surface area */
  float sasArea;  /* per vertex SAS surface area */
  int   atm;	  /* # de l'atome auquel appartient ce sommet */
  int   tvn;	  /* # du sommet affichage et debuging */
  unsigned char fl; /* bits 1,2,4,8 used in triangulation,
		       32 and 64 buried vertex, 128 selection bit */
};
typedef struct TRIV__ TRIV;

%extend TRIV__ {
  void _c( float OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->c[0];
    OUT_VECTOR[1] = self->c[1];
    OUT_VECTOR[2] = self->c[2];
  }
  void _n( float OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->n[0];
    OUT_VECTOR[1] = self->n[1];
    OUT_VECTOR[2] = self->n[2];
  }
  void _uvw( float OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->uvw[0];
    OUT_VECTOR[1] = self->uvw[1];
    OUT_VECTOR[2] = self->uvw[2];
  }
}

%name(TRI) struct TRI__ {  /* triangular face */
  struct TRIV__ *s[3];     /* vertices */
};
typedef struct TRI__ TRI;

%extend TRI__ {
 TRIV *_s0() { return self->s[0]; }
 TRIV *_s1() { return self->s[1]; }
 TRIV *_s2() { return self->s[3]; }
}

/*------------------------------------------------------------------------
 *   data structures for Analytic Solvent Excluded Surface
 *----------------------------------------------------------------------*/
/*"Solvent Excluded Surface Data Structures"*/

%name(SESV) struct SESV__ {           /* Solvent Excluded Surface Vertex */
    struct SESV__ *nxt;
    struct SESV__ *prv;
    struct SESV__ *id;          /* pointeur vers un autre sommet si duplique */
    struct SESE__ **a;          /* tableau de SESE */
    void   *data;
    float  *n;		        /* vecteur normal a la surface */
    float  c[3];		/* coord. */
    float  beta;		/* angle between edges of the reentrant face */
    int	   num;			/* numero d'ordre du sommet */
    int	   at;			/* numero de l'atome associe au sommet */
    short  nba;		        /* nba nombre de SESE */    
    short  mnba;		/* nba nombre de *SESE alloues */    
    struct TRIV__ *triv;        /* vertex de la surface triangulee */
    unsigned char type;		/* =0 pour un sommet de base */
				/* =1 pour un sommet singulier */
				/* =2 pour un sommet sur un cercle sing. */
};
typedef struct SESV__ SESV;

%extend SESV__ {
  SESE *get_a(int i) {
    if (i>=0 && i<self->nba) return self->a[i];
    else return NLSESE;
  }
  void _c( double OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->c[0];
    OUT_VECTOR[1] = self->c[1];
    OUT_VECTOR[2] = self->c[2];
  }
  void _n( double OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->n[0];
    OUT_VECTOR[1] = self->n[1];
    OUT_VECTOR[2] = self->n[2];
  }  
}

%name(SESE) struct SESE__ {           /* Solvent Excluded Surface Edge */
    struct SESE__  *nxt;
    struct SESE__  *prv;
    struct SESV__  *s[2];
    struct SESF__  *f[2];
    struct RSE__   *ar;
    struct TRIV__  **triv;  /* som. de la surface triangulee pour cette arete
		            triv[0] and triv[e->nbtriv-1] are the edge's end */
    struct SESE__  *nxtSESaUp;  /* list of SESE to look at during update */
    void   *data;
    double *n;	            /* vecteur unitaire ortho. au plan de l'arete */
    float  c[3];	    /* centre du cercle portant l'arete */
    float  r;		    /* rayon de ce cercle */
    float  theta;	    /* */
    float  free_cap_area;   /* */
    float  ang;      	    /* angle de l'arc s[0],c,s[1]*/
    int	   nbtriv;	    /* number of triangulation vertices internal to the
			       edge + 2 (for the ends) */
    int	   num;		    /* numero de l'arete */
    short  type;	    /* 1 : brin spherique saillant
			       2 : brin spherique rentrant
			       3 : brin interne a une face torique rentrante
			       4 : brin singulier
			      16 : traitee lors de la recherche des face
				   spheriques saillantes */
    unsigned char cont_num; /* number of the cycle the edge belongs to
			       the most external contour has number 0 */
};
typedef struct SESE__ SESE;

%extend SESE__ {
  SESV *_s0() { return self->s[0]; }
  SESV *_s1() { return self->s[1]; }
  SESF *_f0() { return self->f[0]; }
  SESF *_f1() { return self->f[1]; }
  TRIV * get_triv(int i) {
    if (i>=0 && i<self->nbtriv) return self->triv[i];
    else return NLTRIV;
  }
  void _n( double OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->n[0];
    OUT_VECTOR[1] = self->n[1];
    OUT_VECTOR[2] = self->n[2];
  }
  void _c( double OUT_VECTOR[3] ) {
    OUT_VECTOR[0] = self->c[0];
    OUT_VECTOR[1] = self->c[1];
    OUT_VECTOR[2] = self->c[2];
  }
}

%name(SESF) struct SESF__ {           /* Solvent Excluded Surface Face */
    struct SESF__  *nxt;
    struct SESF__  *prv;
    struct SESV__  **s;     /* pointeurs vers les sites */
    struct SESE__  **a;     /* aretes bordant cette face */
    struct SESF__  *nxtUp;  /* linked list of deleted SESF (Update Surface) */
    struct SESF__  *nxttorUp; /* linked list of toric SESF that need to be
			        looked at to find contact faces */
    struct SESF__  *nxttUp; /* linked list of SESF that need a new
			       triangulation(Update Surface) */
    void  *dual;	    /* pointeur vers l'element dual de SR */
    void  *data;
    struct TRI__   * tri;   /* array of triangular faces */
    struct TRIV__  **triv;  /* array of triangulation vertices (internal) */
    float  buried;          /* ratio of buried_vertices/not_buried vertices */
    float  a_ses_area;
    int    nbtri;           /* number of triangles */
    int    nbtriv;          /* number of internal triangulation vertices */
    int	   num;		    /* numero d'ordre de la face (debug) */
    short  nb;		    /* nombre d'aretes allouees */
    short  nba;	            /* nombre d'aretes */
    short  type;	    /*  1 : spherique saillante
				2 : spherique rentrante
				3 : torique rentrante
				4 : full torus 1
				5 : full torus 2
				6 : update_rs ==> has to be deleted */
    unsigned char pb;       /* set to 1 when problem sing. or tri
			       bit 2: used in remove_bad_faces to tag faces to
			              be removed for an update */

    unsigned char fl;       /* bit 1: set when face needs to have edges
			              re-ordered
                            */
    unsigned char cont_nb;  /* number of cycles forming the edge */
    unsigned char forceDot; /* set to 1 to force at least 1 dot in the edge */ 
    unsigned char *direct;  /* 1 quand l'arete est parcourue positivement
			       0 sinon */
    unsigned char nbtriv_iso;  /* number of isolated triangulation verts 
				  for that face */
};
typedef struct SESF__ SESF;

%extend SESF__ {
  SESE * get_a(int i) {
    if (i>=0 && i<self->nba) return self->a[i];
    else return NLSESE;
  }
  TRI * get_tri(int i) {
    if (i>=0 && i<self->nbtri) return &self->tri[i];
    else return NLTRI;
  }
  TRIV * get_triv(int i) {
    if (i>=0 && i<self->nbtriv) return self->triv[i];
    else return NLTRIV;
  }
  int get_direct(int i) {
    if (i>=0 && i<self->nbtriv) return (int)self->direct[i];
    else return -999;
  }
}

%name(SES) struct SES__ {
    struct SES__ *nxt;
    struct SES__ *prv;
  /* analytical surface description */
    struct SESF__  *ffa;    /* pointer to first face */
    struct SESF__  *lfa;    /* pointer to last face */
    struct SESE__  *far0;    /* pointer to first arete */
    struct SESE__  *lar;    /* pointer to last arete */
    struct SESV__  *fso;    /* pointer to first sommet */
    struct SESV__  *lso;    /* pointer to last sommet */
%mutable;
    struct SESF__  *lsesf;  /* last ses face kept during update */

    struct SESF__  *fsttfup; /* linked list of faces which need triangulat. */
    struct SESF__  *lsttfup; /* last lelement in list */

    struct SESF__ *fsttorup;  /* list of toric SESF to look at for finding */
    struct SESF__ *lsttorup;  /* contact faces during an update operation */

    struct SESE__  *lsese;  /* last ses edge kept during update */
    struct SESV__  *lsesv;  /* last ses vertex kept during update */
%immutable;
    void *data;
    int    nbf;		   /* number of faces */
    int    nba;		   /* number of aretes */
    int    nbs;		   /* number of sites */
    int    nbss;	   /* number of singular sites */
    int    nbse;	   /* number of singular edges */
    int    nbft[5];   /* used to count the number of contact faces of each
        type (0:contact 1:reentrant 2,3:toric). The number of contact face
	corresponds to the number of vertices in the reduced surface.
	This number can be larger than envred.nbs because of the multiplicity
	index of vertices in SR */
    int    num;
    int    holes_reent;
    int    holes_cont;
    int    nb_comp;     /* number of components in that SES surface
		  this can happen when we have a singular free RS edge */
    float  a_reent_area;
    float  a_toric_area;
    float  a_contact_area;
    float  a_ses_area;
    float  a_sas_area;
    float  a_ses_volume;
    float  a_sas_volume;
    float  a_buried_ses_area;
    float  a_buried_sas_area;

  /* triangulated surface */
    float  density,density_sq;
    float  hdensity,hdensity_sq;
    float  cercang;    /* angle between 2 dots on a cercle of radius 1.0 */
    float  hdcercang;  /* angle between 2 dots on a cercle of radius 1.0 */
    int    npcerc;     /* number of dots on a cercle of radius 1.0 */
    int    hdnpcerc;   /* number of dots on a cercle of radius 1.0 */
    int    nbtri1;     /* number of triangles in spheric reentrant faces */
    int    nbtri2;     /* number of triangles in toric reentrant faces */
    int    nbtri3;     /* number of triangles in contact faces */
    int    nbtri;      /* number of triangles */
    int    nbtriv;     /* number of triangulation vertices */
    int    nbtriv_iso; /* nombre de sommets n'appartenant a aucune arete */
    int    tri_cont;   /* remember if contact faces have already
				       been triangulated */
    int    n_area_mode; /* MS_NUMERICAL or MS_SEMI_ANALYTICAL */
    float  n_ses_area;
    float  n_sas_area;
    float  n_buried_ses_area;
    float  n_buried_sas_area;
    float  n_ses_volume;
    float  n_sas_volume;
    float  genusAna;
    float  genusTri;
};
typedef struct SES__ SES;


%extend SES__ {
  void _nbft( int OUT_VECTOR[6] ) {
    int i;
    for (i=0;i<5;i++) OUT_VECTOR[i] = self->nbft[i];
  }
}

%name(SESR) struct SESR__ {
%mutable;
  struct SES__ *fst;
  struct SES__ *lst;
  int nb;
%immutable;
};
typedef struct SESR__ SESR;

/*------------------------------------------------------------------------
 *   data structure for molecular surface
 *----------------------------------------------------------------------*/

/*"Molecular Surface"*/

typedef struct {
  char *name;
  RSR  *rsr;     /* pointer to the root of the RS tree */
  SESR *sesr;    /* pointer to the root of the SES tree */
  void *data;

  /* sphere set stuff */
  float maxr, minr;
  int   extreme[6]; /* indices of extreme atoms along x,y and z axis */
  int   nbat, nats, natc, maxat;
  char *filename;

  /* surface parameters and variables */
%mutable;
  float rp;
  float rp1;
  float density;
%immutable;

  /* triangulation stuff */
  struct TM__ **tmtab; /* pointers to template spheres */
  TDOT *dots;          /* array used to store translated template vertices 
		      during the triangulation of reentrant and contact SESF */
  int Maxnbt;          /* number of allocated pointers to templates spheres */
  int nbt;             /* number of templates spheres */
  int MaxDots;         /* largest number of vertices on template */
  int MaxEdges;        /* largest number of edges on template */
  int MaxTriangles;    /* largest number of triangles on template */
  int allocated_MaxDots;
  float MinDistSites2; /* used in reduced.c to decide if two singular points
			  are the same and in surf.c to decide if a point
			  is on a sphere or not */

  /* computation parameters */
%mutable;
  short all_components;
  short free_vert;
  short cusp_trim;
  short noh;
  short dotsOnly;      /* write only .vert file, no triangulation */
  short cleanup_rs;
  short cleanup_ses;

%immutable;
  /* update stuff */
  int fstup;               /* first updated RS vertex */
  int up_num;              /* current update number */
%mutable;
  int up_mode;             /* update_mode */
  float up_density;        /* update_density */
  unsigned char MS_mfl;
  unsigned char restart_mode; /* MS_PARTIAL (default) or MS_NON_PARTIAL */
%immutable;

  /* restart options */
%mutable;
  float mod_dist;    /* restart modification distance ??? every atom within
			rest_mod_dist of a modified atom is marked modified */
  short restart_ana; /* restart computation after analytical SES pb */
  short restart_tri; /* restart computation after triang. pb */

  short rest_on_pbr; /* restart after pb while triangulating a reent. face */
  short rest_on_pbc; /* restart after pb while triangulating a contact face */
  short max_try;     /* Max number of re-try after SES and TRI */
%immutable;
  short try_num;     /* current number of re-tries */
} MOLSRF;

%extend MOLSRF {
  void _extreme( int OUT_VECTOR[6] ) {
    int i;
    for (i=0;i<6;i++) OUT_VECTOR[i] = self->extreme[i];
  }
}


/*"Functions"*/
/*********************************************************************

    INTERFACE FUNCTIONS PROTOTYPE

**********************************************************************/

%extend MOLSRF {
  RSV * get_atm(int i) {
    if (i>=0 && i<self->nbat) return &self->rsr->atm[i];
    else return NLRSV;
  }
  %apply int VECTOR[ANY] { int surfflags[1] }
  %apply int VECTOR[ANY] { int hdflags[1] }

  MOLSRF(char *name="No Name", float *coords=NULL, int nat=0,
	 int maxat=0, char **names=NULL, int surfflags[1]=NULL, 
	 int hdflags[1]=NULL) {

      float **c=NULL;
      int i, j;
      MOLSRF *ms;

      for (i=0;i<nat;i++)
      if (coords==NULL) nat=0;
      if (nat>0) {
	c = (float **)malloc(nat * sizeof(float *));
	if (!c) return NULL;
	for(i = 0; i < nat; i++) {
	  c[i] = (float *)malloc(4*sizeof(float));
	  if(c[i]==NULL) {
	    for (j=0; j<i; j++) free(c[i]);
	    free(c);
	    return NULL;
	  }
	  for(j = 0; j < 4; j++) c[i][j] = coords[i*4+j];
 	  //printf("%d %d %f %f %f %f\n", i, hdflags[i], c[i][0], c[i][1],
 	  //	 c[i][2], c[i][3]);
	}
      }
      ms = MS_add_molsrf(msr, name);
      if (! ms){
	printf ("In MSMS: ms == NULL\n");
       }
      if (maxat < 0) maxat=nat;
      if ( MS_add_msms_atoms(ms, nat, maxat, c, names, surfflags, hdflags) == MS_ERR ) {
	MS_delete_molsrf( ms );
	return NULL;
      }
      
      if (nat>0) {
	for(i = 0; i < nat; i++)  free(c[i]);
	free(c);
      }
      return ms;
    }

    ~MOLSRF() { MS_delete_molsrf( self ); }

    int findBuriedVertices(SES *su,float *coords, int nat) { 
      float **c=NULL;
      int   i, j, stat;

      c = (float **)malloc(nat * sizeof(float *));
      if (!c) return NULL;
      for(i = 0; i < nat; i++) {
	c[i] = (float *)malloc(4*sizeof(float));
	if(c[i]==NULL) {
	  for (j=0; j<i; j++) free(c[i]);
	  free(c);
	  return NULL;
	}
	for(j = 0; j < 4; j++) c[i][j] = coords[i*4+j];
      }
      stat = MS_findBuriedVertices(self, su, c, nat);
      for(i = 0; i < nat; i++)  free(c[i]);
      free(c);
      return stat;
    }
}

extern void    MS_detailed_info_molsrf( MOLSRF *ms );
extern void    MS_info_molsrf( MOLSRF *ms );
/********
not wrapped because msr is not exposed
extern void    MS_list_molsrf( MOLSRF_ROOT *msr, int verbose );
extern MOLSRF *MS_find_molsrf_by_num( MOLSRF_ROOT *msr, int num );
extern MOLSRF *MS_find_molsrf_by_name( MOLSRF_ROOT *msr, char *name );
********/
/*not sure how to handle the void *   */
/*******
extern int     MS_set_molsrf_param( MOLSRF *ms, char *key, void *val );
*******/

/* atom coordinates */
%{
static PyObject *MS_readxyzr(PyObject *self, PyObject *args)
{
  char   *filename, **names;
  float **coords, *dptr, *surfflags;
  int     stat, nat, i, j, dims[2], *hdflags;
  PyArrayObject *array;
  PyObject *pynames, *pysurf, *pyhd;

  if (MS_setjmp()) {
       PyErr_SetString(PyExc_RuntimeError, MS_err_msg);
       return NULL;
     }

  if (!PyArg_ParseTuple(args, "s", &filename))
    return NULL;

  MS_reset_msms_err();
  stat = MS_read_xyzr( filename, &nat, &coords, &names, &surfflags, &hdflags);

  /* build numeric array with coordiantes */
  if(!coords || nat==0)
    {
      PyErr_SetString(PyExc_RuntimeError, MS_err_msg);
      return NULL;
    }

  dims[0] = nat;
  dims[1] = 4;
  array = (PyArrayObject *)PyArray_FromDims(2, dims, PyArray_FLOAT);

  if(!array)
    {
      PyErr_SetString(PyExc_RuntimeError, 
		      "Failed to allocate array for coordinates");
      return NULL;
    }

  dptr = (float *)array->data;
  for(i = 0; i < dims[0]; i++)
    {
      for(j = 0; j < dims[1]; j++)
	{
	  *dptr = coords[i][j];
	  dptr++;
	}
    }

  /* build list of names */
  pynames = PyList_New(0);
  pysurf = PyList_New(0);
  pyhd = PyList_New(0);

  for (i=0; i<nat; i++) {
    if (names[i]!=NULL) PyList_Append(pynames, PyString_FromString(names[i]));
    if (surfflags!=NULL) PyList_Append(pysurf, Py_BuildValue("i", surfflags[i]));
    if (hdflags!=NULL) PyList_Append(pyhd, Py_BuildValue("i", hdflags[i]));
/*     if (names[i]!=NULL) { */
/*       PyList_Append(pynames, PyString_FromString(names[i])); */
/*     } else { */
/*       PyList_Append(pynames, PyString_FromString("")); */
/*     } */
  }

  /* FIXME ... when I call free I resulting names are wrong */
  MS_free_coord(coords, nat, names, surfflags, hdflags);
  
  return Py_BuildValue("OOOO", array, pynames, pysurf, pyhd);
}
%}
%native (MS_readxyzr) MS_readxyzr;

/*not yet wrapped because of float ***  */
/*extern void    MS_free_coord(float **Coord,int nbat, char *names[]); */

extern void  MS_reset_atom_update_flag( MOLSRF *ms );
extern int   MS_get_xyzr_update( MOLSRF *ms, char *file, int max );
/*
%apply float VECTOR[1] { float *coords }
extern int MS_updateSpheres(MOLSRF *ms, int nb,int *indices,float *coords);
extern int MS_addSpheres(MOLSRF *ms, int nb, float *coords, char **names);
*/
%apply float ARRAY2D[ANY][ANY] { float coords[1][4] }
%apply int VECTOR[ANY] { int indices[1] }
extern int MS_updateSpheres(MOLSRF *ms, int nb,int indices[1],float coords[1][4]);
extern int MS_addSpheres(MOLSRF *ms, int nb, float coords[1], char **names);

/* reduced surface */
extern int MS_reduced_surface( MOLSRF *ms );
extern int MS_update_reduced_surface( MOLSRF *ms, RS *rs, int nup );
/* not really useful since I can "walk the pointers" in Python */
/* extern RS   *MS_find_rs_component_by_num( MOLSRF *ms, int i ); */
extern int MS_write_rs_component(MOLSRF *ms, RS *rs, char *name, int mode );
extern int   MS_writeSolventAccessibleatoms( MOLSRF *ms, char *name, int mode);
extern int MS_reset_RSR( MOLSRF *ms );
extern void  MS_free_RS_vertices(MOLSRF *ms);

/* analytical solvent excluded surface */
extern int  MS_solvent_excluded_surface( MOLSRF *ms, RS *rs);
extern int  MS_update_ses( MOLSRF *ms, RS *rs, int nup );
extern int  MS_compute_SES_area(MOLSRF *ms);
extern int  MS_update_SES_area( MOLSRF *ms, SES *su );
/* not really useful since I can "walk the pointers" in Python */
/*extern SES   *MS_find_SES_component_by_num( MOLSRF *ms, int i );*/
extern int  MS_write_surface_areas( MOLSRF *ms, char *f, int cpnum );
extern int  MS_set_SES_density( SES *ses, float density, float hd, float rp );
extern void MS_free_SES_comp( SES *su );
extern void MS_reset_SESR( MOLSRF *ms );


/* triangulated solvent excluded surface */
extern int MS_make_templates( MOLSRF *ms, SES *s );
extern int MS_triangulate_SES_component( MOLSRF *ms, RS *rs );
extern int MS_compute_numerical_area_vol(MOLSRF *ms, SES *sc, int mode);
extern int MS_update_triangulation_SES_component( MOLSRF *ms, RS *rs,
			int mode, float density, int nup );
extern int MS_write_triangulated_SES_component( char *file, MOLSRF *ms,
				   SES *s, int no_header, int mode );
extern void  MS_free_templates( MOLSRF *ms );
extern void  MS_free_triangulated_surface( SES *s );

/* buried surface */
void    MS_resetBuriedVertexFlags( SES *su );
void    MS_resetBuriedVertexArea( SES *su );

/*  %apply float ARRAY2D[1][4] { float **atm } */
/*  #int   MS_findBuriedVertices(MOLSRF *ms,SES *su,float **atm, int nbat); */
int     MS_vertexBuriedSurfaceArea(SES *su);

/*  %apply float ARRAY2D[1][4] { float **coord1 } */
/*  %apply float ARRAY2D[1][4] { float **coord2 } */
/*  int     MS_find_interface_atoms( float **coord1, float **coord2, */
/*  				 int nat1, int nat2, float rp, */
/*  				 int **indClose, int *n1 ); */

/* misc */
extern int MS_write_coordinates( MOLSRF *ms, char *filename );
extern void MS_restore_radii(MOLSRF *ms, RS *rs);
extern void MS_printInfo(MOLSRF *ms, char *keyString);
extern float MS_genus( int c, int v, int e, int f );
extern int MS_tagCloseProbes( MOLSRF *ms, RS *rs, float cut );
extern int MS_compute_surface( MOLSRF *ms, float rp, float dens, 
			       float hdens=3.0);
extern int MS_update_surface( MOLSRF *ms, RS *rs, int mode,
			        float density, int nup );
extern void  MS_reset_msms_err(void);

/* not yet wrapped because of float **   */
/*extern int   MS_disjointSpheres(float **atm, int nbat, int *disj); */

/* help functions */
/*
%native (MS_get_triangles)PyObject *MS_get_triangles(PyObject *self, 
                                                     PyObject *args,
				                     PyObject *kwargs);
*/
%{

/*#include "arrayobject.h"*/
#include "arrayget.h"  /* function to get data from Numeric array */

static PyObject *_wrap_MS_get_triangles(PyObject *self, PyObject *args) {
 /*
  static char * argnames[] = { "ms", "ses", "atoms", "selnum", "base", NULL };
  */
  SES    *ses=NULL;
  MOLSRF *ms=NULL;
  PyObject * _ms = 0;
  PyObject * _ses = 0;

  PyArrayObject *v_array, *vi_array, *f_array;

  PyObject *atomIndices=NULL;
  int *atomIndices_data = NULL;
  int atomIndices_nd, *atomIndices_dims = NULL;
  unsigned char *flag;
  int keepOriginalIndices = 0 ;

  float *v;
  int   i, stat, dims[2], nv, nf, *f, *ind, selnum=3, base=1;
  swig_type_info *ty1 = SWIG_TypeQuery("MOLSRF *");
  swig_type_info *ty2 = SWIG_TypeQuery("SES *");
/*
  if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|Oii",  argnames,
				  &_ms, &_ses, &atomIndices, &selnum, &base))
*/
  if (MS_setjmp()) {
       PyErr_SetString(PyExc_RuntimeError, MS_err_msg);
       return NULL;
     }
  if(!PyArg_ParseTuple(args,(char *)"OOOiii:MS_get_triangles",&_ms, &_ses, &atomIndices, &selnum, &base, &keepOriginalIndices)) 
    return NULL;

    if ((SWIG_ConvertPtr(_ms,(void **) &ms, ty1, 1)) == -1){
    PyErr_SetString(PyExc_TypeError,"Type error in argument 1 of MS_get_triangles. Expected p_MOLSRF.");
    return NULL;
  }

    if ((SWIG_ConvertPtr(_ses,(void **) &ses, ty2, 1)) == -1){
      PyErr_SetString(PyExc_TypeError,"Type error in argument 1 of MS_get_triangles. Expected p_SES.");
      return NULL;
  }

  /*if (atomIndices) */
  if (atomIndices == Py_None)
	{
		atomIndices=NULL;
	}
  else
  {
    if(!Array_DupData(atomIndices, (void **)&atomIndices_data, PyArray_INT,
		&atomIndices_nd, &atomIndices_dims, 1, NULL) )
		{
		  PyErr_SetString(PyExc_RuntimeError,
				  "problem with the atom indices array");
		  return NULL;
		}
  }

  if(ses->nbtriv==0)
  {
    PyErr_SetString(PyExc_RuntimeError,
	      "This SES component is not yet triangulated");
    return NULL;
  }

  /* allocate array for atom flags */
  flag = (unsigned char *)malloc(ms->nbat * sizeof(unsigned char));
  if (atomIndices)
  {
    for (i=0; i<ms->nbat; i++) flag[i] = 0;
    for (i=0; i< atomIndices_dims[0]; i++)
    {
    	flag[atomIndices_data[i]] = 1;
    }
  }
  else
  {
    for (i=0; i<ms->nbat; i++) flag[i] = 1;
  }
  
  stat = MS_getTri( flag, ses, base, selnum, &nv, &v, &ind, &nf, &f
                    , keepOriginalIndices
                    );

  if (stat==MS_OK) {
    dims[0] = nv;
    dims[1] = 8;
    v_array = (PyArrayObject *)PyArray_FromDimsAndData(2, dims, PyArray_FLOAT,
						       (char *)v);
#ifdef _MSC_VER
  switch ( WinVerMajor() )
  {
    case 6: break; // Vista
	default: v_array->flags |= NPY_OWNDATA;
  }
#else
  // so we'll free this memory when this
  // array will be garbage collected
  v_array->flags |= NPY_OWNDATA; 
#endif

    dims[0] = nv;
    dims[1] = 3;
    vi_array = (PyArrayObject *)PyArray_FromDimsAndData(2, dims, PyArray_INT,
							(char *)ind);
#ifdef _MSC_VER
  switch ( WinVerMajor() )
  {
    case 6: break; // Vista
	default: vi_array->flags |= NPY_OWNDATA;
  }
#else
  // so we'll free this memory when this
  // array will be garbage collected
  vi_array->flags |= NPY_OWNDATA; 
#endif

    dims[0] = nf;
    dims[1] = 5;
    f_array = (PyArrayObject *)PyArray_FromDimsAndData(2, dims, PyArray_INT,
						       (char *)f);
#ifdef _MSC_VER
  switch ( WinVerMajor() )
  {
    case 6: break; // Vista
	default: f_array->flags |= NPY_OWNDATA;
  }
#else
  // so we'll free this memory when this
  // array will be garbage collected
  f_array->flags |= NPY_OWNDATA; 
#endif

  } else {
    Py_INCREF(Py_None); v_array = (PyArrayObject *)Py_None;
    Py_INCREF(Py_None); vi_array = (PyArrayObject *)Py_None;
    Py_INCREF(Py_None); f_array = (PyArrayObject *)Py_None;
  }
  if(atomIndices_dims && atomIndices_data)
    {
/*  memory bug on 2tbv ????
      free((void *)atomIndices_dims);
      free((void *)atomIndices_data);
*/
    }

  return Py_BuildValue("OOO", v_array, vi_array, f_array);
}
%}

/*%native(MS_get_triangles) extern PyObject *_wrap_MS_get_triangles (PyObject *self, PyObject *args, PyObject *kwargs);
*/
