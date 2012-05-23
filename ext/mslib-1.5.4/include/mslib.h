/* $Header: /opt/cvs/mslibDIST/include/mslib.h,v 1.3 2008/05/23 21:12:44 vareille Exp $
 *
 * $Id: mslib.h,v 1.3 2008/05/23 21:12:44 vareille Exp $
 *
 * $Log: mslib.h,v $
 * Revision 1.3  2008/05/23 21:12:44  vareille
 * replace far with far0 to avoid pb with windef.h that defines far,
 * this allows removal of /Za compiler option
 *
 * Revision 1.2  2006/04/21 21:11:12  sanner
 * - moved to lib1.4, updated the include files, the wrapper and the __init__.py
 *
 * Revision 1.1  2003/07/02 18:07:07  sanner
 * - added files in src/lib and src/tools
 *
 * Revision 1.1.1.1  2002/03/29 19:43:17  sanner
 * mslib1.3 initial checking
 *
 * Revision 0.8  2000/08/18 00:01:31  sanner
 * added Greg Couch's modifications
 *
 * Revision 0.8  2000/08/12 02:09:06  gregc
 * switch from short to int for speed
 *
 * Revision 0.7  2000/03/08 18:56:29  sanner
 * *** empty log message ***
 *
 * Revision 0.6  1999/11/23 22:26:04  sanner
 * added surface update triangulation modes
 *
 * Revision 0.5  1999/11/20 03:05:55  sanner
 * - removed lrs argument in MS_reduced_surface
 * - removed lsu argument in MS_solvent_excluded_surface
 * - modified MS_find_interface_atoms prototype
 *
 * Revision 0.4  1999/11/12 01:15:19  sanner
 * before frs change
 *
 * Revision 0.3  1999/11/08 17:49:49  sanner
 * synchronized with win32 source code
 *
 * Revision 0.2  1999/11/03 03:52:50  sanner
 * moved uprbRBHT and sprbRBHT from rsr to rs
 *
 * Revision 0.1  1999/10/28 16:56:00  sanner
 * *** empty log message ***
 *
 * Revision 0.0  1999/10/27 23:02:11  sanner
 * *** empty log message ***
 *
 * Revision 0.3  1998/09/03  22:39:21  sanner
 * added needFreeEdgeCheck to the RS structure
 *
 * Revision 0.2  1998/03/25 22:36:14  sanner
 * temporary version with growable atom BHtree but buggy.
 *
 * Revision 0.1  1998/03/19  18:35:02  sanner
 * *** empty log message ***
 *
 */
#ifndef MSLIBDEF
#define MSLIBDEF

#ifdef WIN32
#define strdup _strdup
#define strncasecmp strncmp
#endif

#include "meshtemp.h"  /* needed for trangulation template structures */
#include "rbhtree.h"   /* needed for binary spatial division trees */ 

/*************************************************************************

  DEFINES

*************************************************************************/

/* reduced surface output formats */
#define MS_RS_SIMPLE_ASCII1 0
#define MS_RS_SIMPLE_ASCII2 1
#define MS_RS_SIMPLE_ASCII1_NORMALS 2

/* restart modes */
#define MS_NON_PARTIAL 0
#define MS_PARTIAL 1

/* triangulation surface output formats */
#define MS_TSES_ASCII 0   /* whole triangulated surface */
#define MS_ASES_ASCII 1   /* discrete representation of the Analytical surf. */
#define MS_TSES_ASCII_AVS 10  /* same as MS_TSES_ASCII and AVS headers */
#define MS_ASES_ASCII_AVS 11  /* same as MS_ASES_ASCII and AVS headers */

/* buried surface computation modes */
#define MS_NUMERICAL 1         /* numerical calculation using flat triangles */
#define MS_SEMI_ANALYTICAL 2   /* numerical calculation using spherical tri. */
#define MS_BOTH 3

/* SESF buried flag
#define MS_NOT_BURIED 0
#define MS_BURIED 1
#define MS_PARTIALLY_BURIED 2
*/

/* analytical SES edges types */
#define SAILLANT 1
#define RENTRANT 2
#define TORIQUE 3
#define SINGULIER 4
#define SINGULIER_TRAITE 8
#define SELECT 16
#define FULL_EATEN 32
#define RENT_FUS 64

#define FULL_TORUS1 4  /* SES face type for full torus or first cone */
#define FULL_TORUS2 5  /* SES face type for second cone of full torus */

/* error handling stuff */
#define MS_OK 0
#define MS_ERR 1
#define MS_ErrMsgMaxLen 512

/* surface update triangulation modes */
#define FULL 0              /* the surface is rebuilt completely */
#define TORIC 1             /* only toric faces are triangulated */
#define ALL_WITH_DENSITY 2  /* all rebuilt faces are triangulated with 
			       the given density */
#define NO_TRIANGULATION 3   /* no triangulation at all */

/*************************************************************************

  GLOBALS

*************************************************************************/
extern char *MS_VERSION;
extern int MS_err_flag;
extern char MS_err_msg[MS_ErrMsgMaxLen];

/*************************************************************************

  DATA STRUCTURES

*************************************************************************/

/* Reduced Surface stuff */

/* RS edges */
struct RSE__ {
    struct RSE__  *nxt;
    struct RSE__  *prv;
    struct RSF__  *f[2];
    struct SESF__ *rsedual[2]; /* pointeur vers la ou les faces toric dual */
    struct RSE__ *nxtUp;    /* chain of deleted or reconstructed edges */
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

/* RS vertex */
struct RSV__ {
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
    unsigned char hdensity; /* set to 1 if surface for this atom should be 
			     high density triangualtion */
    unsigned char split;     /* =1 when has been checked for free edge */
    char *name;              /* atom name */
};
typedef struct RSV__ RSV;


/* RS face */
struct RSF__ {
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

/* RS header */
struct RS__ {
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

/* RS tree root */
struct RSR__ {
  struct RS__ *fst;
  struct RS__ *lst;
  struct RSV__ *atm;
  TRBHTree *atmBHT;   /* growable and movable atoms BHtree */
  int nb;           /* number of RS components */
  int nb_free_vert; /* number of free vertices in the Reduced Surface */
  int ffnba;        /* used to compute only one component */
  int ffat[3];      /* atom indexes for first RS face */
  short all_comp_done;
  short ext_comp_done;
};
typedef struct RSR__ RSR;

/* REDUCED SURFACE DEFINES */
#define NLRS  (RS *)NULL
#define NLRSF (RSF *)NULL
#define NLRSE (RSE *)NULL
#define NLRSV (RSV *)NULL

/* RSV types */
#define REGULAR 0
#define FREE 1
#define DUMMY 2

#define Max_RS_Components 255

/* Analytic Solvent Excluded Surface structures */

struct TRIV__ {
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

struct TRI__ {  /* triangular face */
  struct TRIV__ *s[3];     /* vertices */
};
typedef struct TRI__ TRI;

#define NLTRI (TRI *)NULL
#define NLTRIV (TRIV *)NULL

/*
   array in MOLSRF used to store the coordinates of the template
   vertices which are in the contact or reentrant face
*/
struct TDOT__ {
  float c[3];	  /* coord. de ce sommet */
  float n[3];	  /* vecteur normal a la surface */
  int   atm;	  /* # de l'atome auquel appartient ce sommet */
  struct TRIV__ *triv;
};
typedef struct TDOT__ TDOT;
#define NLTDOT (TDOT *)NULL

struct SESV__ {           /* Solvent Excluded Surface Vertex */
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

struct SESE__ {           /* Solvent Excluded Surface Edge */
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

struct SESF__ {           /* Solvent Excluded Surface Face */
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

struct SES__ {
    struct SES__ *nxt;
    struct SES__ *prv;
  /* analytical surface description */
    struct SESF__  *ffa;    /* pointer to first face */
    struct SESF__  *lfa;    /* pointer to last face */
    struct SESE__  *far0;    /* pointer to first arete */
    struct SESE__  *lar;    /* pointer to last arete */
    struct SESV__  *fso;    /* pointer to first sommet */
    struct SESV__  *lso;    /* pointer to last sommet */
    struct SESF__  *lsesf;  /* last ses face kept during update */
    struct SESF__  *fsttfup; /* linked list of faces which need triangulat. */
    struct SESF__  *lsttfup; /* last lelement in list */

    struct SESF__ *fsttorup;  /* list of toric SESF to look at for finding */
    struct SESF__ *lsttorup;  /* contact faces during an update operation */

    struct SESE__  *lsese;  /* last ses edge kept during update */
    struct SESV__  *lsesv;  /* last ses vertex kept during update */
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
    float  density, density_sq;
    float  hdensity, hdensity_sq;
    float  cercang;    /* angle between 2 dots on a cercle of radius 1.0 */
    float  hdcercang;  /* angle between 2 dots on a cercle of radius 1.0 high density*/
    int    npcerc;     /* number of dots on a cercle of radius 1.0 */
    int    hdnpcerc;   /* number of dots on a cercle of radius 1.0 high density*/
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

struct SESR__ {
  struct SES__ *fst;
  struct SES__ *lst;
  int nb;
};
typedef struct SESR__ SESR;

#define NLSES (SES *)NULL
#define NLSESF (SESF *)NULL
#define NLSESE (SESE *)NULL
#define NLSESV (SESV *)NULL

/* structure for a molecular surface */
struct MOLSRF__ {
  struct MOLSRF__ *nxt;
  struct MOLSRF__ *prv;
  struct MOLSRF_ROOT__ *up;
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
  float rp;
  float rp1;
  float density;
  float hdensity;

  /* triangulation stuff */
  struct TM__ **tmtab; /* pointers to template spheres */
  struct TM__ **hdtmtab; /* pointers to template spheres at high density */
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
  short all_components;
  short free_vert;
  short cusp_trim;
  short noh;
  short dotsOnly;      /* write only .vert file, no triangulation */
  short cleanup_rs;
  short cleanup_ses;

  /* update stuff */
  int fstup;               /* first updated RS vertex */
  int up_num;              /* current update number */
  int up_mode;             /* update_mode */
  float up_density;        /* update_density */
  unsigned char MS_mfl;
  unsigned char restart_mode; /* MS_PARTIAL (default) or MS_NON_PARTIAL */

  /* restart options */
  float mod_dist;    /* restart modification distance ??? every atom within
			rest_mod_dist of a modified atom is marked modified */
  short restart_ana; /* restart computation after analytical SES pb */
  short restart_tri; /* restart computation after triang. pb */

  short rest_on_pbr; /* restart after pb while triangulating a reent. face */
  short rest_on_pbc; /* restart after pb while triangulating a contact face */
  short max_try;     /* Max number of re-try after SES and TRI */
  short try_num;     /* current number of re-tries */
};
typedef struct MOLSRF__ MOLSRF;

#define NLMOLSRF (MOLSRF *)NULL

struct MOLSRF_ROOT__ {           /* MOLSRF ROOT */
    int  nb;
    struct MOLSRF__ *fst;
    struct MOLSRF__ *lst;
};
typedef struct MOLSRF_ROOT__ MOLSRF_ROOT;

#define NLMOLSRF_ROOT (MOLSRF_ROOT *)NULL

/* /\********************************************************************* */

/*     INTERFACE FUNCTIONS PROTOTYPE */

/* **********************************************************************\/ */

/* /\* initialization *\/ */
/* extern int        MS_init_msmslib(void); */
/* extern MOLSRF_ROOT *MS_init_molsrf_root(void); */

/* /\* errors *\/ */
/* extern void  MS_warning_msg(char *form, ...); */
/* extern void  MS_reset_msms_err(void); */
/* extern void  MS_error_msg(char *form, ...); */

/* /\* MOLSRF *\/ */
/* extern MOLSRF  *MS_add_molsrf(MOLSRF_ROOT *msr, char *name) */
/* extern void    MS_delete_molsrf( MOLSRF *ms ); */
/* extern void    MS_detailed_info_molsrf( MOLSRF *ms ); */
/* extern void    MS_info_molsrf( MOLSRF *ms ); */
/* extern void    MS_list_molsrf( MOLSRF_ROOT *msr, int verbose ); */
/* extern MOLSRF *MS_find_molsrf_by_num( MOLSRF_ROOT *msr, int num ); */
/* extern MOLSRF *MS_find_molsrf_by_name( MOLSRF_ROOT *msr, char *name ); */
/* extern int     MS_set_molsrf_param( MOLSRF *ms, char *key, void *val ); */


/* /\* atom coordinates *\/ */
/* /\* */
/* extern float **MS_read_xyzr( char *file, int *nat ); */
/* extern void    MS_free_coord(float **Coord,int nbat); */
/* *\/ */
/* extern int     MS_get_xyzr_update( MOLSRF *ms, char *file, int max ); */
/* extern int     MS_read_xyzr(char *file, int *nat,  */
/* 			    float ***coords, char ***atnames, */
/* 			    int **surfflag, int **hdflag); */
/* extern void    MS_free_coord(float **Coord,int nbat, char *names[], int *surf, */
/* 			     int *hd); */
/* extern void  MS_reset_atom_update_flag( MOLSRF *ms ); */


/* /\* reduced surface *\/ */
/* extern int   MS_reduced_surface( MOLSRF *ms ); */
/* extern int   MS_update_reduced_surface( MOLSRF *ms, RS *rs, int nup ); */
/* extern RS   *MS_find_rs_component_by_num( MOLSRF *ms, int i ); */
/* extern int   MS_write_rs_component(MOLSRF *ms, RS *rs, char *name, int mode ); */
/* extern int   MS_writeSolventAccessibleatoms( MOLSRF *ms, char *name, int mode); */
/* extern int   MS_reset_RSR( MOLSRF *ms ); */
/* extern void  MS_free_RS_vertices(MOLSRF *ms); */


/* /\* analytical solvent excluded surface *\/ */
/* extern int    MS_solvent_excluded_surface( MOLSRF *ms, RS *rs ); */
/* extern int    MS_update_ses( MOLSRF *ms, RS *rs, int nup ); */
/* extern int    MS_compute_SES_area(MOLSRF *ms); */
/* extern int    MS_update_SES_area( MOLSRF *ms, SES *su ); */
/* extern SES   *MS_find_SES_component_by_num( MOLSRF *ms, int i ); */
/* extern int    MS_write_surface_areas( MOLSRF *ms, char *f, int cpnum ); */
/* extern int    MS_set_SES_density( SES *ses, float density, float hdensity, float rp ); */
/* extern void   MS_free_SES_comp( SES *su ); */
/* extern void   MS_reset_SESR( MOLSRF *ms ); */


/* /\* triangulated solvent excluded surface *\/ */
/* extern int   MS_make_templates( MOLSRF *ms, SES *s ); */
/* extern int   MS_triangulate_SES_component( MOLSRF *ms, RS *rs ); */
/* extern int   MS_compute_numerical_area_vol(MOLSRF *ms, SES *sc, int mode); */
/* extern int   MS_update_triangulation_SES_component( MOLSRF *ms, RS *rs, */
/* 			int mode, float density, int nup ); */
/* extern int   MS_write_triangulated_SES_component( char *file, MOLSRF *ms, */
/* 				   SES *s, int no_header, int mode ); */
/* extern void  MS_free_templates( MOLSRF *ms ); */
/* extern void  MS_free_triangulated_surface( SES *s ); */


/* /\* buried surface *\/ */
/* extern void    MS_resetBuriedVertexFlags( SES *su ); */
/* extern void    MS_resetBuriedVertexArea( SES *su ); */
/* extern int     MS_findBuriedVertices(MOLSRF *ms,SES *su,float **atm, int nbat); */
/* extern int     MS_vertexBuriedSurfaceArea(SES *su); */
/* extern int     MS_find_interface_atoms(float **coord1, float **coord2, */
/* 				       int nat1, int nat2, float rp, */
/* 				       int **indClose, int *n1); */

/* /\* misc *\/ */

/* extern int   MS_write_coordinates( MOLSRF *ms, char *filename ); */
/* extern void  MS_restore_radii(MOLSRF *ms, RS *rs); */
/* extern void  MS_printInfo(MOLSRF *ms, char *keyString); */
/* extern float MS_genus( int c, int v, int e, int f ); */
/* extern int   MS_tagCloseProbes( MOLSRF *ms, RS *rs, float cut ); */
/* extern int   MS_compute_surface( MOLSRF *ms, float rp, float dens, float hdens); */
/* extern int   MS_update_surface( MOLSRF *ms, RS *rs, int mode, */
/* 			        float density, int nup ); */
/* extern int   MS_disjointSpheres(float **atm, int nbat, int *disj); */

#endif
