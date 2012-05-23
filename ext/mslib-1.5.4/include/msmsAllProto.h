/* $Header: /opt/cvs/mslibDIST/include/msmsAllProto.h,v 1.6 2008/05/23 21:12:44 vareille Exp $
 *
 * $Id: msmsAllProto.h,v 1.6 2008/05/23 21:12:44 vareille Exp $
 *
 * $Log: msmsAllProto.h,v $
 * Revision 1.6  2008/05/23 21:12:44  vareille
 * replace far with far0 to avoid pb with windef.h that defines far,
 * this allows removal of /Za compiler option
 *
 * Revision 1.5  2007/08/04 03:13:26  vareille
 * 1.4.3 - corrected ignored atoms bug
 *
 * Revision 1.4  2007/01/22 22:26:53  vareille
 * modified getTriangles to accept new parameter keepOriginalIndices
 *
 * Revision 1.3  2006/04/21 21:11:12  sanner
 * - moved to lib1.4, updated the include files, the wrapper and the __init__.py
 *
 * Revision 1.2  2003/09/26 17:28:51  annao
 * Added Greg Couch's modifications
 *
 * Revision 1.1  2003/07/02 18:07:07  sanner
 * - added files in src/lib and src/tools
 *
 * Revision 1.1.1.1  2002/03/29 19:43:17  sanner
 * mslib1.3 initial checking
 *
 * Revision 0.5  2000/08/18 00:01:31  sanner
 * added Greg Couch's modifications
 *
 * Revision 0.5  2000/08/12 02:09:06  gregc
 * switch from short to int for speed
 *
 * Revision 0.4  1999/11/20 03:07:09  sanner
 * - rmove lrs and lsu arguments
 * - modifed MS_find_interface_atoms prototype
 *
 * Revision 0.3  1999/11/12 01:15:34  sanner
 * before frs change
 *
 * Revision 0.2  1999/11/09 01:07:56  sanner
 * moved MS_free_SES, MS_free_SES_comp and MS_free_atom_area_list to surf.c
 *
 * Revision 0.1  1999/10/28 16:56:05  sanner
 * *** empty log message ***
 *
 * Revision 0.0  1999/10/27 23:02:11  sanner
 * *** empty log message ***
 *
 * Revision 0.2  1998/03/25  22:36:14  sanner
 * temporary version with growable atom BHtree but buggy.
 *
 * Revision 0.1  1998/03/19  18:35:02  sanner
 * *** empty log message ***
 *
 */
#ifndef __MSMS_ALL_PROTO
#define __MSMS_ALL_PROTO

/* this file contains prototypes for all functions in the library */

/* RBHTREE.C see rbhtree. h */


/* DIVERS.C */
extern void  MS_reset_atoms(MOLSRF *ms);

extern void  MS_change_contact_atoms(MOLSRF *ms,SES *su,float val);
extern void  MS_mark_atoms(MOLSRF *ms,SES *su);
extern void  MS_change_radii(MOLSRF *ms,float val,float dist, int up);
extern void  MS_modify_atoms(MOLSRF *ms,RS *rs,float val,float dist, int up);
extern void  MS_restore_radii(MOLSRF *ms, RS *rs);
extern void  MS_reset_atom_update_flag( MOLSRF *ms );

extern void  MS_extfil(char *filext,char *file,char *ext,int replace);
extern FILE *MS_open_file(char *name,char *mode,char *ext);

extern int   MS_sortKeyList(char *str[],int nbkey);
extern void  MS_printKey(char *str[],int nbkey);
extern int   MS_findKey(char *str, char *keylist[], int nbkey);

extern void  MS_print(MOLSRF *ms, char *key);
extern void  MS_printInfo(MOLSRF *ms, char *keyString);

extern float MS_genus(int c, int v, int e, int f);
extern int   MS_MakeAVSHeader(char *FileName, int veclen, int dim1, int dim2,
			    char *label, char *type, int skip);

extern int   MS_getTri( unsigned char *flag, SES *ses, int base, int selnum,
	       int *nnv, float **nv, int **ni, int *nnf, int **nf
	       , int  keepOriginalIndices 
	       );

/* MSMS.C */
extern void  MS_warning_msg(char *form, ...);
extern void  MS_reset_msms_err(void);
extern void  MS_error_msg(char *form, ...);
#ifdef LONGJMP_EXIT
#include <setjmp.h>
extern jmp_buf	MS_jmp_buf;
#define	MS_setjmp()	setjmp(MS_jmp_buf)
#else
#define	MS_setjmp()	0
#endif
extern void  MS_exit(const char *fmt, ...);

extern int          MS_init_msmslib(void);
extern MOLSRF_ROOT *MS_init_molsrf_root(void);
extern SESR        *MS_new_SESR(char *name);
extern RSR         *MS_new_RSR(char *name);
extern MOLSRF      *MS_add_molsrf(MOLSRF_ROOT *msr, char *name);
extern void         MS_delete_molsrf(MOLSRF *ms);
extern void         MS_detailed_info_molsrf(MOLSRF *ms);
extern void         MS_info_molsrf(MOLSRF *ms);
extern void         MS_list_molsrf(MOLSRF_ROOT *msr,int verbose);
extern MOLSRF      *MS_find_molsrf_by_num(MOLSRF_ROOT *msr,int num);
extern MOLSRF      *MS_find_molsrf_by_name(MOLSRF_ROOT *msr,char *name);
extern int          MS_is_number(char *str);

extern int    MS_write_coordinates( MOLSRF *ms, char *filename );
extern void   MS_init_atm(MOLSRF *ms);
extern int    MS_fill_one_atom(MOLSRF *ms, RSV *atm, float *Coord, int l,
			      char *name, int surf, int hd);
extern int    MS_add_msms_atoms(MOLSRF *ms,int nat, int maxat, float **Coord,
				char **names, int *surf, int *hd);
extern int    MS_move_msms_atoms(MOLSRF *ms,float **Coord,int *Index,int Nato);
extern RS    *MS_get_rs_comp_by_num(MOLSRF *ms,int i);
extern SES   *MS_get_ses_comp_by_num(MOLSRF *ms,int i);
extern int    MS_disjointSpheres(float **atm, int nbat, int *disj);
extern int    MS_compute_surface( MOLSRF *ms, float rp, float dens, float hdens);
extern int    MS_update_surface( MOLSRF *ms, RS *rs, int mode,
			         float density, int nup );
extern int    MS_set_molsrf_param(MOLSRF *ms, char *key, void *val);


/* GEOM.C */
#include "geom.h"

/* TIMING.C */
#include "timing.h"

/* MESHTEMP.C */
extern int   MS_find_tmpl_edge(MAR *ar,int a,int b,int nb);
extern int   MS_add_mar(TM *t,int a,int b,int n);
extern int   MS_tmpl_face(TM *t,int a,int b,int c,int v);
extern void  MS_check_template(TM *t);
extern int   MS_mesh_template(MOLSRF *ms,float dens,int num,int hd);
extern void  MS_free_template_data(MOLSRF *ms,int i);
extern void  MS_free_templates(MOLSRF *ms);
extern TM   *MS_new_template_pointer(MOLSRF *ms,float r,float d);
extern int   MS_make_templates(MOLSRF *ms,SES *s);
extern int   MS_update_templates(MOLSRF *ms,SES *s);


/* READXYZR.C */
extern double  MS_str2d(char *s);
extern float   MS_str2f(char *s);
extern char   *MS_strsqz(char *target,char *source);
extern void    MS_free_coord(float **Coord,int nbat, char *names[], int *surf,
			     int *hd);
extern int     MS_read_xyzr(char *file, int *nat, 
			    float ***coords, char ***atnames,
			    int **surfflag, int **hdflag);
/* extern float **MS_read_xyzr(char *file,int *nat); */
extern int     MS_get_xyzr_line(FILE *in,float v[],int i);
extern int     MS_get_xyzr_update(MOLSRF *ms, char *file, int max);
extern int     MS_updateSpheres(MOLSRF *ms, int nb,int *indices,float coords[][4]);


/* REDUCED.C */
extern RSF  *MS_find_rsf(RS *rs, int num);
extern RSE  *MS_find_rse(RS *rs, int s1, int s2);
extern int   MS_check_rs_edge(RS *rs);
extern int   MS_check_rs_face(RS *rs);
extern int   MS_write_RS_component(MOLSRF *ms,RS *rs,char *name);
extern void  MS_free_RS_vertices(MOLSRF *ms);
extern int   MS_reset_RSR(MOLSRF *ms);
extern void  MS_free_RS_vertices(MOLSRF *ms);
extern RS   *MS_find_rs_component_by_num(MOLSRF *ms,int i);
extern int   MS_is_f_RS_face(RSF *f,int a1,int a2,int a3);
extern RSF  *MS_find_rs_face_from_edge(RS *rs,int s1,int s2,int s3);
extern RSF  *MS_find_rs_face(RS *rs,int a1, int a2, int a3);
extern void  MS_delete_rs_component(MOLSRF *ms,RS *s);
extern RS   *MS_merge_rs_components(MOLSRF *ms,RS *rs1, RS *rs2);
extern RS   *MS_add_rs_component(MOLSRF *ms);
extern void  MS_add_RSE2RSV(int s, RSE *a);
extern void  MS_delete_rs_edge(RS *rs,RSE *a);
extern RSE  *MS_add_rs_edge(RS *su,int s1,int s2,RSF *f1,RSF *f2,int type);
extern void  MS_delete_rs_face(MOLSRF *ms, RS *rs,RSF *f);
extern RSF  *MS_add_rs_face(RS *rs,RSE *ar1, int a3, double c[],int cp);
extern int   MS_is_bp_in_face(int a1,int a2,int a3,double bp[3]);
extern void  MS_edge_rs_contact_circles(RSE *a);
extern void  MS_edge_rs_singular_points(RSE *a);
extern int   MS_find_u(RSE *a, int a1, int a2, float rp, double z[],
	       TRBHTree *bht, double c[], double bp[], double *h, RS *rs);
extern void  MS_invert_rs_edge(RSE *ar);
extern void  MS_treat_rs_edge(int edge,RS *rs,RSF *f,float rp,TRBHTree *bht);
extern void  MS_treat_rs_face(RS *rs,RSF *f,float rp,TRBHTree *bht);
extern int   MS_find_first_3_atoms(MOLSRF *ms, int *a1, int *a2, int *a3,
			    double c[], double bp[3], double *h);
extern int   MS_find_first_rs_face_external(TRBHTree *bht, int a1, int *a2,
				int *a3, float rp,double c[],double bp[3],
				double *h,int axis);
extern int   MS_find_first_rs_face(TRBHTree *bht, int a1, int *a2, int *a3,
			    float rp, double c[], double bp[3], double *h);
extern int   MS_find_first_rs_face_cavity(TRBHTree *bht, int ffnba, int ffat[],
		       float rp, double c[], double bp[3], double *h);
extern int   MS_clear_rs(MOLSRF *ms, RS *rs, RSF *ffa);
extern int   MS_create_first_face(RS *rs, int a1, int a2, int a3, float rp,
                       double c[], double h, double bp[], int num);
extern void  MS_find_free_vertices(MOLSRF *ms,float rp,TRBHTree *bht);
extern int   MS_remove_neigh(TBHPoint *bhPts, int ato, int nb, int *cl);
extern int   MS_find_free_edges(MOLSRF *ms,RS *rs,TRBHTree *bht,float rp,int a1);
extern int   MS_reduced_surface(MOLSRF *ms);
extern void  MS_check_probe(TRBHTree *bht,RSF *fr,RSV *atm,float rp);
extern void  MS_write_rs_simple_ascii(FILE *out, MOLSRF *ms, RS *rs, int norm);
extern int   MS_write_rs_component(MOLSRF *ms,RS *rs,char *name,int mode);
extern int   MS_merge_RSV(MOLSRF *ms,RS *rs);
extern int   MS_tagCloseProbes(MOLSRF *ms, RS *rs, float cut);
extern void  MS_treat_boundary_rs_edge(int n, RS *rs, RSE *ar, RSF *fr1,
				       RSF *fr2);
extern void MS_addDeleteListRSF( RS *rs, RSF *fr );

extern int   MS_update_reduced_surface(MOLSRF *ms,RS *rs,int nup);
extern int   MS_nappe(RSV *v,RSF **f,RSE **a,int *nba,int *nbf);
extern int   MS_duplic_rsv(RS *rs, RSV *v, RSV *newv, RSF **f, RSE **a,
			   int nba, int nbf,int nv, int nnewv,int nbos);
extern void  MS_cp_rsv(RSV *from,RSV *to);
extern int   MS_split_RSV(MOLSRF *ms,RS *rs);
extern int   MS_writeSolventAccessibleatoms(MOLSRF *ms, char *name, int mode);
extern int   MS_set_MOLSRF_param(MOLSRF *ms, char *key, void *val);

/* SURF.C */
extern void   MS_free_atom_area_list(MOLSRF *ms);
extern void   MS_free_SES_comp(SES *su);
extern void   MS_reset_SESR(MOLSRF *ms);
extern SES   *MS_find_SES_component_by_num(MOLSRF *ms,int i);
extern int    MS_pra(SESE *a);
extern int    MS_prf(SESF *f);
extern SESF  *MS_find_sesf(SES *s, int num);
extern void   MS_renumberSES(SES *ses);
extern int    MS_set_SES_density( SES *ses, float density, float hdensity, float rp );

extern int    MS_toric_face(SESF *f);
extern SESV  *MS_last_id(SESV *s);
extern void   MS_merge_SESV(SESE *a, int i);
extern double MS_SES_edge_angle(SESE *a);
extern int    MS_point_in_edge(double p[3], SESE *a, int sommet_inclus);

extern void  MS_delete_ses_component(MOLSRF *ms,SES *s);
extern void  MS_delete_SES_vertex(SES *su,SESV *s);
extern void  MS_rm_SES_edge_face(SESE *a,SESF *f);
extern int   MS_remove_sese_from_sesv(SESE *a,SESV *s);
extern void  MS_delete_SES_edge(SES *su,SESE *a);
extern void  MS_delete_SES_face(SES *su,SESF *f);
extern void  MS_free_triangulated_surface(SES *s);
extern SES  *MS_add_SES_component(MOLSRF *ms,SESR *root);
extern SESV *MS_add_SES_vertex(SES *su,int type);
extern int   MS_add_sese_to_sesv(SESE *a,SESV *s);
extern SESE *MS_add_SES_edge(SES *su, int type, SESV *s1, SESV *s2,
			     SESF *f1, SESF *f2, RSE *ar,
			     double c[], float r, double n[]);
extern SESF *MS_add_SES_face(SES *su,int type,int nba,void *dual);

extern void  MS_circle_displacement (double *c1, SESE *a);
extern void  MS_singular_SES_vertex_coord(double c[3], double r, double v[3],
					  int a, float *x, float *vn);
extern void  MS_SES_vertex_coord(float x1, float y1, float z1, float d,
				 float x2, float y2, float z2, float rp,
				 float *x, float *n);
extern void  MS_base_SES_vertices(RSF *mf, SES *su, SESF *f, float rp);
extern int   MS_re_allocate_edge_list(SESF *f, int n);
extern void  MS_add_SES_edge_face(SESF *f, SESE *a, int direct);
extern void  MS_insert_SES_edge_face(SESF *f, SESE *a, int direct,
				     int before, SESE *b);
extern void  MS_assoc_vertex_atom(SESV *s, RSE *ar);
extern void  MS_treat_SES_edge(int edge, RSF *mf, SESF *f, SES *su, SESV *s1,
			       SESV *s2, SESV **ss1, SESV **ss2, float rp,
			       RSF *fr);
extern void  MS_make_torique_face(RSF *mf, RSF *pf, SESF *nf, SES *su,
				  SESE *lar, int edge, SESV *s[3], SESV *os1,
				  SESV *os2, SESV *ss1, SESV *ss2, float rp);
extern void  MS_treat_face_rs(RSF *mf, RSF *pf, SESE *lar, int edge, SES *su,
			      float rp, SESV *os1, SESV *os2, SESV *ss1,
			      SESV *ss2);
extern int   MS_find_SES_edge_cycle(SESF *f);
extern int   MS_SES_vertex_in_face(SESV *s, SESF *f, int nba, float rp);
extern int   MS_ordonne_aretes(SES *su, float rp, int fl);
#if defined(decmips)
extern int   MS_comp(void *a,void *b);
#else
extern int   MS_comp(const void *a,const void *b);
#endif
extern int   MS_is_point_in_contact_face(float *s, SESF *f, float cut);
extern void  MS_find_SES_contact_faces(SES *su, SESF *ffa);
extern void  MS_treat_free_RS_edges(MOLSRF *ms, SES *su, RSE *ffar);
extern int   MS_match_coord(SESE *a1,SESE *a2);
extern SESE *MS_find_SES_edge_from_points(SESF *f, double p1[3], double p2[3],
					  int   *croise);
extern void  MS_mid_edge_point(double v1[3], double v2[3], float c[3],
			       float r, float ang, double v3[3], double mp[3]);
extern int   MS_edges_inter(SESE *a1,SESE *a2, double p[2][3]);
extern int   MS_closest_inter(SESE *a, int s, int *frt, int nb, double pi[],
			      int *full);
extern int   MS_sphere_inter_edge(double sc[3], double sr, SESE *a, 
				  double pcs[2][3], int *full, double v1[3],
				  double u[3], double v4[3], double an[2]);
extern int   MS_sphere_mange_arete1(double sc[3], double sr, SESE *a,
				   double pcs[2][3], int *full, double v1[3],
				   double v2[3], double u[3],
				   double v3[3], double an[2], int *nbi);
extern int    MS_sphere_mange_arete(double sc[3], double sr, SESE *a,
				   double pcs[2][3], int *full, double v1[3],
				   double v2[3],double u[3], double v3[3],
				   double an[2], int *nbi);
extern int    MS_sphere_inter_circle(double sc[3], double sr, double cc[3],
				    double cr,double u[3], double v1[3],
				    double p[2][3]);
extern int    MS_three_spheres_inter(double sc1[3], double sc2[3],
				     double sc3[3], double p[][3]);
extern int    MS_sphere_inter_circle2(double sc[],SESE *a,double p[][3]);
extern int    MS_dummy_edge_end(SESE *a, SESF *f, double rp, double p[2][3],
			       SESV *s);
extern int    MS_full_eaten(SESE *a, RSF *fr3);
extern int    MS_valide_SES_edge(SESE *a, SESF **f1, SESF **f2, double p[2][3],
				double rp, int *full);
extern SESF  *MS_cut_SES_face(SES *su, SESF *f, int a1, int a2);
extern void   MS_stich_SES_faces(SESF *f1, SESF *f2, SESE *a1, SESE *a2,
			       double u[3]);
extern int    MS_srdf(SES *su, RSF *ffa, float rp);
extern SESV  *MS_find_SES_singular_vertex_from_coord(SESF *f,double p[3]);
extern int    MS_neighbor_SES_faces(SESF *f1, SESF *f2);
extern int    MS_neighbor_RS_faces(RSF *fr1,RSF *fr2);
extern int    MS_sing_neighbor_RS_faces(RSF *fr1,RSF *fr2);
extern int    MS_remove_own_probes(int *frt,int nb,RSF *fr1,RSF *fr2);
extern SESE  *MS_copy_edge(SES *su,SESE *a);
extern int    MS_full_edge_eaten_1probe(SES *su,float rp,SESE *a,RSF *fr1,
				RSF *fr2, RSF *fr3,SESF *f1,SESF *f2,SESF *f3);
extern int    MS_same_edge(SESE *a1,SESE *a2);
extern int    MS_remove_double_edges(SES *su, SESF *f);
extern SESV  *MS_end_SES_edge(int num,SESF *f1);
extern int    MS_find_closest_atom(SESV *s,RSF *fr1,RSF *fr2,RSF *fr3);
extern int    MS_correct_edge(SES *su, double rp, SESE *a, int s,
			      SESV *ss1,SESF *f3, double p[3],int full);
extern int    MS_check_SES_edge(SES *su, double rp, SESE *a);
extern int    MS_SES_singularities(SES *su, SESE *far0, double rp);
extern void   MS_compute_beta_angles(SES *su);
extern double MS_singular_SES_vertex_beta(RSF *fr, SESE *a, SESE *na,
					  int sig1, int sig2, SESV *s);
extern void   MS_compute_SES_component_area(SES *su, double rp);
extern int    MS_update_SES_area(MOLSRF *ms, SES *su);
extern int    MS_compute_SES_area(MOLSRF *ms);
extern int    MS_write_surface_areas(MOLSRF *ms,char *f,int cpnum);
extern int    MS_restart_from_ses(MOLSRF *ms, RS *rs, int nup);
extern int    MS_solvent_excluded_surface(MOLSRF *ms,RS *rs);
extern int    MS_update_ses(MOLSRF *ms,RS *rs,int nup);


/* TRIANG.C */
extern int   MS_restart_from_triangulation(MOLSRF *ms, RS *rs, int mode,
				float density, int nup);
extern void  MS_write_triangulated_analytical_surface(MOLSRF *ms, SES *sc,
				   FILE *outf, FILE *outv, int no_header);
extern void  MS_write_triangulated_surface(MOLSRF *ms, SES *sc, FILE *outf,
					   FILE *outv, int no_header);
extern int   MS_write_triangulated_SES_component(char *file, MOLSRF *ms,SES *s,
						 int no_header, int mode);
extern int   MS_full_torus(SESF *f);
extern void  MS_free_SESF_tri(SES *s,SESF *f);
extern void  MS_free_SESE_tri(SES *s,SESE *a);
extern int   MS_add_SES_vertices(SES *s, SESV *v);
extern int   MS_triangulate_isolated_sphere(MOLSRF *ms,SES *su);
extern void  MS_add_triangle(SESF *f,TRIV *i,TRIV *j,TRIV *k);
extern int   MS_triangulate_toric_face(MOLSRF *ms,SESF *f);
extern int   MS_triangulate_full_torus(MOLSRF *ms,SESF *f);
extern int   MS_SESE_tri(SES *s,SESE *a);
extern int   MS_find_edge(TRIV *ar[][2], int nb, TRIV *a, TRIV *b);
extern int   MS_edge_cross(TRIV *d1, TRIV *d2, TRIV *a[][2], int nb,
			   float av[][3], unsigned char lib[],
			   float v[3], float c[3]);
extern void  MS_copy_tdot_to_triv(TRIV *to, TDOT *from);
extern int   MS_scrs(MOLSRF *ms,TM *tm, int nb,SESF *fc,int *pb);
extern void  MS_select_neigh(int s,MDOT *prb,int *n);
extern int   MS_triangulate_reentrant_face(MOLSRF *ms,SESF *f,int *pb);
extern int   MS_find_oriented_edge(TRIV *ar[][2], int nb, TRIV *a, TRIV *b);
extern int   MS_sccs(MOLSRF *ms,TM *tm,int nb,SESF *fc,double *wn,int *pb);
extern int   MS_triangulate_contact_face(MOLSRF *ms,SESF *f,int *pb);
extern int   MS_compute_numerical_area_vol(MOLSRF *ms, SES *sc, int mode);
extern int   MS_triangulate_SES_component(MOLSRF *ms,RS *rs);
extern int   MS_update_triangulation_SES_component(MOLSRF *ms,RS *rs, int mode,
                                                  float density, int nup);


/* BURIED.C */
extern void    MS_resetBuriedVertexFlags(SES *su);
extern void    MS_resetBuriedVertexArea(SES *su);
extern void    MS_buriedReentrantFace(SESF *f, TBHTree *bht, float **atm,
				      float rp);
extern void    MS_buriedContactFace(SESF *f,TBHTree *bht,float **atm,
				    float rp);
extern void    MS_buriedToricFace(SESF *f, TBHTree *bht, float **atm,
				  float rp);
extern int     MS_findBuriedVertices(MOLSRF *ms,SES *su,float **atm,
				     int nbat);
extern int     MS_vertexBuriedSurfaceArea(SES *su);
extern float   MS_faceBuriedSurfaceArea(SES *su);
extern float   MS_buriedSurfaceArea(MOLSRF *ms, SES *su, int mode);
extern int     MS_find_interface_atoms(float **coord1, float **coord2,
				       int nat1, int nat2, float rp,
				       int **indClose, int *n1);

#endif
