/* $Header: /opt/cvs/mslibDIST/include/geom.h,v 1.2 2006/04/21 21:11:12 sanner Exp $
 *
 * $Id: geom.h,v 1.2 2006/04/21 21:11:12 sanner Exp $
 *
 * $Log: geom.h,v $
 * Revision 1.2  2006/04/21 21:11:12  sanner
 * - moved to lib1.4, updated the include files, the wrapper and the __init__.py
 *
 * Revision 1.1  2003/07/02 18:07:07  sanner
 * - added files in src/lib and src/tools
 *
 * Revision 1.1.1.1  2002/03/29 19:43:17  sanner
 * mslib1.3 initial checking
 *
 * Revision 0.2  2000/08/18 00:01:31  sanner
 * added Greg Couch's modifications
 *
 * Revision 0.2  2000/08/12 02:09:06  gregc
 * switch from short to int for speed
 *
 * Revision 0.1  2000/03/08 18:56:17  sanner
 * *** empty log message ***
 *
 * Revision 0.0  1999/10/27 23:02:11  sanner
 * *** empty log message ***
 *
 * Revision 0.1  1998/03/19  18:35:02  sanner
 * *** empty log message ***
 *
 */
#ifndef GEOMDEF
#define GEOMDEF

extern void MS_rotax(float a[3],float v[3],float ang,float rot[4][4]);

extern double MS_distance2(double x1,double y1,double z1,
	                double x2,double y2,double z2);

extern void MS_vvmult(double x1,double y1,double z1,
	    double x2,double y2,double z2,
	    float  x3[]);

extern void MS_dvvmult(double x1,double y1,double z1,
	     double x2,double y2,double z2,
	     double x3[]);

extern void MS_dvvdiff(double x1,double y1,double z1,
	     double x2,double y2,double z2,
	     double x3[]);

extern double MS_dvvscal(double x1,double y1,double z1,
	              double x2,double y2,double z2);

extern double MS_dvnorm(double x1,double y1, double z1);

extern int MS_dmknormvect(double a[],double b[],double c[],double *n);

extern int MS_mknormvect(float a[],float b[],float c[],double *n);

extern void MS_dvnormalize(double v[]);

extern void MS_dvortho(double a[], double b[]);

extern int MS_sphere_inter_torus(float ct[], float gr, float pr, double n[],
				float c[], float r);

extern double MS_sphad_inter_sphad(float c1[],float r1,float c2[],float r2,
				float rp,double c[],double *d);

extern double MS_sph_inter(double c1[],double c2[],float r,double c[]);

extern int MS_sph_tg_3sph(float c3[3],float r3,double ct[3],double rt,
	double a1a2[3],float rp,double c[3],double bp[3],double *h);

extern double MS_full_angle(double ba[],double bc[],double v[]);

extern void MS_tri_normal(float p1[],float p2[],float p3[],double v[]);

extern double MS_triang_area(float *p1,float *p2,float *p3);

extern int MS_circle_inter_circle(double c1[3],double r1,double n1[3],
			  double c2[3],double r2,double n2[3],
			  double p[2][3]);

extern double MS_sphericalTriangleArea(double r, float *c,
				       float *p1, float *p2, float *p3);

#endif
