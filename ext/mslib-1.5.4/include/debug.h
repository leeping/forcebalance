/* $Header: /opt/cvs/mslibDIST/include/debug.h,v 1.2 2006/04/21 21:11:12 sanner Exp $
 *
 * $Id: debug.h,v 1.2 2006/04/21 21:11:12 sanner Exp $
 *
 * $Log: debug.h,v $
 * Revision 1.2  2006/04/21 21:11:12  sanner
 * - moved to lib1.4, updated the include files, the wrapper and the __init__.py
 *
 * Revision 1.1  2003/07/02 18:07:07  sanner
 * - added files in src/lib and src/tools
 *
 * Revision 1.1.1.1  2002/03/29 19:43:17  sanner
 * mslib1.3 initial checking
 *
 * Revision 0.1  2000/08/18 00:01:31  sanner
 * added Greg Couch's modifications
 *
 * Revision 0.1  2000/08/12 02:09:06  gregc
 * switch from short to int for speed
 *
 * Revision 0.0  1999/10/27 23:02:11  sanner
 * *** empty log message ***
 *
 * Revision 0.1  1998/03/19  18:35:02  sanner
 * *** empty log message ***
 *
 */
#ifndef MS_DEBUG__
#define MS_DEBUG__

/* used for debugging */

extern RSE *drse;
extern RSF *drsf;

extern SESV *dsesv;
extern SESE *dsese;
extern SESF *dsesf;

extern RSF  *MS_find_rsf(RS *rs, int num);
extern RSE  *MS_find_rse(RS *rs, int s1, int s2);

extern int MS_prf(SESF *f);
extern int MS_pra(SESE *a);
extern SESF  *MS_find_sesf(SES *s, int num);

#endif
