/* $Header: /opt/cvs/mslibDIST/include/timing.h,v 1.2 2006/04/21 21:11:12 sanner Exp $
 *
 * $Id: timing.h,v 1.2 2006/04/21 21:11:12 sanner Exp $
 *
 * $Log: timing.h,v $
 * Revision 1.2  2006/04/21 21:11:12  sanner
 * - moved to lib1.4, updated the include files, the wrapper and the __init__.py
 *
 * Revision 1.1  2003/07/02 18:07:08  sanner
 * - added files in src/lib and src/tools
 *
 * Revision 1.1.1.1  2002/03/29 19:43:17  sanner
 * mslib1.3 initial checking
 *
 * Revision 0.1  1999/11/08 17:49:49  sanner
 * synchronized with win32 source code
 *
 * Revision 0.0  1999/10/27 23:02:11  sanner
 * *** empty log message ***
 *
 * Revision 0.1  1998/03/19  18:35:02  sanner
 * *** empty log message ***
 *
 */
#ifndef _PROC_TIMING
#define _PROC_TIMING

#ifdef TIMING
#include <sys/times.h>

extern long MS_get_clktck(void);
extern void MS_pr_times(char *str, clock_t real, struct tms *start, struct tms *end);
#endif

#endif
