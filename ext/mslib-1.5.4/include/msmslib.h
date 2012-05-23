/* $Header: /opt/cvs/mslibDIST/include/msmslib.h,v 1.2 2006/04/21 21:11:12 sanner Exp $
 *
 * $Id: msmslib.h,v 1.2 2006/04/21 21:11:12 sanner Exp $
 *
 * $Log: msmslib.h,v $
 * Revision 1.2  2006/04/21 21:11:12  sanner
 * - moved to lib1.4, updated the include files, the wrapper and the __init__.py
 *
 * Revision 1.1  2003/07/02 18:07:07  sanner
 * - added files in src/lib and src/tools
 *
 * Revision 1.1.1.1  2002/03/29 19:43:17  sanner
 * mslib1.3 initial checking
 *
 * Revision 0.1  2000/03/08 18:56:33  sanner
 * *** empty log message ***
 *
 * Revision 0.0  1999/10/27 23:02:11  sanner
 * *** empty log message ***
 *
 * Revision 0.1  1998/03/19  18:35:02  sanner
 * *** empty log message ***
 *
 */
#ifndef MSMSLIBDEF
#define MSMSLIBDEF

/* general static variables */
#define MS_PI 3.14159265
#define MS_TWOPI 6.28318530
#define MS_ONETHIRD (float).3333333

/* hostname */
extern char *MS_host;
extern char *MS_VERSION;
extern char *MS_CPFL;

#include "mslib.h"

/* Reduced Surface stuff */
#define IS_TO_SPLIT(i)      i & 1
#define SET_TO_SPLIT(i)     i |= 1
#define RESET_TO_SPLIT(i) ( i &= 254 )
#define IS_SPLITED(i)       i & 2
#define SET_SPLITED(i)      i |= 2
#define RESET_SPLITED(i)  ( i &= 253 )
#define IS_TO_MERGE(i)       i & 4
#define SET_TO_MERGE(i)      i |= 4
#define RESET_TO_MERGE(i)  ( i &= 251 )

#define MS_IS_SINGULAR_SESE(i) i & 12

extern SESE *MS_lastSESE; /* pointer to last kept edge when partial update */
extern SESF *MS_lastSESF; /* pointer to last kept face when partial update */

#include "msmsAllProto.h"

#ifdef DEBUG
#include "debug.h"
#endif

#endif
