/* $Header: /opt/cvs/mslibDIST/include/rbhtree.h,v 1.2 2006/04/21 21:11:12 sanner Exp $
 *
 * $Id: rbhtree.h,v 1.2 2006/04/21 21:11:12 sanner Exp $
 *
 * $Log: rbhtree.h,v $
 * Revision 1.2  2006/04/21 21:11:12  sanner
 * - moved to lib1.4, updated the include files, the wrapper and the __init__.py
 *
 * Revision 1.1  2003/07/02 18:07:07  sanner
 * - added files in src/lib and src/tools
 *
 * Revision 1.1.1.1  2002/03/29 19:43:17  sanner
 * mslib1.3 initial checking
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
#ifndef RBHTREEDEF
#define RBHTREEDEF
#define BH_MAXFINDCOUNT                   512 /* Maximum number of points
                                                 that can be returned in a
                                                 Find call */
#define BH_FINDCOUNT                      512 /* See above */
#define BH_MAXBOX                         10  /* Maximum original size of
                                                 a leaf node in points */
#define BH_PADDING                        10  /* Size of a chunk of memory
                                                 reallocated when
                                                 InsertPadding or
                                                 DeletePadding runs out */
#define BH_SEARCH_UP                      0   /* for FindDirection, search
                                                 up and then down the tree */
#define BH_SEARCH_DOWN                    1   /* for FindDirection, search
                                                 from the root downward */
#define RBH_INSERTPADDING                 25  /* Not used in the code */
#define RBH_DELETEPADDING                 50  /* Not used in the code */
#define RBH_SPACEPADDING                  15.0
#define BH_LARGE_SPACE_PADDING     999999.9   /* used for infinite padding */
#define BH_LEAFPADDING                    10  /* Not used in the code */
#define BH_OUTSIDE_TREE                   3   /* Warning: Point being moved
                                                 or inserted is outside the
                                                 tree boundaries */
#define BH_FILLED_PADDING                 4   /* Warning: The leaf node
                                                 array is full; cannot
                                                 insert or move a point */
#define BH_EMPTY_BOX                      5   /* Warning: Cannot delete
                                                 this point; its parent
                                                 box would become empty */
#define BH_ALREADY_DELETED                6   /* Warning: This point has
                                                 already been deleted */
#define BH_INVALID_POINT                  7   /* Warning: The ID number
                                                 passed is invalid for
                                                 one of a number of reasons */
#define BH_MEMORY_ERROR                   8   /* Failed in malloc or realloc*/
#define BH_X                              0
#define BH_Y                              1
#define BH_Z                              2
#define BH_YES                            1
#define BH_NO                             0
#define NSTEPS                            128 /* Number of steps in the
                                                 histogram used for dividing
                                                 nodes */
#define FLAG_OWNSMEMORY                   1
#define FLAG_EMPTY_TREE                   2

typedef struct TBHPoint {
  float Pos[3];                /* 3D coordinate of point */
  float Rad;                   /* Radius of point */
  void *Data;                  /* void Data, for reduced surface */
  int uInt;                    /* user settable integer */
  int ID;                      /* identification, same as "at", address
                                  in the array */
  struct TBHNode *Box;         /* box to which this point belongs */
} TBHPoint;

typedef struct TBHIndex {
  struct TBHPoint **Pts;       /* leaf array of pointers to points plus
                                  any padding space */
  int NumPts;                  /* number of valid points in Pts array;
                                  does not include padding; points are
                                  consecutive */
  int Size;                    /* size of Pts array; is the number of
                                  points originally in this leaf plus
                                  the padding; does not change, no
                                  reallocating is done for leaves */
} TBHIndex;

typedef struct TBHNode {
    struct TBHNode *Left, *Right, *Parent;
                               /* The Left and Right child nodes of
                                  this node, and the Parent node; Parent
                                  is NULL if this is root; Left and Right
                                  are NULL if this is a leaf */
    struct TBHPoint **Buffer;  /* Array of pointers to members of the
                                  tree->Pts array; this array is shuffled
                                  whereas the actual tree->Pts array is
                                  not */
    struct TBHIndex Index;     /* Index, see above */
    float xmin[3];             /* Minimum extents on the three axes */
    float xmax[3];             /* Maximum extents on the three axes */
    float  cut;                /* Place along the axis where the cutting
                                  plane is constructed */
    int    dim;                /* Dimension on which the cutting plane is
                                  constructed; -1 if leaf */
} TBHNode;

/* Static BH Tree */
typedef struct TBHTree {
    struct TBHNode *Root;      /* Root node */
    TBHPoint *Pts;             /* Pts array, is not reshuffled */
    int NumPts;                /* Number of points in Pts array */
    float xmin[3];             /* Minimum extents on the three axes */
    float xmax[3];             /* Maximum extents on the three axes */
    float rm;
#ifdef STATBHTREE
    long tot;    /* total number of neighbors returned by findBHclose */
    int max,min; /* min and max of these numbers */
    int nbr;     /* number of calls to findBHclose */
#endif
    short bfl;
} TBHTree;

typedef struct TRBHTree {
    struct TBHNode *Root;      /* Root node */
    TBHPoint *Pts;             /* Pts array, is not reshuffled */ 
    TBHIndex FreePts;          /* Index to the free points in the
                                  Pts array */
    int NumPts;                /* Number of points in Pts array
                                  ORIGINALLY; is not maintained
                                  through updates; Pts array is
                                  NOT CONTIGUOUS */
    int SizePts;               /* Size of Pts array */
    float xmin[3];             /* Minimum extents on the three axes */
    float xmax[3];             /* Maximum extents on the three axes */
    float rm;
#ifdef STATBHTREE
    long tot;    /* total number of neighbors returned by findBHclose */
    int max,min; /* min and max of these numbers */
    int nbr;     /* number of calls to findBHclose */
#endif
    short bfl;
    int Flags;
    int Granularity;
    int LeafPadding;
    float SpacePadding;
} TRBHTree;

/* LeafPadding: Padding added to the array of points in each leaf;
                The number of new points that can be added to each box
                through the move and insert commands */
/* Static BH Tree */

extern TBHTree *MS_GenerateBHTree(TBHPoint *Pts,
				  int NumPts,
				  int Granularity,
				  int LeafPadding,
				  float SpacePadding);
extern TBHNode *MS_FindBHNodeUp(TBHNode *node, float x[3]);
extern TBHNode *MS_FindBHNode(TBHTree *tree, float *x);
extern void MS_FreeBHTree(TBHTree *tree);
extern void MS_FreeBHNode(TBHNode *node);

/* ID: Identification number of the point to be moved; same as "at";
   NewPos: New position of the point;
   FindDirection: Either BH_SEARCH_UP (go up the tree and then down)
                  or BH_SEARCH_DOWN (start at the root and search down)
   This function is for the Static BH Tree */

extern int MS_MoveBHPoint(TBHTree *tree,
			  int ID,
			  float NewPos[3],
			  int FindDirection);

/* LeafPadding: The same as in the Generate call, passed to this
                function by Generate */

extern void MS_DivideBHNode(TBHNode *node,
			    float *xmin,
			    float *xmax,
			    float *sxmin,
			    float *sxmax,
			    int granularity,
			    int LeafPadding);
extern int MS_FindBHCloseAtomsDist(TBHTree *tree, float *x, float cutoff,
				   int *atom, float *dist, int maxn);
extern int MS_FindBHCloseAtomsInNodeDist(TBHNode *node, float *x,
					 float cutoff, int *atom, 
					 float *dist, int maxn);
extern int MS_FindBHCloseAtoms(TBHTree *tree, float *x, float cutoff,
			       int *atom, int maxn);
extern int MS_FindBHCloseAtomsInNode(TBHNode *node, float *x,
				     float cutoff, int *atom, int maxn);
/* 
 LeafPadding :  for every leaf an array of leaf->Index.NumPts + LeafPadding
                pointer is allocated to store pointers to TBHPoints in box
 InsertPadding: Number of points that can be added to the tree. If limit is
                reached error message and error code returned.
 DeletePadding: Number of points that can be deleted from the tree. 
 SpacePadding:  The amount of void "padding" space stored around the
                tree to allow inserts and moves outside of the original
                boundaries of the tree
*/

extern TRBHTree *MS_GenerateRBHTree(TBHPoint *Pts,
				    int NumPts,
				    int MaxPts,
				    int Granularity,
				    int LeafPadding,
				    int DeletePadding,
				    float SpacePadding,
				    int OwnsMemory);
extern TBHNode *MS_FindRBHNode(TRBHTree *tree, float *x);
extern void MS_FreeRBHTree(TRBHTree *tree);

/* *ID: Pointer to integer, filled with the new identification number
        of the inserted point; the array to which this point is added
        is not contiguous; thus this number cannot be predicted and
        should be stored when returned for future reference
*/
/* The below functions apply only to the Dynamic BH Tree */

extern int MS_InsertRBHPoint(TRBHTree *tree,
		      float *Pos, float Rad, void *Data, int uInt,
		      int *ID);
extern int MS_DeleteRBHPoint(TRBHTree *tree,
			     int ID);

/* FindDirection: Either BH_SEARCH_UP (search up and then down the
                  tree) or BH_SEARCH_DOWN (start at the root and search
                  down)
*/

extern int MS_MoveRBHPoint(TRBHTree *tree,
			   int ID,
			   float NewPos[3],
			   int FindDirection);
extern int MS_ModifyRBHPoint(TRBHTree *tree,
			     int ID,
			     float Rad);
extern int MS_ModifyBHPoint(TBHTree *tree,
			    int ID,
			    float Rad);
extern int MS_FindRBHCloseAtomsDist(TRBHTree *tree, float *x, float cutoff,
				    int *atom, float *dist, int maxn);
extern int MS_FindRBHCloseAtoms(TRBHTree *tree, float *x, float cutoff,
				int *atom, int maxn);

#endif
