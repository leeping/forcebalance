#include "Python.h"
#include "arrayobject.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "apc.h"

static PyObject *_Assign(PyObject *self, PyObject *args) {
  // Given a cost matrix as input,
  // return the column numbers that are paired with
  // row numbers (1....N) that give the minimum cost assignment.

  /**********************************/
  /*   Initialize input variables   */
  /**********************************/
  PyArrayObject *Mat_;
  
  if (!PyArg_ParseTuple(args, "O", &Mat_)) {
    printf("Mao says: Inputs / outputs not correctly specified!\n");
    return NULL;
  }
  
  /**********************************/
  /*   Initialize local variables   */
  /**********************************/
  int inf=1e9;           // Infinity parameter (needs to be bigger than costs)
  int ans=0;             // The minimized cost
  int i;                 // Row and column indices
  long unsigned int *Mat = (long unsigned int*) Mat_->data;
  // Dimensions of the matrix.
  int DIM = (int) Mat_->dimensions[0];
  // Allocate the assignment array.  I'm not too familiar with the Python-C interface
  // so it feels pretty clumsy.
  npy_intp dim1[1];
  dim1[0] = DIM;
  PyArrayObject *idx_;
  idx_ = (PyArrayObject*) PyArray_SimpleNew(1,dim1,NPY_INT);
  int *idx = (int*) PyArray_DATA(idx_);
  // The matrix passed from Python is "long unsigned int", which causes problems for apc.
  // We're going to create a new matrix but with "int" instead.
  int *Mat_Int = calloc(DIM*DIM, sizeof(int));
  for (i=0; i<DIM*DIM; i++) {
    Mat_Int[i] = (int) Mat[i];
  }
  /*
  int j;
  printf("Solving assignment problem for the following matrix:\n");
  for (i=0; i<DIM; i++) {
    for (j=0; j<DIM; j++) {
      printf("%8i ",(int) Mat[i*DIM+j]);
    }
    printf("\n");
  }
  */
  // Solve the assignment problem.
  apc(DIM,Mat_Int,inf,&ans,idx);
  //printf("The optimal assignment has cost %i\n",ans);
  free(Mat_Int);
  return PyArray_Return(idx_);
}

static PyMethodDef _assign_methods[] = {
  {"Assign", (PyCFunction)_Assign, METH_VARARGS, "Assignment problem."},
  {NULL, NULL, 0, NULL}
};

DL_EXPORT(void) init_assign(void)
{
  Py_InitModule3("_assign", _assign_methods, "Numpy wrapper for linear assignment problem.");
  import_array();
}
