%init %{
	import_array(); /* load the Numeric PyCObjects */
%}

%{

#ifdef _MSC_VER
#include <windows.h>
#define WinVerMajor() LOBYTE(LOWORD(GetVersion()))
#endif

#include "numpy/arrayobject.h"
static PyArrayObject *contiguous_typed_array(PyObject *obj, int typecode,
                                      int expectnd, int *expectdims, int argnum)
{
  PyArrayObject *arr;
  int i, numitems, itemsize;
  char buf[255];

  /* if the shape and type are OK, this function increments the reference
     count and arr points to obj */
  if((arr = (PyArrayObject *)PyArray_ContiguousFromObject(obj,
                                                          typecode, 0,
                                                          10)) == NULL)
    {
      sprintf(buf,"Failed to make a contiguous array for argument arg%d of type %d\n", argnum, typecode);
      PyErr_SetString(PyExc_ValueError, buf);
      return NULL;
    }

  if(expectnd>0)
    {
      if(arr->nd > expectnd + 1 || arr->nd < expectnd)
        {
          Py_DECREF((PyObject *)arr);
          PyErr_SetString(PyExc_ValueError,
                          "Array has wrong number of dimensions");
          return NULL;
        }
      if(arr->nd == expectnd + 1)
        {
          if(arr->dimensions[arr->nd - 1] != 1)
            {
              Py_DECREF((PyObject *)arr);
              PyErr_SetString(PyExc_ValueError,
                              "Array has wrong number of dimensions");
              return NULL;
            }
        }
      if(expectdims)
        {
          for(i = 0; i < expectnd; i++)
            if(expectdims[i]>0)
              if(expectdims[i] != arr->dimensions[i])
                {
                  Py_DECREF((PyObject *)arr);
                  sprintf(buf,"The extent of dimension %d is %d while %d was expected\n",
                          i, arr->dimensions[i], expectdims[i]);
                  PyErr_SetString(PyExc_ValueError, buf);
                  return NULL;
                }
                  
        }
    }

  return arr;
}

%}


/**********************************************************/
/*                      OUTPUT                            */
/**********************************************************/
%{
static PyObject* l_output_helper2(PyObject* target, PyObject* o) {
    PyObject*   o2;
    PyObject*   o3;
    if (!target) {                   
        target = o;
    } else if (target == Py_None) {  
        Py_DECREF(Py_None);
        target = o;
    } else {                         
        if (!PyList_Check(target)) {
            o2 = target;
            target = PyList_New(0);
            PyList_Append(target, o2);
            Py_XDECREF(o2);
        }
        PyList_Append(target,o);
        Py_XDECREF(o);
    }
    return target;
}
%}
/**********************************************************/
/*                OUTPUT: int VECTOR, ARRAY             */
/**********************************************************/

%typemap(argout) int OUT_VECTOR[ANY],
		 int OUT_ARRAY2D[ANY][ANY]
{
   $result = l_output_helper2($result, (PyObject *)array$argnum);
}

%typemap(in, numinputs=0) int OUT_VECTOR[ANY] (PyArrayObject *array, int out_dims[1])
%{
  out_dims[0] = $1_dim0;
  $1 = (int *)malloc($1_dim0*sizeof(int));
  if ($1 == NULL) {
    PyErr_SetString(PyExc_ValueError, "failed to allocate memory");
    return NULL;
  }
  array = (PyArrayObject *)PyArray_FromDimsAndData(1, out_dims,
					PyArray_INT, (char *)$1);

#ifdef _MSC_VER
  switch ( WinVerMajor() )
  {
    case 6: break; // Vista
	default: array->flags |= NPY_OWNDATA;
  }
#else
  // so we'll free this memory when this
  // array will be garbage collected
  array->flags |= NPY_OWNDATA; 
#endif

%}

%typemap(in, numinputs=0) int OUT_ARRAY2D[ANY][ANY] (PyArrayObject *array, 
					           int out_dims[2])
{
  int *data = (int *)malloc($1_dim0*$1_dim1*sizeof(int));
  out_dims[0] = $1_dim0;
  out_dims[1] = $1_dim1;
  if (!data) {
    PyErr_SetString(PyExc_ValueError, "failed to allocate data array");
    return NULL;
  }
  array = (PyArrayObject *)PyArray_FromDimsAndData(2, out_dims,
						   PyArray_INT,
						   (char *)(data));

#ifdef _MSC_VER
  switch ( WinVerMajor() )
  {
    case 6: break; // Vista
	default: array->flags |= NPY_OWNDATA;
  }
#else
  // so we'll free this memory when this
  // array will be garbage collected
  array->flags |= NPY_OWNDATA; 
#endif

  $1 = (int (*)[$1_dim1])data;
}

/**********************************************************/
/*                      OUTPUT: float VECTOR, ARRAY       */
/**********************************************************/

%typemap(argout) float OUT_VECTOR[ANY],
		 float OUT_ARRAY2D[ANY][ANY]
{
   $result = l_output_helper2($result, (PyObject *)array$argnum);
}

%typemap(in, numinputs=0) float OUT_VECTOR[ANY](PyArrayObject *array, int out_dims[1])
%{
  out_dims[0] = $1_dim0;
  $1= (float *)malloc($1_dim0*sizeof(float));
  if ($1 == NULL) {
    PyErr_SetString(PyExc_ValueError, "failed to allocate memory");
    return NULL;
  }
  array = (PyArrayObject *)PyArray_FromDimsAndData(1, out_dims,
						PyArray_FLOAT, (char *)($1));

#ifdef _MSC_VER
  switch ( WinVerMajor() )
  {
    case 6: break; // Vista
	default: array->flags |= NPY_OWNDATA;
  }
#else
  // so we'll free this memory when this
  // array will be garbage collected
  array->flags |= NPY_OWNDATA; 
#endif

%}

%typemap(in, numinputs=0) float OUT_ARRAY2D[ANY][ANY] (PyArrayObject *array, 
					           int out_dims[2])
{
  float *data = (float *)malloc($1_dim0*$1_dim1*sizeof(float));
  out_dims[0] = $1_dim0;
  out_dims[1] = $1_dim1;
  if (!data) {
    PyErr_SetString(PyExc_ValueError, "failed to allocate data array");
    return NULL;
  }
  array = (PyArrayObject *)PyArray_FromDimsAndData(2, out_dims,
						   PyArray_FLOAT,
						   (char *)(data));

#ifdef _MSC_VER
  switch ( WinVerMajor() )
  {
    case 6: break; // Vista
	default: array->flags |= NPY_OWNDATA;
  }
#else
  // so we'll free this memory when this
  // array will be garbage collected
  array->flags |= NPY_OWNDATA; 
#endif

  $1 = (float (*)[$1_dim1])data;
}



/**********************************************************/
/*                 OUTPUT: double VECTOR, ARRAY              */
/**********************************************************/

%typemap(argout) double OUT_VECTOR[ANY],
		 double OUT_ARRAY2D[ANY][ANY]
{
   $result = l_output_helper2($result, (PyObject *)array$argnum);
}

%typemap(in, numinputs=0) double OUT_VECTOR[ANY] (PyArrayObject *array, int out_dims[1])
{
  out_dims[0] = $1_dim0;
  $1= (double *)malloc($1_dim0*sizeof(double));
  if ($1 == NULL) {
    PyErr_SetString(PyExc_ValueError, "failed to allocate memory");
    return NULL;
  }
  array = (PyArrayObject *)PyArray_FromDimsAndData(1, out_dims,
						PyArray_DOUBLE, (char *)($1));

#ifdef _MSC_VER
  switch ( WinVerMajor() )
  {
    case 6: break; // Vista
	default: array->flags |= NPY_OWNDATA;
  }
#else
  // so we'll free this memory when this
  // array will be garbage collected
  array->flags |= NPY_OWNDATA; 
#endif

}

%typemap(in, numinputs=0) double OUT_ARRAY2D[ANY][ANY] (PyArrayObject *array, 
					           int out_dims[2])
{
  double *data = (double *)malloc($1_dim0*$1_dim1*sizeof(double));
  out_dims[0] = $1_dim0;
  out_dims[1] = $1_dim1;
  if (!data) {
    PyErr_SetString(PyExc_ValueError, "failed to allocate data array");
    return NULL;
  }
  array = (PyArrayObject *)PyArray_FromDimsAndData(2, out_dims,
						   PyArray_DOUBLE,
						   (char *)(data));

#ifdef _MSC_VER
  switch ( WinVerMajor() )
  {
    case 6: break; // Vista
	default: array->flags |= NPY_OWNDATA;
  }
#else
  // so we'll free this memory when this
  // array will be garbage collected
  array->flags |= NPY_OWNDATA; 
#endif

  $1 = (double (*)[$1_dim1])data;
}



/*************************************************************/
/*                      INPUT                                */ 
/*************************************************************/


/*************************************************************/
/*                  INPUT: u_char VECTOR                     */
/*************************************************************/


%typemap(in) u_char VECTOR[ANY] (PyArrayObject *array, int expected_dims[1])
%{
if ($input != Py_None)
  {
    expected_dims[1] = $1_dim0;
    if (expected_dims[0]==1) expected_dims[0]=0;
    array = contiguous_typed_array($input, PyArray_UBYTE, 1, expected_dims, $argnum);
    if (! array) return NULL;
    $1 = (u_char *)array->data;
  }
 else
   {
     array = NULL;
     $1 = NULL;
   }
%}

%typemap(freearg) u_char VECTOR[ANY]

%{
  if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
   
%}

/*************************************************************/
/*                  INPUT: u_char ARRAY2D                    */
/*************************************************************/


%typemap(in) u_char ARRAY2D[ANY][ANY](PyArrayObject *array,
			            	  int expected_dims[2]) 
%{
  if ($input != Py_None)
  {
    expected_dims[0] =  $1_dim0;
    expected_dims[1] = $1_dim1;
    if (expected_dims[0]==1) expected_dims[0]=0;
    if (expected_dims[1]==1) expected_dims[1]=0;
    array = contiguous_typed_array($input, PyArray_UBYTE, 2, expected_dims, $argnum);
    if (! array) return NULL;
    $1 = (u_char (*)[$1_dim1])array->data;
 }
else
  {
   array = NULL;
   $1 = NULL;
  }
%}

%typemap(freearg) u_char ARRAY2D[ANY][ANY]
%{
   if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
%}

/*************************************************************/
/*                  INPUT: int VECTOR	                     */
/*************************************************************/

%typemap(in) int VECTOR[ANY] (PyArrayObject *array, int expected_dims[1])
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $1_dim0;
    if (expected_dims[0]==1) expected_dims[0]=0;
    array = contiguous_typed_array($input, PyArray_INT, 1, expected_dims,$argnum);
    if (! array) return NULL;
    $1 = (int *)array->data;
  }
  else
    {
      array = NULL;
      $1 = NULL;
    }
%}

%typemap(freearg) int VECTOR[ANY]

%{
  if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
   
%}

/*************************************************************/
/*                  INPUT: int ARRAY2D                       */
/*************************************************************/

%typemap(in) int ARRAY2D[ANY][ANY](PyArrayObject *array,
			            	  int expected_dims[2]) 
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $1_dim0;
    expected_dims[1] = $1_dim1;
    if (expected_dims[0]==1) expected_dims[0]=0;
    if (expected_dims[1]==1) expected_dims[1]=0;
    array = contiguous_typed_array($input, PyArray_INT, 2, expected_dims,$argnum);
    if (! array) return NULL;
    $1 = (int (*)[$1_dim1])array->data;
  }
  else
  {
   array = NULL;
   $1 = NULL;
  }
%}
  
%typemap(freearg) int ARRAY2D[ANY][ANY]
%{
   if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
%}

/*************************************************************/
/*                  INPUT: float VECTOR                     */
/*************************************************************/

%typemap(in) float VECTOR[ANY] (PyArrayObject *array, int expected_dims[1])
%{
  if ($input != Py_None)
  {
  expected_dims[0] = $1_dim0;
  if (expected_dims[0]==1) expected_dims[0]=0;
  array = contiguous_typed_array($input, PyArray_FLOAT, 1, expected_dims, $argnum);
  if (! array) return NULL;
  $1 = (float *)array->data;
 }
else
  {
   array = NULL;
   $1 = NULL;
  }
%}

%typemap(freearg) float VECTOR[ANY]

%{
  if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
   
%}

/*************************************************************/
/*                  INPUT: float ARRAY2D                     */
/*************************************************************/

%typemap(in) float ARRAY2D[ANY][ANY](PyArrayObject *array,
			            	  int expected_dims[2]) 
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $1_dim0;
    expected_dims[1] = $1_dim1;
    if (expected_dims[0]==1) expected_dims[0]=0;
    if (expected_dims[1]==1) expected_dims[1]=0;
    array = contiguous_typed_array($input, PyArray_FLOAT, 2, expected_dims, $argnum);
    if (! array) return NULL;
    $1 = (float (*)[$1_dim1])array->data;
  }
  else
  {
   array = NULL;
   $1 = NULL;
  }
%}
  
%typemap(freearg) float ARRAY2D[ANY][ANY]
%{
   if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
%}

/*******************************************************/
/**              Input: float ARRAY2D_NULL            **/
/*******************************************************/

%typemap(in) float ARRAY2D_NULL[ANY][ANY] (PyArrayObject *array ,
					   int expected_dims[2])

 {
  if ($input == Py_None) {
    $1 = NULL;
  } else {
    expected_dims[0] = $1_dim0;
    expected_dims[1] = $1_dim1;
    if (expected_dims[0]==1) expected_dims[0]=0;
    if (expected_dims[1]==1) expected_dims[1]=0;
    array = contiguous_typed_array($input, PyArray_FLOAT, 2, expected_dims, $argnum);
    if (! array) return NULL;
    $1 = (float (*)[$1_dim1])array->data;
  }
}
 
/*************************************************************/
/*                  INPUT: double VECTOR                     */
/*************************************************************/

%typemap(in) double VECTOR[ANY] (PyArrayObject *array, int expected_dims[1])
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $1_dim0;
    if (expected_dims[0]==1) expected_dims[0]=0;
    array = contiguous_typed_array($input, PyArray_DOUBLE, 1, expected_dims,$argnum);
    if (! array) return NULL;
    $1 = (double *)array->data;
  }
  else
  {
    array = NULL;
    $1 = NULL;
  }
  %}

%typemap(freearg) double VECTOR[ANY]

%{
  if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
   
%}

/*************************************************************/
/*                  INPUT: double ARRAY2D                    */
/*************************************************************/

%typemap(in) double ARRAY2D[ANY][ANY](PyArrayObject *array,
			            	  int expected_dims[2]) 
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $1_dim0;
    expected_dims[1] = $1_dim1;
    if (expected_dims[0]==1) expected_dims[0]=0;
    if (expected_dims[1]==1) expected_dims[1]=0;
    array = contiguous_typed_array($input, PyArray_DOUBLE, 2, expected_dims,$argnum);
    if (! array) return NULL;
    $1 = (double (*)[$1_dim1])array->data;
  }
  else
  {
   array = NULL;
   $1 = NULL;
  }
%}
  
%typemap(freearg) double ARRAY2D[ANY][ANY]
%{
   if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
%}

%define UCHAR_ARRAY2D( DIM, ARRAYNAME, ARRAYSHAPE)
%typemap(in) ( int* DIM, u_char ARRAYNAME##ARRAYSHAPE) (PyArrayObject *array,
			            		  int expected_dims[2], int intdims[2])
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $2_dim0;
    expected_dims[1] = $2_dim1;
    if (expected_dims[0]==1) expected_dims[0]=0;
    if (expected_dims[1]==1) expected_dims[1]=0;
    array = contiguous_typed_array($input, PyArray_UBYTE, 2, expected_dims,$argnum);
    if (! array) return NULL;
	  intdims[0] = ((PyArrayObject *)(array))->dimensions[0];
	  intdims[1] = ((PyArrayObject *)(array))->dimensions[1];
	  $1 = intdims;
    $2 = (u_char (*)[$2_dim1])array->data;
  }
  else
  {
   array = NULL;
   $2 = NULL;
   $1 = NULL;
  }

%}

%typemap(freearg) (int* DIM, u_char ARRAYNAME##ARRAYSHAPE) %{
%{
   if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
%}
%enddef

/**************************************************************/
/*** Macros to generate typemaps for pairs of arguments *******/
/**************************************************************/

/**************************************************************/
/*                Input: FLOAT_ARRAY4D                        */
/**************************************************************/

%define FLOAT_ARRAY4D(DIM, ARRAYNAME, ARRAYSHAPE)
%typemap(in) (int *DIM, float ARRAYNAME##ARRAYSHAPE)(PyArrayObject *array,
			            	  int expected_dims[4], int intdims[4]) 
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $2_dim0;
    expected_dims[1] = $2_dim1;
    expected_dims[2] = $2_dim2;
    expected_dims[3] = $2_dim3;
    if (expected_dims[0]==1) expected_dims[0]=0;
    if (expected_dims[1]==1) expected_dims[1]=0;
    if (expected_dims[2]==1) expected_dims[2]=0;
    if (expected_dims[3]==1) expected_dims[3]=0;
    array = contiguous_typed_array($input, PyArray_FLOAT, 4, expected_dims,$argnum);
    if (! array) return NULL;
    $2 = (float(*)[$2_dim1][$2_dim2][$2_dim3])array->data;
	  intdims[0] = ((PyArrayObject *)(array))->dimensions[0];
	  intdims[1] = ((PyArrayObject *)(array))->dimensions[1];
	  intdims[2] = ((PyArrayObject *)(array))->dimensions[2];
	  intdims[3] = ((PyArrayObject *)(array))->dimensions[3];
	  $1 = intdims;
  }
  else
  {
   array = NULL;
   $2 = NULL;
   $1 = NULL;
  }
%}
  
%typemap(freearg) (int *DIM, float ARRAYNAME##ARRAYSHAPE)

%{
   if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
%}
%enddef

/**************************************************************/
/*                Input: SHORT_ARRAY4D                        */
/**************************************************************/

%define SHORT_ARRAY4D( DIM, ARRAYNAME, ARRAYSHAPE)
%typemap(in) ( int *DIM, short ARRAYNAME##ARRAYSHAPE)(PyArrayObject *array,
			            	  int expected_dims[4], int intdims[4]) 
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $2_dim0;
    expected_dims[1] = $2_dim1;
    expected_dims[2] = $2_dim2;
    expected_dims[3] = $2_dim3;
    if (expected_dims[0]==1) expected_dims[0]=0;
    if (expected_dims[1]==1) expected_dims[1]=0;
    if (expected_dims[2]==1) expected_dims[2]=0;
    if (expected_dims[3]==1) expected_dims[3]=0;
    array = contiguous_typed_array($input, PyArray_SHORT, 4, expected_dims,$argnum);
    if (! array) return NULL;
    $2 = (short(*)[$2_dim1][$2_dim2][$2_dim3])array->data;
	  intdims[0] = ((PyArrayObject *)(array))->dimensions[0];
	  intdims[1] = ((PyArrayObject *)(array))->dimensions[1];
	  intdims[2] = ((PyArrayObject *)(array))->dimensions[2];
	  intdims[3] = ((PyArrayObject *)(array))->dimensions[3];
	  $1 = intdims;
  }
  else
  {
   array = NULL;
   $2 = NULL;
   $1 = NULL;
  }
%}
  
%typemap(freearg) (int *DIM, short ARRAYNAME##ARRAYSHAPE)

%{
   if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
%}
%enddef

/**************************************************************/
/*                Input: UCHAR_ARRAY4D                        */
/**************************************************************/

%define UCHAR_ARRAY4D( DIM, ARRAYNAME, ARRAYSHAPE)
%typemap(in) (int *DIM, u_char ARRAYNAME##ARRAYSHAPE)(PyArrayObject *array,
			            	  int expected_dims[4], int intdims[4]) 
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $2_dim0;
    expected_dims[1] = $2_dim1;
    expected_dims[2] = $2_dim2;
    expected_dims[3] = $2_dim3;
    if (expected_dims[0]==1) expected_dims[0]=0;
    if (expected_dims[1]==1) expected_dims[1]=0;
    if (expected_dims[2]==1) expected_dims[2]=0;
    if (expected_dims[3]==1) expected_dims[3]=0;
    array = contiguous_typed_array($input, PyArray_UBYTE, 4, expected_dims,$argnum);
    if (! array) return NULL;
    $2 = (u_char(*)[$2_dim1][$2_dim2][$2_dim3])array->data;
	  intdims[0] = ((PyArrayObject *)(array))->dimensions[0];
	  intdims[1] = ((PyArrayObject *)(array))->dimensions[1];
	  intdims[2] = ((PyArrayObject *)(array))->dimensions[2];
	  intdims[3] = ((PyArrayObject *)(array))->dimensions[3];
	  $1 = intdims;
  }
  else
  {
   array = NULL;
   $2 = NULL;
   $1 = NULL;
  }
%}
  
%typemap(freearg) (int *DIM, u_char ARRAYNAME##ARRAYSHAPE)

%{
   if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
%}
%enddef

%define SHORT_ARRAY5D(DIM, ARRAYNAME, ARRAYSHAPE)
%typemap(in) (int *DIM, short ARRAYNAME##ARRAYSHAPE)(PyArrayObject *array,
			            	  int expected_dims[5], int intdims[5]) 
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $2_dim0;
    expected_dims[1] = $2_dim1;
    expected_dims[2] = $2_dim2;
    expected_dims[3] = $2_dim3;
    expected_dims[4] = $2_dim4;
    if (expected_dims[0]==1) expected_dims[0]=0;
    if (expected_dims[1]==1) expected_dims[1]=0;
    if (expected_dims[2]==1) expected_dims[2]=0;
    if (expected_dims[3]==1) expected_dims[3]=0;
    if (expected_dims[4]==1) expected_dims[4]=0;
    array = contiguous_typed_array($input, PyArray_SHORT, 5, expected_dims,$argnum);
    if (! array) return NULL;
    $2 = (short(*)[$2_dim1][$2_dim2][$2_dim3][$2_dim4])array->data;
	  intdims[0] = ((PyArrayObject *)(array))->dimensions[0];
	  intdims[1] = ((PyArrayObject *)(array))->dimensions[1];
	  intdims[2] = ((PyArrayObject *)(array))->dimensions[2];
	  intdims[3] = ((PyArrayObject *)(array))->dimensions[3];
	  intdims[4] = ((PyArrayObject *)(array))->dimensions[4];
	  $1 = intdims;
  }
  else
  {
   array = NULL;
   $2 = NULL;
   $1 = NULL;
  }
%}
  
%typemap(freearg) (int *DIM, short ARRAYNAME##ARRAYSHAPE)

%{
   if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
%}
%enddef

/**************************************************************/
/*                Input: UCHAR_ARRAY5D                        */
/**************************************************************/

%define UCHAR_ARRAY5D(DIM, ARRAYNAME, ARRAYSHAPE)
%typemap(in) (int *DIM, u_char ARRAYNAME##ARRAYSHAPE)(PyArrayObject *array,
			            	  int expected_dims[5], int intdims[5]) 
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $2_dim0;
    expected_dims[1] = $2_dim1;
    expected_dims[2] = $2_dim2;
    expected_dims[3] = $2_dim3;
    expected_dims[4] = $2_dim4;
    if (expected_dims[0]==1) expected_dims[0]=0;
    if (expected_dims[1]==1) expected_dims[1]=0;
    if (expected_dims[2]==1) expected_dims[2]=0;
    if (expected_dims[3]==1) expected_dims[3]=0;
    if (expected_dims[4]==1) expected_dims[4]=0;
    array = contiguous_typed_array($input, PyArray_UBYTE, 5, expected_dims,$argnum);
    if (! array) return NULL;
    $2 = (u_char(*)[$2_dim1][$2_dim2][$2_dim3][$2_dim4])array->data;
	  intdims[0] = ((PyArrayObject *)(array))->dimensions[0];
	  intdims[1] = ((PyArrayObject *)(array))->dimensions[1];
	  intdims[2] = ((PyArrayObject *)(array))->dimensions[2];
	  intdims[3] = ((PyArrayObject *)(array))->dimensions[3];
	  intdims[4] = ((PyArrayObject *)(array))->dimensions[4];
	  $1 = intdims;
  }
  else
  {
   array = NULL;
   $2 = NULL;
   $1 = NULL;
  }
%}
  
%typemap(freearg) (int *DIM, u_char ARRAYNAME##ARRAYSHAPE)

%{
   if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
%}
%enddef

/**************************************************************/
/*                Input: FLOAT_ARRAY5D                        */
/**************************************************************/

%define FLOAT_ARRAY5D( DIM, ARRAYNAME, ARRAYSHAPE)
%typemap(in) ( int* DIM, float ARRAYNAME##ARRAYSHAPE)(PyArrayObject *array,
			            	  int expected_dims[5], int intdims[5]) 
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $2_dim0;
    expected_dims[1] = $2_dim1;
    expected_dims[2] = $2_dim2;
    expected_dims[3] = $2_dim3;
    expected_dims[4] = $2_dim4;
    if (expected_dims[0]==1) expected_dims[0]=0;
    if (expected_dims[1]==1) expected_dims[1]=0;
    if (expected_dims[2]==1) expected_dims[2]=0;
    if (expected_dims[3]==1) expected_dims[3]=0;
    if (expected_dims[4]==1) expected_dims[4]=0;
    array = contiguous_typed_array($input, PyArray_FLOAT, 5, expected_dims,$argnum);
    if (! array) return NULL;
    $2 = (float(*)[$2_dim1][$2_dim2][$2_dim3][$2_dim4])array->data;
	  intdims[0] = ((PyArrayObject *)(array))->dimensions[0];
	  intdims[1] = ((PyArrayObject *)(array))->dimensions[1];
	  intdims[2] = ((PyArrayObject *)(array))->dimensions[2];
	  intdims[3] = ((PyArrayObject *)(array))->dimensions[3];
	  intdims[4] = ((PyArrayObject *)(array))->dimensions[4];
	  $1 = intdims;
  }
  else
  {
   array = NULL;
   $2 = NULL;
   $1 = NULL;
  }
%}
  
%typemap(freearg) (int* DIM, float ARRAYNAME##ARRAYSHAPE)

%{
   if ( array$argnum )
      Py_DECREF((PyObject *)array$argnum);
%}
%enddef

/**************************************************************/
/*                Input: INT_VECTOR                           */
/**************************************************************/

%define INT_VECTOR( ARRAYNAME, ARRAYSHAPE, LENGTH )
%typemap(in) (int ARRAYNAME##ARRAYSHAPE, int LENGTH) (PyArrayObject *array, 
						      int expected_dims[1]) 
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $1_dim0;
    if (expected_dims[0]==1) expected_dims[0]=0;
    
    array = contiguous_typed_array($input, PyArray_INT, 1, expected_dims,$argnum);
    if (! array) return NULL;
    $1 = (int *)array->data;
    $2 = ((PyArrayObject *)(array))->dimensions[0];
  }
  else
  {
    array = NULL;
    $1 = NULL;
    $2 = 0;
  }
%}

%typemap(freearg) (int ARRAYNAME##ARRAYSHAPE, int LENGTH) %{
   if (array$argnum)
     Py_DECREF((PyObject *)array$argnum);
%}
%enddef

/**************************************************************/
/*                Input: FLOAT_VECTOR                         */
/**************************************************************/

%define FLOAT_VECTOR( ARRAYNAME, ARRAYSHAPE, LENGTH )
%typemap(in) (float ARRAYNAME##ARRAYSHAPE, int LENGTH) (PyArrayObject *array, 
						      int expected_dims[1]) 
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $1_dim0;
    if (expected_dims[0]==1) expected_dims[0]=0;
    array = contiguous_typed_array($input, PyArray_FLOAT, 1, expected_dims,$argnum);
    if (! array) return NULL;
    $1 = (float *)array->data;
    $2 = ((PyArrayObject *)(array))->dimensions[0];
  }
  else
  {
    array = NULL;
    $1 = NULL;
    $2 = 0;
  }
%}

%typemap(freearg) (float ARRAYNAME##ARRAYSHAPE, int LENGTH) %{
   if (array$argnum)
     Py_DECREF((PyObject *)array$argnum);
%}
%enddef


/**************************************************************/
/*                Input: INT_ARRAY2D                          */
/**************************************************************/

%define INT_ARRAY2D( ARRAYNAME, ARRAYSHAPE, DIMENSIONS )
%typemap(in) ( int ARRAYNAME##ARRAYSHAPE,  int* DIMENSIONS)(PyArrayObject *array, 
						       int expected_dims[2], int intdims[2])
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $1_dim0;
    expected_dims[1] = $1_dim1;
    if (expected_dims[0]==1) expected_dims[0]=0;
    if (expected_dims[1]==1) expected_dims[1]=0;
    array = contiguous_typed_array($input, PyArray_INT, 2, expected_dims,$argnum);
    if (! array) return NULL;
    $1 = (int (*)[$1_dim1])array->data;
	  intdims[0] = ((PyArrayObject *)(array))->dimensions[0];
	  intdims[1] = ((PyArrayObject *)(array))->dimensions[1];
	  $2 = intdims;
  }
  else
  {
    array = NULL;
    $1 = NULL;
    $2 = 0;
  }
%}

%typemap(freearg) (int ARRAYNAME##ARRAYSHAPE, int* DIMENSIONS) %{
   if (array$argnum)
      Py_DECREF((PyObject *)array$argnum);
%}

%enddef

/**************************************************************/
/*                Input: FLOAT_ARRAY2D                        */
/**************************************************************/

%define FLOAT_ARRAY2D( ARRAYNAME, ARRAYSHAPE, DIMENSIONS )
%typemap(in) ( float ARRAYNAME##ARRAYSHAPE,  int* DIMENSIONS)(PyArrayObject *array, 
                                                        int expected_dims[2]
                                                        , int intdims[2])
%{
  if ($input != Py_None)
  {
    expected_dims[0] = $1_dim0;
    expected_dims[1] = $1_dim1;
    if (expected_dims[0]==1) expected_dims[0]=0;
    if (expected_dims[1]==1) expected_dims[1]=0;
    array = contiguous_typed_array($input, PyArray_FLOAT, 2, expected_dims,$argnum);
    if (! array) return NULL;
    $1 = (float (*)[$1_dim1])array->data;
	  intdims[0] = ((PyArrayObject *)(array))->dimensions[0];
	  intdims[1] = ((PyArrayObject *)(array))->dimensions[1];
	  $2 = intdims;
  }
  else
  { 
    array = NULL;
    $1 = NULL;
    $2 = 0;
  }
%}

%typemap(freearg) (float ARRAYNAME##ARRAYSHAPE, int* DIMENSIONS ) %{
   if (array$argnum )
      Py_DECREF((PyObject *)array$argnum);
%}

%enddef

