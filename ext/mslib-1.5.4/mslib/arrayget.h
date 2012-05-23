#ifndef __ARRAY_H__

/****************************************************************

  PYTHON ARRAY ACCESS FUNCTIONS

******************************************************************/

#define ArrayData_GetMember2D(arr, dims, i, j)        arr[(i * dims[1]) + j]
#define ArrayData_GetMember3D(arr, dims, i, j, k)     arr[(i * dims[1] * dims[2]) + (j * dims[2]) + k]
#define ArrayData_GetMember4D(arr, dims, i, j, k, l)  arr[(i * dims[1] * dims[2] * dims[3]) + (j * dims[2] * dims[3]) + (k * dims[3]) + l]

int Array_GetData(PyObject *obj, void *data, int typecode, int checkdims, 
		  int nd, int *dims)
{
  PyArrayObject *arr;
  int i, numitems;

  if((arr = (PyArrayObject *)PyArray_ContiguousFromObject(obj,
							  typecode, 0,
							  10)) == NULL)
    return 0;

  if(checkdims)
    if(nd != arr->nd)
      {
	Py_DECREF((PyObject *)arr);
	return 0;
      }

  for(i = 0, numitems = 1; i < nd; i++)
    {
      if(checkdims)
	if(arr->dimensions[i] != dims[i])
	  {
	    Py_DECREF((PyObject *)arr);
	    return 0;
	  }
      numitems *= arr->dimensions[i];
    }

  memcpy(data, (void *)arr->data, arr->descr->elsize * numitems);
  Py_DECREF((PyObject *)arr);
  return 1;
}

int Array_DupData(PyObject *obj, void **data, int typecode,
		  int *nd, int **dims, int expectnd, int *expectdims)
{
  PyArrayObject *arr;
  int i, numitems, itemsize;

  if((arr = (PyArrayObject *)PyArray_ContiguousFromObject(obj,
							  typecode, 0,
							  10)) == NULL)
    return 0;

  *nd = arr->nd;
  if(expectnd)
    {
      if(arr->nd > expectnd + 1 || arr->nd < expectnd)
	{
	  Py_DECREF((PyObject *)arr);
	  return 0;
	}
      if(arr->nd == expectnd + 1)
	{
	  if(arr->dimensions[arr->nd - 1] != 1)
	    {
	      Py_DECREF((PyObject *)arr);
	      return 0;
	    }
	  else
	    *nd -= 1;
	}
      if(expectdims)
	{
	  for(i = 0; i < expectnd; i++)
	    if(expectdims[i]>0)
	      if(expectdims[i] != arr->dimensions[i])
		{
		  Py_DECREF((PyObject *)arr);
		  return 0;
		}
		  
	}
    }

  *dims = (int *)malloc(sizeof(int) * arr->nd);
  if(!(*dims))
    {
      Py_DECREF((PyObject *)arr);
      return 0;
    }

  for(i = 0, numitems = 1; i < arr->nd; i++)
    {
      numitems *= arr->dimensions[i];
      (*dims)[i] = arr->dimensions[i];
    }

  itemsize = arr->descr->elsize;
  *data = (void *)malloc(itemsize * numitems);

  if(!(*data))
    {
      free((void *)(*dims));
      Py_DECREF((PyObject *)arr);
      return 0;
    }

  memcpy(*data, (void *)arr->data, itemsize * numitems);

  Py_DECREF((PyObject *)arr);
  return 1;
}

/**************************************************************************

   GENERAL HELP FUNCTIONS

**************************************************************************/

static void print_Dict(PyObject *dict)
{
  PyObject *klist, *vlist, *member;
  int i, j;

  klist = PyDict_Keys(dict);
  vlist = PyDict_Values(dict);

  for(i = 0; i < PyList_Size(klist); i++)
    {
      member = PyList_GetItem(vlist, i);

      if(PyList_Check(member))
	for(j = 0; j < PyList_Size(member); j++)
	  {
	    printf("  %s: %s\n",
		   PyString_AsString(PyList_GetItem(klist, i)),
		   PyString_AsString(PyObject_Repr(PyList_GetItem(member,
								  j))));
	  }
      else
	printf("  %s: %s\n",
	       PyString_AsString(PyList_GetItem(klist, i)),
	       PyString_AsString(PyObject_Repr(PyList_GetItem(vlist, i))));
    }
}

#endif
