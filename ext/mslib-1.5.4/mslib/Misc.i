/*
# AUTHOR         Michel F. SANNER
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appear in all copies and that
# both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of TSRI not be used in 
# advertising or publicity pertaining to distribution of the software 
# without specific, written prior permission.
#
# TSRI DISCLAIMS ALL WARRANTIES WITH REGARD TO
# THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS, IN NO EVENT SHALL TSRI BE LIABLE
# FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
# Copyright (C) Michel F. SANNER, TSRI, 1999
# Unpublished work.  All Rights Reserved.
# 
*/

/**************************************************************************
Type maps for

	char ** <==> list of strings


**************************************************************************/


/**************************************************************************

	char ** <==> list of strings

**************************************************************************/


/**************************************************************************

	Python file object  ==> FILE *

**************************************************************************/

// This tells SWIG to treat char ** as a special case
%typemap(in) char ** {
  /* Check if is a list */
  if ($input==Py_None) {
    $1=NULL;
  } else if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $1 = (char **) malloc((size+1)*sizeof(char *));
    for (i = 0; i < size; i++) {
      /* memory is allocated by PyList_GetItem */
      PyObject *o = PyList_GetItem($input,i);
      if (PyString_Check(o))
        $1[i] = PyString_AsString(PyList_GetItem($input,i));
      else {
        PyErr_SetString(PyExc_TypeError,"list must contain strings");
        free($1);
        return NULL;
      }
    }
    $1[i] = 0;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

// This cleans up the char ** array we malloc'd before the function call
%typemap(freearg) char ** {
  if ($1) {
    free((char *) $1);
  }
}

// This allows a C function to return a char ** as a Python list
%typemap(out) char ** {
  int len,i;
  len = 0;
  while ($1[len]) len++;
  $result = PyList_New(len);
  for (i = 0; i < len; i++) {
    PyList_SetItem($result,i,PyString_FromString($1[i]));
  }
}

// This allows a C function to receive Python File object
%typemap(in) FILE * {
  if (!PyFile_Check($input)) {
    PyErr_SetString(PyExc_ValueError, "Need a file!");
    return NULL;
  }
  $1 = PyFile_AsFile($input);
}
