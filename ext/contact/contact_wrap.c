#include <math.h>
#include "Python.h"
#include "contact.h"
#include <numpy/arrayobject.h>
#include <stdio.h>


extern PyObject *atomic_contact_wrap(PyObject *self, PyObject *args) {
    PyArrayObject *xyzlist_, *contacts_, *results_;
    int traj_length, num_contacts, num_atoms, num_dims, width_contacts;
    float *results;
    const float *xyzlist;
    const int *contacts;
    if (!PyArg_ParseTuple(args, "O!O!O!",
            &PyArray_Type, &xyzlist_, &PyArray_Type, &contacts_, &PyArray_Type, &results_)) {
        return 0;
    }
    else {
        xyzlist = (const float*) xyzlist_->data;
        contacts = (const int*) contacts_->data;
        results = (float*) results_->data;
        
        traj_length = xyzlist_->dimensions[0];
        num_atoms = xyzlist_->dimensions[1];
        num_dims = xyzlist_->dimensions[2];
        num_contacts = contacts_->dimensions[0];
        width_contacts = contacts_->dimensions[1];
        
        if ((num_dims != 3) || (width_contacts != 2)) {
            printf("Incorrect call to dihedrals_from_trajectory_wrap! Aborting");
            exit(1);
        }
        
        //printf("traj_length %d\n", traj_length);
        //printf("num_atoms %d\n", num_atoms);
        //printf("num_contacts %d\n", num_contacts);
        
        
        atomic_contact(xyzlist, contacts, num_contacts, traj_length, num_atoms, results);
    }
    return Py_BuildValue("d", 0.0);
}


extern PyObject *atomic_contact_rect_image_wrap(PyObject *self, PyObject *args) {
    PyArrayObject *xyzlist_, *box_, *contacts_, *results_;
    int traj_length, num_contacts, num_atoms, num_dims, width_contacts;
    float *results;
    const float *xyzlist;
    const float *box;
    const int *contacts;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
                          &PyArray_Type, &xyzlist_, &PyArray_Type, &box_, &PyArray_Type, &contacts_, &PyArray_Type, &results_)) {
        return 0;
    }
    else {
        xyzlist = (const float*) xyzlist_->data;
        box = (const float*) box_->data;
        contacts = (const int*) contacts_->data;
        results = (float*) results_->data;
        
        traj_length = xyzlist_->dimensions[0];
        num_atoms = xyzlist_->dimensions[1];
        num_dims = xyzlist_->dimensions[2];
        num_contacts = contacts_->dimensions[0];
        width_contacts = contacts_->dimensions[1];
        
        if ((num_dims != 3) || (width_contacts != 2)) {
            printf("Incorrect call to dihedrals_from_trajectory_wrap! Aborting");
            exit(1);
        }
        
        //printf("traj_length %d\n", traj_length);
        //printf("num_atoms %d\n", num_atoms);
        //printf("num_contacts %d\n", num_contacts);
        
        
        atomic_contact_rect_image(xyzlist, box, contacts, num_contacts, traj_length, num_atoms, results);
    }
    return Py_BuildValue("d", 0.0);
}



extern PyObject *closest_contact_wrap(PyObject *self, PyObject *args) {
    PyArrayObject *xyzlist_, *residues_, *atoms_per_residue_, *contacts_, *results_;
    int traj_length, num_atoms, num_dims;
    int num_residues, residue_width, num_residues2;
    int num_contacts, width_contacts;
    float *results;
    const float *xyzlist;
    const int *residues, *atoms_per_residue, *contacts;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!",
            &PyArray_Type, &xyzlist_, &PyArray_Type, &residues_,
            &PyArray_Type, &atoms_per_residue_, &PyArray_Type, &contacts_,
            &PyArray_Type, &results_)) {
        return 0;
    }
    else {
        xyzlist = (const float*) xyzlist_->data;
        residues = (const int*) residues_->data;
        atoms_per_residue = (const int*) atoms_per_residue_->data;
        contacts = (const int*) contacts_->data;
        results = (float*) results_->data;
        
        traj_length = xyzlist_->dimensions[0];
        num_atoms = xyzlist_->dimensions[1];
        num_dims = xyzlist_->dimensions[2];
        num_residues = residues_->dimensions[0];
        residue_width = residues_->dimensions[1];
        num_contacts = contacts_->dimensions[0];
        width_contacts = contacts_->dimensions[1];
        num_residues2 = atoms_per_residue_->dimensions[0];
        
        
        if ((num_dims != 3) || (width_contacts != 2)) {
            printf("Incorrect call to dihedrals_from_trajectory_wrap! Aborting");
            exit(1);
        }
        if (num_residues2 != num_residues) {
          printf("Bad news bears");
          exit(1);
        }
        
        //printf("traj_length %d\n", traj_length);
        //printf("num_atoms %d\n", num_atoms);
        //printf("num_contacts %d\n", num_contacts);

        closest_contact(xyzlist, residues, num_residues, residue_width,
                               atoms_per_residue, contacts, num_contacts, traj_length,
                               num_atoms, results);
    }
    return Py_BuildValue("d", 0.0);
}


static PyMethodDef _contactWrapMethods[] = {
  {"atomic_contact_wrap", atomic_contact_wrap, METH_VARARGS},
  {"atomic_contact_rect_image_wrap", atomic_contact_rect_image_wrap, METH_VARARGS},
  {"closest_contact_wrap", closest_contact_wrap, METH_VARARGS},
  {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

DL_EXPORT(void) init_contact_wrap(void) {
  Py_InitModule3("_contact_wrap", _contactWrapMethods, "Wrappers for contact map calculation.");
  import_array();
}

