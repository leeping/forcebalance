#include <stdio.h>
#include <float.h>
#include <math.h>


inline float sqeuclidean3(const float a[], const float b[]) {
  //Calculate the dot product between length-three vectors b and c
  return (a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1]) + (a[2] - b[2])*(a[2] - b[2]) ;
}

void atomic_contact(const float *xyzlist, const int *contacts, int num_contacts,
                    int traj_length, int num_atoms, float *results) {
  // For each length-2 row of contacts, compute the distance between the atoms with indices
  // in the first and second entries
  int i, j;
  const int* atom_ind;
  const float *frame, *atom1, *atom2;
  float *results_ptr;
  
  #pragma omp parallel for default(none) shared(results, xyzlist, contacts, num_contacts, num_atoms, traj_length) private(j, atom_ind, frame, atom1, atom2, results_ptr)
  for (i = 0; i < traj_length; i++) {
    frame = (const float*) xyzlist + num_atoms * 3 * i;
    results_ptr = results + num_contacts * i;
    atom_ind = contacts;
    for (j = 0; j < num_contacts; j++, results_ptr++, atom_ind = atom_ind + 2) {
      //indices of the two atoms
      atom1 = frame + *(atom_ind) * 3;
      atom2 = frame + *(atom_ind + 1) * 3;
      *results_ptr = sqrt(sqeuclidean3(atom1, atom2));
    }
  }
}

inline float sqeuclidean3_rect_image(const float a[], const float b[], const float box[]) {
    float dx, dy, dz;
    dx = (a[0] - b[0]);
    dy = (a[1] - b[1]);
    dz = (a[2] - b[2]);
    if (dx < -0.5*box[0]) {dx = dx + box[0];}
    if (dy < -0.5*box[1]) {dy = dy + box[1];}
    if (dz < -0.5*box[2]) {dz = dz + box[2];}
    if (dx >= 0.5*box[0]) {dx = dx - box[0];}
    if (dy >= 0.5*box[1]) {dy = dy - box[1];}
    if (dz >= 0.5*box[2]) {dz = dz - box[2];}
  //Calculate the dot product between length-three vectors b and c
    return dx*dx + dy*dy + dz*dz;
}

void atomic_contact_rect_image(const float *xyzlist, const float *box, 
                               const int *contacts, int num_contacts,
                               int traj_length, int num_atoms, float *results) {
  // For each length-2 row of contacts, compute the distance between the atoms with indices
  // in the first and second entries (uses minimum image convention for rectangular cells)
  int i, j;
  const int* atom_ind;
  const float *frame, *atom1, *atom2;
  float *results_ptr;
  
#pragma omp parallel for default(none) shared(results, xyzlist, box, contacts, num_contacts, num_atoms, traj_length) private(j, atom_ind, frame, atom1, atom2, results_ptr)
  for (i = 0; i < traj_length; i++) {
    frame = (const float*) xyzlist + num_atoms * 3 * i;
    results_ptr = results + num_contacts * i;
    atom_ind = contacts;
    for (j = 0; j < num_contacts; j++, results_ptr++, atom_ind = atom_ind + 2) {
      //indices of the two atoms
      atom1 = frame + *(atom_ind) * 3;
      atom2 = frame + *(atom_ind + 1) * 3;
      *results_ptr = sqrt(sqeuclidean3_rect_image(atom1, atom2, box));
    }
  }
}

void closest_contact(const float *xyzlist, const int *residues,
                            const int num_residues, const int residue_width,
                            const int* atoms_per_residue, 
                            const int *contacts, int num_contacts, int traj_length,
                            int num_atoms, float *results) {
  /*
  xyzlist - traj_length x num_atoms x 3
  residue_atoms - num_residues x residue_width, but only the first num_residue_atoms are used
  num_residue_atoms - num_residues x 1 -- max column index ofresidue_atoms that we care about (rest is padding)
  contacts - num_contacts x 2 -- each row is the indices of the RESIDUES who we monitor for contact
  results traj_length x num_contacts
  */
  
  int i, j, k, l, max_k, max_l;
  int *atom0_ind_ptr, *atom1_ind_ptr, *a1_ind_ptr;
  int *contact_ptr;
  float min, curr;
  float *results_ptr;
  const float *frame, *atom0, *atom1;
  
  #pragma omp parallel for default(none) shared(results, xyzlist, contacts, num_contacts, num_atoms, traj_length, residues, atoms_per_residue) private(j, k, l, max_k, max_l, atom0_ind_ptr, atom1_ind_ptr, a1_ind_ptr, contact_ptr, min, curr, results_ptr, frame, atom0, atom1)
  for (i = 0; i < traj_length; i++) {
    frame = (const float*) xyzlist + num_atoms * 3 * i;
    contact_ptr = (int*) contacts;
    results_ptr = results + num_contacts * i;
    for (j = 0; j < num_contacts; j++, contact_ptr += 2, results_ptr++) {
      //Calculate the distance between each atom in residue_atoms[contacts[j,0]]
      //and residue_atoms[contacts[j,1]]
      atom0_ind_ptr = (int*) residues + *(contact_ptr) * residue_width;
      atom1_ind_ptr = (int*) residues + *(contact_ptr + 1) * residue_width;
      
      max_k = *(atoms_per_residue + *(contact_ptr));
      max_l = *(atoms_per_residue + *(contact_ptr + 1));
      min = FLT_MAX;
      for (k = 0; k < max_k; k++, atom0_ind_ptr++) {
        a1_ind_ptr = atom1_ind_ptr;
        for (l = 0; l < max_l; l++, a1_ind_ptr++ ) {
          //printf("Comparing atoms %d and %d\n", *atom0_ind_ptr, *a1_ind_ptr);
          atom0 = frame + *(atom0_ind_ptr) * 3;
          atom1 = frame + *(a1_ind_ptr) * 3;
          //printf("With x coords %f, %f\n", *atom0, *atom1);
          curr = sqeuclidean3(atom0, atom1);
          min = curr < min ? curr : min;
        }
      }
      //printf("Min is %f\n", min);
      *results_ptr = sqrt(min);
    }
    //printf("Next frame\n");
    
  }
}
