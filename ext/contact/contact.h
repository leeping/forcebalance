#ifndef _CPY_CONTACT_H
#define _CPY_CONTACT_H

void atomic_contact(const float *xyzlist, const int *contacts, int num_contacts, int traj_length, int num_atoms, float *results);
void atomic_contact_rect_image(const float *xyzlist, const float *box, const int *contacts, int num_contacts, int traj_length, int num_atoms, float *results);
void closest_contact(const float *xyzlist, const int *residues, const int num_residues, const int residue_width, const int* atoms_per_residue,  const int *contacts, int num_contacts, int traj_length, int num_atoms, float *results);

#endif
