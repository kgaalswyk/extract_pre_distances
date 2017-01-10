#!/usr/bin/env python


from __future__ import print_function
import mdtraj as md
import numpy as np
from scipy.spatial.distance import cdist, squareform


# this offset makes the indexing match the
# nmr data
SEQ_OFFSET = 5

# this is the offset to match the PDB file
PDB_OFFSET = -4

# starting residue of peptide in trajectory
PEPTIDE_START_TRAJ = 146

# starting residue of peptide in NMR data
PEPTIDE_START_NMR = 202


def fix_peptide_indices(input_index):
    '''This will fix the indices of the peptide residues.

    It will convert from the indexing in the trajectory such
    that fix_peptide_indices(input_index) + SEQ_OFFSET = nmr_index.
    '''
    if input_index >= PEPTIDE_START_TRAJ:
        out = input_index + (PEPTIDE_START_NMR - PEPTIDE_START_TRAJ) - SEQ_OFFSET
    else:
        out = input_index
    return out


traj = md.load('trajectory.pdb')[100:]
# traj = md.load('short.pdb')

nitrogens = traj.topology.select('name N')
spin_labels = traj.topology.select('name OND')

distances = np.zeros((len(nitrogens), len(spin_labels), traj.n_frames))
for i in range(traj.n_frames):
    distances[:, :, i] = cdist(traj.xyz[i, nitrogens, :], traj.xyz[i, spin_labels, :])

# do 1/r^6 average
avg_distances = np.mean(distances**(-6.), axis=2)**(-1.0 / 6.0)


with open('average_distances.dat', 'w') as outfile:
    for j, spin_atom_ind in enumerate(spin_labels):
        close_atoms = []
        for i, N_atom_ind in enumerate(nitrogens):
            N_res = fix_peptide_indices(traj.topology.atom(N_atom_ind).residue.index) + SEQ_OFFSET
            res_type = traj.topology.atom(N_atom_ind).residue.name
            spin_res = traj.topology.atom(spin_atom_ind).residue.index + SEQ_OFFSET

            print('{}\t{}\t{}\t{}'.format(spin_res, N_res, res_type,
                                          avg_distances[i, j]), file=outfile)

            if avg_distances[i, j] < 1.2:
                close_atoms.append(str(N_res + PDB_OFFSET))

        print('(resid {} and name OND) or (resid {} and name N)'.format(
            spin_res + PDB_OFFSET,
            ' '.join(close_atoms)))
        print('')
