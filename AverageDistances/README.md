# Trajectory Analysis

This trajectory was generated from a short REMD
simulation of CaM with virtual spin labels
attached. There was obviously some kind of
problem as the simulations eventualy crashed. I
think this was related to the ions. In any case,
This is sufficient to look at the distribution
of virtual spin label sites.

The file average_distances.dat contains the r^-6 averaged
distances based on a segment of the trajectory.

The script will also output a list of vmd selections for
residues that are within 1.2 nm. Based on this, it looks
like it should be possible to reconstruct the structure
using the data --- assuming that the NMR data is relatively
complete.

The file short.pdb is for testing purposes.
