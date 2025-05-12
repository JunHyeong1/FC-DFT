#include <stdlib.h>

void distance_calculator(double *coords, double *atom_coords, int natm, int ngrids, double *dist);

void grad_sas_drv(double *erf_list, double *er, double *x, double delta2, int ngrids, int natm, double *grad_sas);

void boundary_cond_drv(double kappa, double T, double eps, double eps_sam, double stern_sam, double bottom, double spacing, int ngrids, double *phi_z, double *slope);

void boundary_cond_gradient_drv(double *coords, double *atom_coords, double *atomic_radii,
                                double *phi_z, double *sas, double kappa, double T,
                                double probe, double delta2, double stern_sam, double spacing,
                                double slope, int ngrids, int natm, double *grad_bc, 
                                double *grad_phi_z, double *grad_sas);
