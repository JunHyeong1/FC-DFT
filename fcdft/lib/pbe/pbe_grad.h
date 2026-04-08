#include <stdlib.h>

void nuc_grad_eps_drv(double *erf_list, double *grad_list, double *eps_z, double delta2, int ngrids, int natm, double *nuc_grad_eps);

void grad_nuc_grad_eps_drv(double *erf_list, double *exp_list, double *er, double *x, double *eps_z, double *exp_z, double *dist, double delta1, double delta2, double eps_bulk, double eps_sam, int ngrids, int natm, double *grad_nuc_grad_eps);