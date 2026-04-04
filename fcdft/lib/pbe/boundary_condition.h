#include <stdlib.h>

void grad_sas_drv(double *erf_list, double *grad_list, double *x, double delta2, int ngrids, int natm, double *grad_sas);

void lap_sas_drv(double *erf_list, double *grad_list, double *x, double delta2, int ngrids, int natm, double *lap_sas);

void grad_eps_drv(double *erf_list, double *grad_list, double *x, double *exp_z, double *eps_z, double delta1, double delta2, double eps, double eps_sam, int ngrids, int natm, double *grad_eps);