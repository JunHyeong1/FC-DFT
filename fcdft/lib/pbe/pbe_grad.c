#define _USE_MATH_DEFINES
#include "pbe_grad.h"
#include "constant.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

void nuc_grad_eps_drv(double *erf_list, double *grad_list, double *eps_z, double delta2, int ngrids, int natm, double *nuc_grad_eps) {
    // erf_list: (natm, ngrids**3)
    // grad_list: (natm, ngrids**3, 3)
    // er: (natm, ngrids**3, 3)
    // grad_sas: (ngrids**3,3)
    int ngrids3 = ngrids * ngrids * ngrids;

    #pragma omp parallel for schedule(static)
    for (int l = 0; l < ngrids3; l++) {
        for (int i = 0; i < natm; i++) {
            double erf_prod = 1.0;
            for (int j = 0; j < natm; j++) {
                if (i != j) {
                    erf_prod *= erf_list[j*ngrids3+l];
                }
            }
            double eps_z_1 = eps_z[l] - 1;
            int il = (i*ngrids3+l)*3;
            nuc_grad_eps[il  ] -= eps_z_1 * grad_list[il  ] * erf_prod;
            nuc_grad_eps[il+1] -= eps_z_1 * grad_list[il+1] * erf_prod;
            nuc_grad_eps[il+2] -= eps_z_1 * grad_list[il+2] * erf_prod;
        }
    }
}

void grad_nuc_grad_eps_drv(double *erf_list, double *exp_list, double *er, double *x, double *eps_z, double *exp_z, double *dist, double delta1, double delta2, double eps_bulk, double eps_sam, int ngrids, int natm, double *grad_nuc_grad_eps) {
    // grad_list = numpy.multiply(er, gauss[:,:,None], out=er) / (delta2 * numpy.sqrt(PI))
    // grad_list: (natm, ngrids3, 3)
    int ngrids3 = ngrids * ngrids * ngrids;

    #pragma omp parallel for schedule(static)
    for (int l = 0; l < ngrids3; l++) {
        for (int i = 0; i < natm; i++) {
            double grad_nuc_grad_eps_l = 0.0;
            double erf_prod = 1.0; // A != B
            double tmp_x = 0.0;
            double tmp_y = 0.0;
            double tmp_z = 0.0;
            for (int j = 0; j < natm; j++) {
                double erf_prod2 = 1.0; // A != B and A != C
                if (i != j) {
                    erf_prod *= erf_list[j*ngrids3+l];
                    for (int k = 0; k < natm; k++) {
                        if (i != k && j != k) {
                            erf_prod2 *= erf_list[k*ngrids3+l];
                        }
                    }
                    int JL = (j*ngrids3+l);
                    int jl = (j*ngrids3+l)*3;
                    tmp_x += er[jl  ] * exp_list[JL] * erf_prod2;
                    tmp_y += er[jl+1] * exp_list[JL] * erf_prod2;
                    tmp_z += er[jl+2] * exp_list[JL] * erf_prod2;
                }
            }
            double eps_z_1 = eps_z[l] - 1.0;
            int IL = (i*ngrids3+l);
            int il = (i*ngrids3+l)*3;

            grad_nuc_grad_eps_l -= eps_z_1 / delta2 / delta2 / M_PI * er[il  ] * exp_list[IL] * tmp_x;
            grad_nuc_grad_eps_l -= eps_z_1 / delta2 / delta2 / M_PI * er[il+1] * exp_list[IL] * tmp_y;
            grad_nuc_grad_eps_l -= eps_z_1 / delta2 / delta2 / M_PI * er[il+2] * exp_list[IL] * tmp_z;            

            grad_nuc_grad_eps_l -= eps_z_1 / delta2 / sqrt(M_PI) * (2.0 / dist[IL]) * exp_list[IL] * erf_prod;

            grad_nuc_grad_eps_l += 2.0*eps_z_1 / delta2 / delta2 / sqrt(M_PI) * x[IL] * exp_list[IL] * erf_prod;

            // z contribution
            grad_nuc_grad_eps_l -= (eps_bulk - eps_sam) / delta1 / delta2 / M_PI * exp_z[l] * er[il+2] * exp_list[IL] * erf_prod;

            grad_nuc_grad_eps[IL] = grad_nuc_grad_eps_l;
        }
    }
}