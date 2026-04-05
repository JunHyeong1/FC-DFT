#define _USE_MATH_DEFINES
#include "constant.h"
#include "boundary_condition.h"
#include "gsl/gsl_sf.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <omp.h>

double cost_function(double x, double T, double kappa, double eps, double eps_sam, double stern_sam, double bottom) {
    double func = -2*KB2HARTREE*T*kappa*eps*sinh(-x/2/KB2HARTREE/T) - eps_sam*((bottom-x) / stern_sam);
    return func;
}

double phi_a_finder(double kappa, double T, double eps, double eps_sam, double stern_sam, double bottom) {
    double phi_a, func_a;
    double phi_a1 = 0.0;
    double phi_a2 = bottom;
    double func_1 = cost_function(phi_a1, T, kappa, eps, eps_sam, stern_sam, bottom);
    double func_2 = cost_function(phi_a2, T, kappa, eps, eps_sam, stern_sam, bottom);
    int i, j;
    for (i = 0; i < 20; i++) {
        phi_a = (phi_a1 + phi_a2)*0.5;
        func_a = cost_function(phi_a, T, kappa, eps, eps_sam, stern_sam, bottom);
        double phi[3] = {phi_a1, phi_a2, phi_a};
        double func[3] = {func_1, func_2, func_a};
        double func_tmp[3];
        for (j = 0; j < 3; j++) {
            func_tmp[j] = fabs(func[j]);
        }
        int min1 = 0, min2 = 0;
        for (j = 0; j < 3; j++) {
            if (func_tmp[j] < func_tmp[min1]) {
                min1 = j;
            }
        }
        func_tmp[min1] = DBL_MAX;
        for (j = 0; j < 3; j++) {
            if (func_tmp[j] < func_tmp[min1]) {
                min2 = j;
            }
        }
        phi_a1 = phi[min1], phi_a2 = phi[min2];
        func_1 = func[min1], func_2 = func[min2];
    }
    double phi_a_last;
    double grad;
    for (i = 0; i < 1000; i++) {
        phi_a_last = phi_a;
        func_a = cost_function(phi_a_last, T, kappa, eps, eps_sam, stern_sam, bottom);
        grad = kappa * eps * cosh(-phi_a_last / 2 / KB2HARTREE / T) + eps_sam / stern_sam;
        phi_a = phi_a_last - func_a / grad;
        if (fabs(phi_a - phi_a_last) < 1e-15) {
            return phi_a;
        }
    }
    printf("Maximum phi_a cycle reached.\n");
    exit(0);
}

void grad_sas_drv(double *erf_list, double *grad_list, double delta2, int ngrids, int natm, double *grad_sas) {
    // erf_list: (natm, ngrids**3)
    // grad_list: (natm, ngrids**3, 3)
    // x: (natm, ngrids**3)
    // grad_sas: (ngrids**3,3)
    int ngrids3 = ngrids * ngrids * ngrids;
    #pragma omp parallel for schedule(static)
    for (int l = 0; l < ngrids3; l++) {
        double grad_sas_lx = 0.0;
        double grad_sas_ly = 0.0;
        double grad_sas_lz = 0.0;
        for (int i = 0; i < natm; i++) {
            double erf_prod = 1.0;
            for (int j = 0; j < natm; j++) {
                if (i != j) {
                    erf_prod *= erf_list[j*ngrids3+l];
                }
            }
            int il = (i*ngrids3+l)*3;
            grad_sas_lx += grad_list[il  ] * erf_prod;
            grad_sas_ly += grad_list[il+1] * erf_prod;
            grad_sas_lz += grad_list[il+2] * erf_prod;
        }
        int l3 = l*3;
        grad_sas[l3  ] = grad_sas_lx;
        grad_sas[l3+1] = grad_sas_ly;
        grad_sas[l3+2] = grad_sas_lz;
    }
}

void lap_sas_drv(double *erf_list, double *grad_list, double *x, double delta2, int ngrids, int natm, double *lap_sas) {
    // erf_list: (natm, ngrids**3)
    // grad_list: (natm, ngrids**3, 3)
    // x: (natm, ngrids**3)
    // lap_sas: (ngrids**3,)
    int ngrids3 = ngrids * ngrids * ngrids;
    double coeff = 1.0 / delta2 / delta2 / sqrt(M_PI);

    #pragma omp parallel for schedule(static)
    for (int l = 0; l < ngrids3; l++) {
        double lap_sas_l = 0.0;
        for (int i = 0; i < natm; i++) {
            for (int j = 0; j < natm; j++) {
                double lap_l = 0.0;
                if (i == j) {
                    double x_il = x[i*ngrids3+l];
                    lap_l += -2.0 * coeff * x_il * exp(-x_il*x_il);
                }
                else {
                    int il = i*ngrids3*3+l*3;
                    int jl = j*ngrids3*3+l*3;
                    lap_l += grad_list[il  ] * grad_list[jl  ] 
                          +  grad_list[il+1] * grad_list[jl+1]
                          +  grad_list[il+2] * grad_list[jl+2];
                }
                double erf_prod_l = 1.0;
                for (int k = 0; k < natm; k++) {
                    if (k != i && k != j) {
                        erf_prod_l *= erf_list[k*ngrids3+l];
                    }
                }
                lap_sas_l += lap_l * erf_prod_l;
            }
        }
        lap_sas[l] = lap_sas_l;
    }
}

void grad_eps_drv(double *erf_list, double *grad_list, double *exp_z, double *eps_z, double delta1, double delta2, double eps, double eps_sam, int ngrids, int natm, double *grad_eps) {
    // erf_list: (natm, ngrids**3)
    // grad_list: (natm, ngrids**3, 3)
    // x: (natm, ngrids**3)
    // grad_eps: (ngrids**3,3)
    int ngrids3 = ngrids * ngrids * ngrids;
    double coeff = (eps - eps_sam) / (delta1 * sqrt(M_PI));

    #pragma omp parallel for schedule(static)
    for (int l = 0; l < ngrids3; l++) {
        double grad_eps_lx = 0.0;
        double grad_eps_ly = 0.0;
        double grad_eps_lz = 0.0;
        double erf_cumm[natm];
        double erf_prod = 1.0;
        for (int i = 0; i < natm; i++) {
            erf_cumm[i] = erf_prod;
            erf_prod *= erf_list[i*ngrids3+l];
        }
        double erf_prod_tot = erf_prod;
        double erf_rev = 1.0;
        for (int i = natm-1; i >= 0; i--) {
            int il = (i*ngrids3+l)*3;
            double eps_z_1 = eps_z[l] - 1;
            erf_prod = erf_cumm[i] * erf_rev;

            grad_eps_lx += (grad_list[il  ] * erf_prod) * eps_z_1;
            grad_eps_ly += (grad_list[il+1] * erf_prod) * eps_z_1;
            grad_eps_lz += (grad_list[il+2] * erf_prod) * eps_z_1;
            
            erf_rev *= erf_list[i*ngrids3+l];
        }
        int l3 = l*3;
        grad_eps_lz += coeff * exp_z[l] * erf_prod_tot;
        grad_eps[l3  ] = grad_eps_lx;
        grad_eps[l3+1] = grad_eps_ly;
        grad_eps[l3+2] = grad_eps_lz;
    }
}