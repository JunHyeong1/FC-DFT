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

void boundary_cond_drv(double kappa, double T, double eps, double eps_sam, double stern_sam, double bottom, double spacing, int ngrids, double *phi_z, double *slope) {
    double phi_a = phi_a_finder(kappa, T, eps, eps_sam, stern_sam, bottom);
    *slope = (phi_a - bottom) / stern_sam;
    int i, j, k;
    int ngrids2 = ngrids * ngrids;
    #pragma omp parallel for private(i,j,k)
    for (i = 0; i < ngrids; i++) {
        for (j = 0; j < ngrids; j++) {
            for (k = 0; k < ngrids; k++) {
                if (k*spacing <= stern_sam) {
                    phi_z[i*ngrids2 + j*ngrids + k] = *slope*k*spacing + bottom;
                }
                else {
                    phi_z[i*ngrids2 + j*ngrids + k] = -4*KB2HARTREE*T*atanh(exp(-kappa*(k*spacing - stern_sam))*tanh(-phi_a/4/KB2HARTREE/T));
                }
            }
        }
    }
}

void distance_calculator(double *coords, double *atom_coords, int natm, int ngrids, double *dist) {
    int ngrids3 = ngrids*ngrids*ngrids;
    int i, j;
    #pragma omp parallel for private(i,j)
    for (i = 0; i < natm; i++) {
        for (j = 0; j < ngrids3; j++) {
            dist[i*ngrids3 + j] = sqrt(pow(coords[j*3  ] - atom_coords[i*3  ], 2) 
                                     + pow(coords[j*3+1] - atom_coords[i*3+1], 2)
                                     + pow(coords[j*3+2] - atom_coords[i*3+2], 2));
        }
    }
}

void grad_sas_drv(double *erf_list, double *er, double *x, double delta2, int ngrids, int natm, double *grad_sas) {
    int i, j, k;
    int ngrids3 = ngrids * ngrids * ngrids;
    double *erf_prod = malloc(sizeof(double) * natm * ngrids3);
    double *gauss = malloc(sizeof(double) * natm * ngrids3);
    
    #pragma omp parallel for
    for (i = 0; i < ngrids3*3; i++) {
        grad_sas[i] = 0;
    }

    #pragma omp parallel for private(i, j)
    for (i = 0; i < natm; i++) {
        for (j = 0; j < ngrids3; j++) {
            erf_prod[i*ngrids3 + j] = 1;
            gauss[i*ngrids3 + j] = exp(-pow(x[i*ngrids3 + j], 2));
        }
    }

    #pragma omp parallel for private(i, j, k)
    for (i = 0; i < natm; i++) {
        for (k = 0; k < natm; k++) {
            if (k != i) {
                for (j = 0; j < ngrids3; j++) {
                    erf_prod[i*ngrids3 + j] *= erf_list[k*ngrids3 + j];
                }
            }
        }
    }

    #pragma omp parallel for private(i, j, k) reduction(+:grad_sas[0:ngrids3*3])
    for (i = 0; i < natm; i++) {
        for (j = 0; j < ngrids3; j++) {
            double coeff = 1 / delta2 / sqrt(M_PI) * gauss[i*ngrids3 + j] * erf_prod[i*ngrids3 + j];
            for (k = 0; k < 3; k++) {
                grad_sas[3*j + k] += er[i*ngrids3*3 + 3*j + k] * coeff;
            }
        }
    }
    free(erf_prod);
    free(gauss);
}

void boundary_cond_gradient_drv(double *coords, double *atom_coords, double *atomic_radii,
                                double *phi_z, double *sas, double kappa, double T,
                                double probe, double delta2, double stern_sam, double spacing,
                                double slope, int ngrids, int natm, double *grad_bc, 
                                double *grad_phi_z, double *grad_sas) {
    int ngrids3 = ngrids * ngrids * ngrids;
    int ngrids2 = ngrids * ngrids;
    int i, j, k;

    double *dphidz = malloc(sizeof(double) * ngrids3);
    double *dist = malloc(sizeof(double) * natm * ngrids3);
    double *erf_list = malloc(sizeof(double) * natm * ngrids3);
    double *x = malloc(sizeof(double) * natm * ngrids3);
    double *er = malloc(sizeof(double) * natm * ngrids3 * 3);
    double *rp = malloc(sizeof(double) * natm * ngrids3 * 3);

    distance_calculator(coords, atom_coords, natm, ngrids, dist);
    
    #pragma omp parallel for private(i, j, k)
    for (i = 0; i < ngrids; i++) {
        for (j = 0; j < ngrids; j++) {
            for (k = 0; k < ngrids; k++) {
                if (k*spacing <= stern_sam) {
                    dphidz[i*ngrids2 + j*ngrids + k] = slope;
                }
                else {
                    dphidz[i*ngrids2 + j*ngrids + k] = 2.0*KB2HARTREE*T*kappa*sinh(-phi_z[i*ngrids2 + j*ngrids + k] / 2 / KB2HARTREE / T);
                }
            }
        }
    }

    for (i = 0; i < ngrids3; i++) {
        for (j = 0; j < 3; j++) {
            if (j == 2) {
                grad_phi_z[i*3 + j] = dphidz[i];
            }
            else {
                grad_phi_z[i*3 + j] = 0;
            }
        }
    }

    #pragma omp parallel for private(i, j, k)
    for (i = 0; i < natm; i++) {
        for (j = 0; j < ngrids3; j++) {
            for (k = 0; k < 3; k++) {
                rp[i*ngrids3*3 + j*3 + k] = coords[j*3 + k] - atom_coords[i*3 + k];
            }
        }
    }

    #pragma omp parallel for private(i, j)
    for (i = 0; i < natm; i++) {
        for (j = 0; j < ngrids3; j++) {
            x[i*ngrids3 + j] = (dist[i*ngrids3 + j] - atomic_radii[i] - probe) / delta2;
            erf_list[i*ngrids3 + j] = 0.5 * (1 + gsl_sf_erf(x[i*ngrids3 + j]));
        }
    }

    #pragma omp parallel for private(i, j, k)
    for (i = 0; i < natm; i++) {
        for (j = 0; j < ngrids3; j++) {
            for (k = 0; k < 3; k++) {
                er[i*ngrids3*3 + j*3 + k] = rp[i*ngrids3*3 + j*3 + k] / dist[i*ngrids3 + j];
            }
        }
    }

    grad_sas_drv(erf_list, er, x, delta2, ngrids, natm, grad_sas);

    #pragma omp parallel for private(i, j)
    for (i = 0; i < ngrids3; i++) {
        for (j = 0; j < 3; j++) {
            grad_bc[3*i + j] = grad_phi_z[3*i + j] * sas[i] + grad_sas[3*i + j] * phi_z[i];
        }
    }

    free(erf_list);
    free(x);
    free(er);
    free(dphidz);
    free(dist);
    free(rp);
}
