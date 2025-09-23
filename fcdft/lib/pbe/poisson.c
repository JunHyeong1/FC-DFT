#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <omp.h>
#include <math.h>
#include "lapacke.h"
#include "cblas.h"

double const TWO_PI = 2.0 * M_PI;

double dft_laplacian_fourth(double k, double spacing2) {
    // dft stands for discrete Fourier transform, not density functional theory
    // NOTE: This function returns -nabla^2 in k-space.
    double k2 = -(-1.0/280.0*cos(TWO_PI*4*k) + 16.0/315.0*cos(TWO_PI*3*k) - 2.0/5.0*cos(TWO_PI*2*k) + 16.0/5.0*cos(TWO_PI*k) - 205.0/72.0) / spacing2;
    return k2;
}

void laplacian_2d(double complex *phik, double *lap, int ngrids, double spacing, double *kpts, double complex *rhok) {
    // This function computes the Laplacian of phi in k-space, not the charge density.
    int ngrids2 = ngrids * ngrids;
    int i,j,k,ij;
    double complex alpha = 1.0;
    double complex beta = 0.0;
    double spacing2 = spacing * spacing;
    for (i = 0; i < ngrids; i++) {
        for (j = 0; j < ngrids; j++) {
            double kx2 = dft_laplacian_fourth(kpts[i], spacing2);
            double ky2 = dft_laplacian_fourth(kpts[j], spacing2);
            double c = -(kx2 + ky2);
            double complex *buf = malloc(sizeof(double complex) * ngrids2);
            for (ij = 0; ij < ngrids2; ij++) {
                buf[ij] = lap[ij];
            }
            for (k = 0; k < ngrids; k++) {
                buf[k*ngrids + k] += c;
            }
            cblas_zgemv(CblasRowMajor, CblasNoTrans, ngrids, ngrids, &alpha, buf, ngrids, &phik[i*ngrids2+j*ngrids], 1, &beta, &rhok[i*ngrids2+j*ngrids], 1);
            free(buf);
        }
    }
}

void poisson_fft_2d(double complex *rhok, double *lap, int ngrids, double spacing, double *kpts, double complex *phi, int *info) {
    int ngrids2 = ngrids * ngrids;
    int i,j,k;
    double *D = malloc(sizeof(double) * ngrids);
    double *V = malloc(sizeof(double) * ngrids2);
    double complex alpha = 1.0;
    double complex beta = 0.0;
    double spacing2 = spacing * spacing;
    for (i = 0; i < ngrids2; i++) {
        V[i] = lap[i];
    }
    *info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', ngrids, V, ngrids, D);
    double complex *VC = malloc(sizeof(double complex) * ngrids2);
    for (i = 0; i < ngrids2; i++) {
        VC[i] = V[i];
    }
    free(V);
    // x = V(D+cI)^-1 V^T b
    #pragma omp parallel private(i,j)
    {
    double complex *buf = malloc(sizeof(double complex) * ngrids);
    #pragma omp for collapse(2)
    for (i = 0; i < ngrids; i++) {
        for (j = 0; j < ngrids; j++) {
            double kx2 = dft_laplacian_fourth(kpts[i], spacing2);
            double ky2 = dft_laplacian_fourth(kpts[j], spacing2);
            double c = -(kx2 + ky2);
            cblas_zgemv(CblasRowMajor, CblasTrans, ngrids, ngrids, &alpha, VC, ngrids, &rhok[i*ngrids2+j*ngrids], 1, &beta, buf, 1);
            for (k = 0; k < ngrids; k++) {
                buf[k] /= (D[k] + c);
            }
            cblas_zgemv(CblasRowMajor, CblasNoTrans, ngrids, ngrids, &alpha, VC, ngrids, buf, 1, &beta, &phi[i*ngrids2+j*ngrids], 1);
        }
    }
    free(buf);
    }
    free(D);
    free(VC);
}