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

void poisson_fft_2d(double complex *rhok, double *lap, int ngrids, double spacing, double *kpts, double complex *phik, int *info) {
    int ngrids2 = ngrids * ngrids;
    int ngrids3 = ngrids * ngrids * ngrids;
    int i,j,k;
    double *D = malloc(sizeof(double) * ngrids);
    double *V = malloc(sizeof(double) * ngrids2);
    double alpha = 1.0;
    double beta = 0.0;
    double spacing2 = spacing * spacing;

    cblas_dcopy(ngrids2, lap, 1, V, 1);
    // for (i = 0; i < ngrids2; i++) {
    //     V[i] = lap[i];
    // }
    *info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', ngrids, V, ngrids, D);

    // x = V(D+cI)^-1 V^T b
    double *rhokre = malloc(sizeof(double) * ngrids3);
    double *rhokim = malloc(sizeof(double) * ngrids3);
    double *phikre = malloc(sizeof(double) * ngrids3);
    double *phikim = malloc(sizeof(double) * ngrids3);
    #pragma omp parallel for
    for (i = 0; i < ngrids3; i++) {
        rhokre[i] = creal(rhok[i]);
        rhokim[i] = cimag(rhok[i]);
    }
    #pragma omp parallel private(i,j) shared(phikre, phikim)
    {
    double *bufre = malloc(sizeof(double) * ngrids);
    double *bufim = malloc(sizeof(double) * ngrids);
    #pragma omp for collapse(2)
    for (i = 0; i < ngrids; i++) {
        for (j = 0; j < ngrids; j++) {
            double kx2 = dft_laplacian_fourth(kpts[i], spacing2);
            double ky2 = dft_laplacian_fourth(kpts[j], spacing2);
            double c = -(kx2 + ky2);
            cblas_dgemv(CblasRowMajor, CblasTrans, ngrids, ngrids, alpha, V, ngrids, &rhokre[i*ngrids2+j*ngrids], 1, beta, bufre, 1);
            cblas_dgemv(CblasRowMajor, CblasTrans, ngrids, ngrids, alpha, V, ngrids, &rhokim[i*ngrids2+j*ngrids], 1, beta, bufim, 1);
            for (k = 0; k < ngrids; k++) {
                bufre[k] /= (D[k] + c);
                bufim[k] /= (D[k] + c);
            }
            cblas_dgemv(CblasRowMajor, CblasNoTrans, ngrids, ngrids, alpha, V, ngrids, bufre, 1, beta, &phikre[i*ngrids2+j*ngrids], 1);
            cblas_dgemv(CblasRowMajor, CblasNoTrans, ngrids, ngrids, alpha, V, ngrids, bufim, 1, beta, &phikim[i*ngrids2+j*ngrids], 1);
        }
    }
    free(bufre);
    free(bufim);
    }
    free(D);
    free(V);
    #pragma omp parallel for
    for (i = 0; i < ngrids3; i++) {
        phik[i] = phikre[i] + I * phikim[i];
    }
    free(phikre);
    free(phikim);
    free(rhokre);
    free(rhokim);
}

// Under development
void poisson_fft_2d_opt(double complex *rhok, double *lap, int ngrids, double spacing, double *kpts, double complex *phik, int *info) {
    int ngrids2 = ngrids * ngrids;
    int ngrids3 = ngrids * ngrids * ngrids;
    int i,j,k;
    double *D = malloc(sizeof(double) * ngrids);
    double *V = malloc(sizeof(double) * ngrids2);
    double alpha = 1.0;
    double beta = 0.0;
    double spacing2 = spacing * spacing;
    for (i = 0; i < ngrids2; i++) {
        V[i] = lap[i];
    }
    *info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', ngrids, V, ngrids, D);

    // x = V(D+cI)^-1 V^T b
    double *_rhok = malloc(sizeof(double) * ngrids3 * 2);
    double *_phik = malloc(sizeof(double) * ngrids3 * 2);
    for (i = 0; i < ngrids3; i++) {
        _rhok[i*2] = creal(rhok[i]);
        _rhok[i*2+1] = cimag(rhok[i]);
    }
    #pragma omp parallel private(i,j)
    {
    double *buf = malloc(sizeof(double) * ngrids * 2);
    #pragma omp for collapse(2)
    for (i = 0; i < ngrids; i++) {
        for (j = 0; j < ngrids; j++) {
            double kx2 = dft_laplacian_fourth(kpts[i], spacing2);
            double ky2 = dft_laplacian_fourth(kpts[j], spacing2);
            double c = -(kx2 + ky2);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ngrids, 2, ngrids, alpha, V, ngrids, &_rhok[i*ngrids2+j*ngrids], 2, beta, buf, 2);
            for (k = 0; k < ngrids; k++) {
                buf[2*k] /= (D[k] + c);
                buf[2*k+1] /= (D[k] + c);
            }
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ngrids, 2, ngrids, alpha, V, ngrids, buf, 2, beta, &_phik[i*ngrids2+j*ngrids], 2);
        }
    }
    free(buf);
    }
    free(D);
    free(V);
    for (i = 0; i < ngrids3; i++) {
        phik[i] = _phik[2*i] + I * _phik[2*i+1];
    }
    free(_phik);
    free(_rhok);
}