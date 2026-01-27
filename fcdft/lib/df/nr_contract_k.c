#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include "cblas.h"

void nr_mapdm1(double *buf, double *eri, double *dms, int nao, int naux) {
    // vk shape: (nao, nao), (155,155)
    // eri shape: (naux, nao, nao) (240,155,155)
    // dms shape: (nao, nao) (155,155)
    // buf shape: (naux, nao, nao) (240,155,155)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, naux*nao, nao, nao, 1.0, eri, nao, dms, nao, 0.0, buf, nao);
}