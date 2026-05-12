#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include "cblas.h"

void nr_df_contract_k(double *out, double *buf, double *eri, double *dms, int nao, int naux) {
    // out shape: (2, nao, nao)
    // vk shape: (nao, nao), (155,155)
    // eri shape: (naux, nao, nao) (240,155,155)
    // dms shape: (nao, nao) (2,155,155)
    // buf shape: (naux, nao, nao) (240,155,155)
    int idx, p;
    int nao2 = nao*nao;
    int nauxnao = naux*nao;
    for (idx = 0; idx < 2; idx++) { // spin index
        double *_out = out + idx*nao2;
        const double *_dms = dms + idx*nao2;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nauxnao, nao, nao, 1.0, eri, nao, _dms, nao, 0.0, buf, nao);
        for (p = 0; p < naux; p++) {
            const double *_eri = eri + p*nao2;
            const double *_buf = buf + p*nao2;
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nao, nao, nao, 1.0, _eri, nao, _buf, nao, 1.0, _out, nao);
        }
    }
}