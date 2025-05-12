#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <complex.h>
#include <omp.h>
#include "nr_numint.h"

// 'ip,ip->p'
void VXC_dzcontract_rho(double *rho, double *bra, double complex *ket,
                       int nao, int ngrids)
{
#pragma omp parallel
{
        size_t Ngrids = ngrids;
        int nthread = omp_get_num_threads();
        int blksize = MAX((Ngrids+nthread-1) / nthread, 1);
        int ib, b0, b1, i, j;
#pragma omp for
        for (ib = 0; ib < nthread; ib++) {
                b0 = ib * blksize;
                b1 = MIN(b0 + blksize, ngrids);
                for (j = b0; j < b1; j++) {
                        rho[j] = bra[j] * creal(ket[j]);
                }
                for (i = 1; i < nao; i++) {
                for (j = b0; j < b1; j++) {
                        rho[j] += bra[i*Ngrids+j] * creal(ket[i*Ngrids+j]);
                } }
        }
}
}
