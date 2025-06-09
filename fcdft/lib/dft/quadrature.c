#define _USE_MATH_DEFINES
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>

void roots_legendre(int n, double *abscissas, double *weights){
    double z, z1, pp, p1, p2, p3;
    int m = (n + 1) / 2;
    int i, j;
    for (i = 0; i < m; i++){
        z = cos(M_PI * (i + 0.75e0) / (n + 0.5e0));
        while (1){
            p1 = 1.0e0;
            p2 = 0.0e0;
            for (j = 0; j < n; j++){
                p3 = p2;
                p2 = p1;
                p1 = ((2.0e0*j + 1.0e0)*z*p2 - j*p3) / (j + 1.0e0);
            }
            pp = n * (z*p1 - p2) / (z*z- 1.0e0);
            z1 = z;
            z = z1 - p1/pp;
            if (fabs(z - z1) < 1.0e-15){
                break;
            }
        }
        abscissas[i] = -z;
        abscissas[n-i-1] = z;
        weights[i] = 2.0e0 / ((1.0e0 - z*z)*pp*pp);
        weights[n-i-1] = weights[i];
    }
}

double occ_drv(double sampling, double moe_energy, double fermi, double broad, double smear) {
    double dist = 1 / (exp((sampling - fermi)/smear) + 1);
    return dist * broad / (pow(sampling - moe_energy, 2) + pow(broad / 2, 2)) / 2.0 / M_PI;
}

double occ_grad_drv(double sampling, double moe_energy, double fermi, double broad, double smear) {
    double dist = 1 / (exp((sampling - fermi)/smear) + 1);
    return dist * (1 - dist) * broad / (pow(sampling - moe_energy, 2) + pow(broad / 2, 2)) / 2.0 / M_PI / smear;
}

void fermi_level_drv(double *moe_energy, double *abscissas, double *weights, double fermi, double broad, double smear, double window, int pts, int nbas, double *mo_occ, double *mo_grad) {
    int i, n;
    double sampling;
    double _mo_grad = 0.0;
    #pragma omp parallel for private(n, sampling) reduction(+:_mo_grad)
    for (i = 0; i < nbas; i++) {
        mo_occ[i] = 0;
        for (n = 0; n < pts; n++) {
            sampling = abscissas[n] * window + moe_energy[i];
            mo_occ[i] += window * weights[n] * occ_drv(sampling, moe_energy[i], fermi, broad, smear);
            _mo_grad += window * weights[n] * occ_grad_drv(sampling, moe_energy[i], fermi, broad, smear);
        }
    }
    *mo_grad = _mo_grad;
}
