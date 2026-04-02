#define _USE_MATH_DEFINES
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <gsl/gsl_integration.h>

double const TWO_PI = 2.0 * M_PI;

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
    double a = sampling - moe_energy;
    double b = broad / 2.0;
    return dist * broad / (a*a + b*b) / TWO_PI;
}

double occ_grad_drv(double sampling, double moe_energy, double fermi, double broad, double smear) {
    double dist = 1 / (exp((sampling - fermi)/smear) + 1);
    double a = sampling - moe_energy;
    double b = broad / 2.0;
    return dist * (1 - dist) * broad / (a*a + b*b) / TWO_PI / smear;
}

void fermi_level_drv(double *moe_energy, double *abscissas, double *weights, double fermi, double broad, double smear, double window, int pts, int nbas, double *mo_occ, double *mo_grad) {
    int i, n;
    double sampling;
    double _mo_grad = 0.0;
    double *window_weights = (double *)malloc(sizeof(double) * pts);

    for (n = 0; n < pts; n++) {
        window_weights[n] = window * weights[n];
    }

    #pragma omp parallel for private(n, sampling) reduction(+:_mo_grad)
    for (i = 0; i < nbas; i++) {
        mo_occ[i] = 0;
        for (n = 0; n < pts; n++) {
            sampling = abscissas[n] * window + moe_energy[i];
            mo_occ[i] += window_weights[n] * occ_drv(sampling, moe_energy[i], fermi, broad, smear);
            _mo_grad += window_weights[n] * occ_grad_drv(sampling, moe_energy[i], fermi, broad, smear);
        }
    }
    *mo_grad = _mo_grad;
    free(window_weights);
}

void occupation_drv(double *moe_energy, double *abscissas, double *weights, double fermi, double broad, double smear, double window, int pts, int nbas, double *mo_occ) {
    int i, n;
    double sampling;
    #pragma omp parallel for private(n, sampling)
    for (i = 0; i < nbas; i++) {
        mo_occ[i] = 0;
        for (n = 0; n < pts; n++) {
            sampling = abscissas[n] * window + moe_energy[i];
            mo_occ[i] += window * weights[n] * occ_drv(sampling, moe_energy[i], fermi, broad, smear);
        }
    }
}

void occupation_grad_drv(double *moe_energy, double *abscissas, double *weights, double fermi, double broad, double smear, double window, int pts, int nbas, double *occ_grad) {
    int i, n;
    double sampling;
    #pragma omp parallel for private(n, sampling)
    for (i = 0; i < nbas; i++) {
        occ_grad[i] = 0;
        for (n = 0; n < pts; n++) {
            sampling = abscissas[n] * window + moe_energy[i];
            occ_grad[i] += window * weights[n] * occ_grad_drv(sampling, moe_energy[i], fermi, broad, smear);
        }
    }        
}

struct occ_params {
    double moe_energy;
    double fermi;
    double broad;
    double smear;
};

double gsl_occ_drv(double x, void *p) {
    struct occ_params *params = (struct occ_params *)p;
    double moe_energy = params->moe_energy;
    double fermi = params->fermi;
    double broad = params->broad;
    double smear = params->smear;

    double dist = 1 / (exp((x - fermi)/smear) + 1);
    double a = x - moe_energy;
    double b = broad / 2.0;
    return dist * broad / (a*a + b*b) / TWO_PI;
}

double gsl_occ_grad_drv(double x, void *p) {
    struct occ_params *params = (struct occ_params *)p;
    double moe_energy = params->moe_energy;
    double fermi = params->fermi;
    double broad = params->broad;
    double smear = params->smear;

    double dist = 1 / (exp((x - fermi)/smear) + 1);
    double a = x - moe_energy;
    double b = broad / 2.0;
    return dist * (1 - dist) * broad / (a*a + b*b) / TWO_PI / smear;
}

void gsl_occupation_drv(double *moe_energy, double fermi, double broad, double smear, int nbas, double *mo_occ) {
    struct occ_params params;
    params.fermi = fermi;
    params.broad = broad;
    params.smear = smear;

    double result1, error1;
    double result2, error2;
    double result3, error3;

    int quad_order = 2000;

    gsl_integration_workspace *w = gsl_integration_workspace_alloc(quad_order);
    gsl_function F;
    F.function = &gsl_occ_drv;
    F.params = &params;

    double smear_offset = 2000*smear;
    double broad_offset = 2000*broad;
    double *pts = malloc(sizeof(double)*4);

    for (int i = 0; i < nbas; i++) {
        params.moe_energy = moe_energy[i];
        if (fermi < moe_energy[i]) {
            pts[0] = fermi - smear_offset;
            pts[1] = fermi;
            pts[2] = moe_energy[i];
            pts[3] = moe_energy[i] + broad_offset;
            // gsl_integration_qagil(&F, fermi-smear_offset, 0, 1e-10, 1000, w, &result1, &error1);
            // gsl_integration_qagiu(&F, moe_energy[i]+broad_offset, 0, 1e-10, 1000, w, &result2, &error2);
            // gsl_integration_qagp(&F, pts, 4, 0, 1e-10, 1000, w, &result3, &error3);
            // gsl_integration_qags(&F, fermi, moe_energy[i], 0, 1e-10, 1000, w, &result3, &error3);
        }
        else {
            pts[0] = moe_energy[i] - broad_offset;
            pts[1] = moe_energy[i];
            pts[2] = fermi;
            pts[3] = fermi + smear_offset;
            // gsl_integration_qagil(&F, moe_energy[i]-broad_offset, 0, 1e-10, 1000, w, &result1, &error1);
            // gsl_integration_qagiu(&F, fermi+smear_offset, 0, 1e-10, 1000, w, &result2, &error2);
            // gsl_integration_qagp(&F, pts, 4, 0, 1e-10, 1000, w, &result3, &error3);
            // gsl_integration_qags(&F, moe_energy[i], fermi, 0, 1e-10, 1000, w, &result3, &error3);
        }
        gsl_integration_qagil(&F, pts[0], 1e-12, 1e-11, quad_order, w, &result1, &error1);
        gsl_integration_qagiu(&F, pts[3], 1e-12, 1e-11, quad_order, w, &result2, &error2);
        gsl_integration_qagp(&F, pts, 4, 1e-12, 1e-11, quad_order, w, &result3, &error3);
        mo_occ[i] = result1 + result2 + result3;
    }
    gsl_integration_workspace_free(w);
    free(pts);
}

void gsl_occupation_grad_drv(double *moe_energy, double fermi, double broad, double smear, int nbas, double *occ_grad) {
    struct occ_params params;
    params.fermi = fermi;
    params.broad = broad;
    params.smear = smear;

    double result1, error1;
    double result2, error2;
    double result3, error3;

    int quad_order = 2000;

    gsl_integration_workspace *w = gsl_integration_workspace_alloc(quad_order);
    gsl_function F;
    F.function = &gsl_occ_grad_drv;
    F.params = &params;

    double smear_offset = 2000*smear;
    double broad_offset = 2000*broad;
    double *pts = malloc(sizeof(double)*4);

    for (int i = 0; i < nbas; i++) {
        params.moe_energy = moe_energy[i];
        if (fermi < moe_energy[i]) {
            pts[0] = fermi - smear_offset;
            pts[1] = fermi;
            pts[2] = moe_energy[i];
            pts[3] = moe_energy[i] + broad_offset;
            // gsl_integration_qagil(&F, fermi, 1e-12, 1e-8, 1000, w, &result1, &error1);
            // gsl_integration_qagiu(&F, moe_energy[i], 1e-12, 1e-8, 1000, w, &result2, &error2);
            // gsl_integration_qags(&F, fermi, moe_energy[i], 1e-12, 1e-8, 1000, w, &result3, &error3);
        }
        else {
            pts[0] = moe_energy[i] - broad_offset;
            pts[1] = moe_energy[i];
            pts[2] = fermi;
            pts[3] = fermi + smear_offset;
            // gsl_integration_qagil(&F, moe_energy[i], 1e-12, 1e-8, 1000, w, &result1, &error1);
            // gsl_integration_qagiu(&F, fermi, 1e-12, 1e-8, 1000, w, &result2, &error2);
            // gsl_integration_qags(&F, moe_energy[i], fermi, 1e-12, 1e-8, 1000, w, &result3, &error3);
        }
        gsl_integration_qagil(&F, pts[0], 1e-12, 1e-11, quad_order, w, &result1, &error1);
        gsl_integration_qagiu(&F, pts[3], 1e-12, 1e-11, quad_order, w, &result2, &error2);
        gsl_integration_qagp(&F, pts, 4, 1e-12, 1e-11, quad_order, w, &result3, &error3);
        occ_grad[i] = result1 + result2 + result3;
    }
    gsl_integration_workspace_free(w);
    free(pts);
}

void gsl_fermi_level_drv(double *moe_energy, double fermi, double broad, double smear, int nbas, double *mo_occ, double *mo_grad) {
    gsl_occupation_drv(moe_energy, fermi, broad, smear, nbas, mo_occ);
    gsl_occupation_grad_drv(moe_energy, fermi, broad, smear, nbas, mo_grad);
}