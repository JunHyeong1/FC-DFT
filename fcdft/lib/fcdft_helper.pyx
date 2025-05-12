from cython.parallel cimport prange
cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport M_PI, exp, cos
import os

cdef int OMP_NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])

cdef double occ(double x, double ei, double fermi, double broad, double smear) nogil:
    cdef double dist
    dist = 1.0e0 / (exp((x-fermi)/smear)+1.0e0)
    return dist * broad / ((x-ei)**2+(broad/2.0e0)**2)/2/M_PI

cdef double occ_grad(double x, double ei, double fermi, double broad, double smear) nogil:
    cdef double dist
    dist = 1.0 / (exp((x-fermi)/smear)+1.0e0)
    return dist * (1.0e0-dist) * broad/((x-ei)**2+(broad/2.0e0)**2)/2/M_PI/smear

cdef double occ_hess(double x, double ei, double fermi, double broad, double smear) nogil:
    cdef double dist
    dist = 1.0 / (exp((x-fermi)/smear)+1.0e0)
    return dist * (1.0e0-dist) * (1.0e0-2.0e0*dist) * broad/((x-ei)**2+(broad/2.0e0)**2)/2/M_PI/smear**2

@cython.boundscheck(False)
@cython.wraparound(False)
def quadrature(double[:] mo_energy, double[:] abscissas, double[:] weights, double fermi, double broad, double smear, double window, int order):
    cdef double ni_grad
    cdef Py_ssize_t i, n
    cdef int pts = abscissas.shape[0]
    cdef int nbas = mo_energy.shape[0]
    cdef double[:] mo_occ = np.zeros_like(mo_energy)
    cdef double[:] mo_grad = np.zeros_like(mo_energy)
    cdef double[:] mo_hess = np.zeros_like(mo_energy)
    cdef double[:] sampling = np.empty(pts, dtype=np.float64)

    for i in prange(nbas, nogil=True, num_threads=OMP_NUM_THREADS):
        for n in range(pts):
            sampling[n] = abscissas[n] * window + mo_energy[i]
            mo_occ[i] += window * weights[n] * occ(sampling[n], mo_energy[i], fermi, broad, smear)

    if order > 0:
        for i in prange(nbas, nogil=True, num_threads=OMP_NUM_THREADS):
            for n in range(pts):
                sampling[n] = abscissas[n] * window + mo_energy[i]
                mo_grad[i] += window * weights[n] * occ_grad(sampling[n], mo_energy[i], fermi, broad, smear)
        if order > 1:
            for i in prange(nbas, nogil=True, num_threads=OMP_NUM_THREADS):
                for n in range(pts):
                    sampling[n] = abscissas[n] * window + mo_energy[i]
                    mo_hess[i] += window * weights[n] * occ_hess(sampling[n], mo_energy[i], fermi, broad, smear)
            return np.asarray(mo_occ), np.asarray(mo_grad), np.asarray(mo_hess)
        else:
            return np.asarray(mo_occ), np.asarray(mo_grad)
    else:
        return np.asarray(mo_occ)

def roots_legendre(int n):
    cdef double [:] roots = np.empty(n, dtype=np.float64)
    cdef double [:] weights = np.empty(n, dtype=np.float64)
    cdef double z, pp, p1, p2, p3
    if n % 2 == 0:
        raise RuntimeError('Even number of roots not supported.')
    cdef int m = int((n + 1) / 2)
    cdef Py_ssize_t i, j
    for i in range(m):
        z = cos(M_PI * (i + 0.75e0) / (n + 0.5e0))
        while True:
            p1 = 1.0e0
            p2 = 0.0e0
            for j in range(n):
                p3 = p2
                p2 = p1
                p1 = ((2.0e0*j + 1.0e0)*z*p2 - j*p3)/(j+1.0e0)
            pp = n*(z*p1-p2)/(z**2-1.0e0)
            z1 = z
            z = z1-p1/pp
            if (abs(z-z1) < 1.0e-15): break
        roots[i] = -z
        roots[n-i-1] = z
        weights[i] = 2.0e0 / ((1.0e0-z**2)*pp**2)
        weights[n-i-1] = weights[i]
    return np.asarray(roots), np.asarray(weights)