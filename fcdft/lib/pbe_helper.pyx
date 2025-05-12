from cython.parallel cimport prange
cimport cython
cimport numpy as np
import numpy as np
import scipy
from libc.math cimport exp, erf, sqrt
from libc.math cimport M_PI
import os

cdef int OMP_NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])

@cython.boundscheck(False)
@cython.wraparound(False)
def distance_calculator(double[:,:] coords, double[:,:] atom_coords):
    cdef int natm = atom_coords.shape[0]
    cdef int ndim = coords.shape[0]
    cdef double[:,:] dist = np.zeros((natm, ndim), dtype=np.float64)

    cdef Py_ssize_t i, j
    for i in prange(natm, nogil=True, num_threads=OMP_NUM_THREADS):
        for j in range(ndim):
            dist[i,j] = sqrt((coords[j,0] - atom_coords[i,0])**2 + (coords[j,1] - atom_coords[i,1])**2 + (coords[j,2] - atom_coords[i,2])**2)
    return np.asarray(dist)

@cython.boundscheck(False)
@cython.wraparound(False)
def product_vector_vector(double[:,:] a, double[:,:] b):
    """ numpy.einsum('ij,ij->i')"""
    cdef int ndim = a.shape[0]
    cdef Py_ssize_t i
    cdef double[:] product = np.empty(ndim, dtype=np.float64)
    if a.shape[1] != 3 or b.shape[1] != 3:
        raise ValueError("Input arrays must have exactly 3 columns.")
    for i in prange(ndim, nogil=True, num_threads=OMP_NUM_THREADS):
        product[i] = a[i,0]*b[i,0] + a[i,1]*b[i,1] + a[i,2]*b[i,2]
    return np.asarray(product)

@cython.boundscheck(False)
@cython.wraparound(False)
def product_vector_scalar(double[:,:] a, double[:] b):
    """numpy.einsum('ij,i->ij')"""
    cdef int ndim = a.shape[0]
    cdef Py_ssize_t i
    cdef double[:,:] product = np.empty((ndim, 3), dtype=np.float64)
    if a.shape[1] != 3:
        raise ValueError("Input array a must have exactly 3 columns.")
    for i in prange(ndim, nogil=True, num_threads=OMP_NUM_THREADS):
        product[i,0] = a[i,0]*b[i]
        product[i,1] = a[i,1]*b[i]
        product[i,2] = a[i,2]*b[i]
    return np.asarray(product)

@cython.boundscheck(False)
@cython.wraparound(False)
def product_atom_vector_scalar(double[:,:,:] a, double[:,:] b):
    """numpy.einsum('ijk,jk->ijk')"""
    cdef int natm = a.shape[0]
    cdef int ndim = a.shape[1]
    cdef Py_ssize_t i, j
    cdef double[:,:,:] product = np.empty((natm, ndim, 3), dtype=np.float64)
    if a.shape[2] != 3:
        raise ValueError("Input array a must have exactly 3 columns.")
    for j in prange(ndim, nogil=True, num_threads=OMP_NUM_THREADS):
        for i in range(natm):
            product[i,j,0] = a[i,j,0] * b[i,j]
            product[i,j,1] = a[i,j,1] * b[i,j]
            product[i,j,2] = a[i,j,2] * b[i,j]
    return np.asarray(product)

@cython.boundscheck(False)
@cython.wraparound(False)
def lap_sas(double[:,:] erf_list, double[:,:,:] grad_list, double[:,:] x, double delta2):
    cdef int natm = erf_list.shape[0]
    cdef int tot_grids = erf_list.shape[1]
    cdef double[:] lap_sas = np.empty(tot_grids, dtype=np.float64)
    cdef double[:] lap = np.empty(tot_grids, dtype=np.float64)
    cdef double[:] erf_prod= np.empty(tot_grids, dtype=np.float64)
    cdef Py_ssize_t i, j, k, l

    for i in prange(tot_grids, nogil=True, num_threads=OMP_NUM_THREADS):
        lap_sas[i] = 0.0e0

    for i in range(natm):
        for j in range(natm):
            for k in range(tot_grids):
                lap[k] = 0.0e0
                erf_prod[k] = 1.0e0
            if i == j:
                for k in prange(tot_grids, nogil=True, num_threads=OMP_NUM_THREADS):
                    lap[k] = - 2.0e0 / delta2**2 / M_PI**0.5 * x[i,k] * exp(-x[i,k]**2)
            else:
                for k in prange(tot_grids, nogil=True, num_threads=OMP_NUM_THREADS):
                    for l in range(3):
                        lap[k] += grad_list[i,k,l] * grad_list[j,k,l]

            for k in range(natm):
                for l in prange(tot_grids, nogil=True, num_threads=OMP_NUM_THREADS):
                    if k != i and k != j:
                        erf_prod[l] = erf_prod[l] * erf_list[k,l]

            for l in prange(tot_grids, nogil=True, num_threads=OMP_NUM_THREADS):
                lap_sas[l] += lap[l] * erf_prod[l]

    return np.asarray(lap_sas)

@cython.boundscheck(False)
@cython.wraparound(False)
def grad_eps(double[:,:] atom_coords, double[:,:] coords, double eps_sam, double eps, double probe, double stern_sam, double delta1, double delta2, double[:] r_vdw, double[:] sdiel):

    cdef Py_ssize_t idx, i, j, k, l
    cdef int ndim = coords.shape[0]
    cdef int natm = atom_coords.shape[0]
    cdef double zmin = np.min(coords[0,2])

    cdef double [:,:] dist = np.empty((natm, ndim), dtype=np.float64)
    cdef double [:,:] erf_list = np.empty((natm, ndim), dtype=np.float64)
    cdef double [:] x = np.empty(ndim, dtype=np.float64)
    cdef double [:] z = np.empty(ndim, dtype=np.float64)

    cdef double [:] gauss = np.empty(ndim, dtype=np.float64)
    cdef double [:] er = np.empty(ndim, dtype=np.float64)
    cdef double [:] eps_z = np.empty(ndim, dtype=np.float64)
    

    cdef double [:] xj = np.empty(ndim, dtype=np.float64)
    cdef double [:] sdielB = np.empty(ndim, dtype=np.float64)

    cdef double [:,:] grad_eps = np.zeros((ndim, 3), dtype=np.float64)

    for k in prange(ndim, nogil=True, num_threads=OMP_NUM_THREADS):
        x[k] = (coords[k,2] - zmin - stern_sam) / delta1
        eps_z[k] = eps_sam + 0.5e0 * (eps - eps_sam) * (1.0e0 + erf(x[k]))

    dist = distance_calculator(coords, atom_coords)

    for i in range(natm):
        for k in prange(ndim, nogil=True, num_threads=OMP_NUM_THREADS):
            erf_list[i,k] = 0.5e0 * (1.0e0 + erf((dist[i,k] - r_vdw[i] - probe) / delta2))

    for idx in range(3):
        for i in range(natm):
            for k in prange(ndim, nogil=True, num_threads=OMP_NUM_THREADS):
                er[k] = (coords[k,idx] - atom_coords[i,idx]) / dist[i,k]
                x[k] = (dist[i,k] - r_vdw[i] - probe) / delta2
                gauss[k] = exp(-x[k]**2)
            sdielB = np.ones(ndim, dtype=np.float64)
            for j in range(natm):
                if j != i:
                    for k in prange(ndim, nogil=True, num_threads=OMP_NUM_THREADS):
                        xj[k] = (dist[j,k] - r_vdw[j] - probe) / delta2
                        sdielB[k] = sdielB[k] * 0.5e0 * (1.0e0 + erf(xj[k]))
            for k in prange(ndim, nogil=True):
                grad_eps[k,idx] += (eps_z[k] - 1.0e0) / delta2 / M_PI**0.5 * er[k] * gauss[k] * sdielB[k]
        if idx == 2:
            for k in prange(ndim, nogil=True, num_threads=OMP_NUM_THREADS):
                z[k] = (coords[k,idx] - zmin - stern_sam) / delta1
                grad_eps[k,idx] += (eps - eps_sam) / delta1 / M_PI**0.5 * exp(-z[k]**2) * sdiel[k]

    return np.asarray(grad_eps)

@cython.boundscheck(False)
@cython.wraparound(False)
def db_force_helper(double[:,:] atom_coords, double[:,:] coords, double eps_sam, double eps_bulk, double probe, double stern_sam, double delta1, double delta2, double[:] r_vdw, double[:,:] dphi_opt, double[:,:] grad_bc, double[:] rho, double[:] phi_tot, double spacing, int ngrids):
    cdef Py_ssize_t idx, i, j, k, l
    cdef int ndim = coords.shape[0]
    cdef int natm = atom_coords.shape[0]

    cdef double [:] r = np.empty(3, dtype=np.float64)
    cdef double zmin = np.min(coords[:,2])
    cdef double RAx, RBx

    cdef double [:,:] dist = np.empty((natm, ndim), dtype=np.float64)
    cdef double [:,:] erf_list = np.empty((natm, ndim), dtype=np.float64)
    cdef double [:,:] gauss_list = np.empty((natm, ndim), dtype=np.float64)

    cdef double [:,:] x = np.empty((natm, ndim), dtype=np.float64)

    cdef double [:,:] Fdb = np.zeros((natm, 3), dtype=np.float64)
    cdef np.ndarray placeholder = np.zeros(3, dtype=np.float64)
    cdef double [:,:] depsdRA = np.empty((ndim, 3), dtype=np.float64)
    cdef double [:] ddepsdRA = np.empty(ndim, dtype=np.float64)
    cdef double [:] _sas = np.empty(ndim, dtype=np.float64)
    cdef double [:] __sas = np.empty(ndim, dtype=np.float64)
    cdef double [:] sas_sum = np.empty(ndim, dtype=np.float64)
    cdef double [:] eps_z = np.empty(ndim, dtype=np.float64)
    cdef double [:,:] _nabla = np.empty((ndim, 3), dtype=np.float64)

    cdef double [:,:] er = np.empty((ndim, 3), dtype=np.float64)
    cdef double [:] _er1 = np.empty(ndim, dtype=np.float64)
    cdef double [:] _er2 = np.empty(ndim, dtype=np.float64)
    cdef double [:] _er3 = np.empty(ndim, dtype=np.float64)
    cdef double [:] line1 = np.empty(ndim, dtype=np.float64)
    cdef double [:] line2 = np.empty(ndim, dtype=np.float64)
    cdef double [:] line3 = np.empty(ndim, dtype=np.float64)

    for k in prange(ndim, nogil=True, num_threads=OMP_NUM_THREADS):
        eps_z[k] = eps_sam + 0.5e0 * (eps_bulk - eps_sam) * (1.0e0 + erf((coords[k,2] - zmin - stern_sam) / delta1))

    dist = distance_calculator(coords, atom_coords)
    
    for i in range(natm):
        for k in prange(ndim, nogil=True, num_threads=OMP_NUM_THREADS):
            erf_list[i,k] = 0.5e0 * (1.0e0 + erf((dist[i,k] - r_vdw[i] - probe) / delta2))
            gauss_list[i,k] = exp(-((dist[i,k] - r_vdw[i] - probe) / delta2)**2)
            x[i,k] = (dist[i,k] - r_vdw[i] - probe) / delta2

    for i in range(natm):
        r = atom_coords[i]
        depsdRA = np.zeros((ndim, 3), dtype=np.float64)
        ddepsdRA = np.zeros(ndim, dtype=np.float64)
        _sas = np.ones(ndim)
        for j in range(natm):
            if j != i:
                for k in range(ndim):
                    _sas[k] *= erf_list[j,k]
        for idx in range(3):
            RAx = atom_coords[i][idx]
            for k in prange(ndim, nogil=True, num_threads=OMP_NUM_THREADS):
                _er1[k] = 1.0e0 / dist[i,k] - (coords[k,idx] - RAx)**2 / dist[i,k]**3
                line1[k] = -(eps_z[k] - 1.0e0) / delta2 / sqrt(M_PI) * _er1[k] * gauss_list[i,k] * _sas[k]
                _er2[k] = (coords[k,idx] - RAx) / dist[i,k]
                line2[k] = 2.0e0*(eps_z[k] - 1.0e0) / delta2**2 / sqrt(M_PI) * _er2[k]**2 * x[i,k] * gauss_list[i,k] * _sas[k]
            sas_sum = np.zeros(ndim, dtype=np.float64)
            for j in range(natm):
                if j != i:
                    RBx = atom_coords[j][idx]
                    for k in prange(ndim, nogil=True, num_threads=OMP_NUM_THREADS):
                        _er3[k] = (coords[k,idx] - RBx) / dist[j,k]
                    __sas = np.ones(ndim, dtype=np.float64)
                    for l in range(natm):
                        if l != i and l != j:
                            for k in range(ndim):
                                __sas[k] *= erf_list[l,k]
                    for k in range(ndim):
                        sas_sum[k] += _er3[k] * gauss_list[j,k] * __sas[k]
            for k in prange(ndim, nogil=True, num_threads=OMP_NUM_THREADS):
                line3[k] = -(eps_z[k] - 1.0e0) / delta2**2 / sqrt(M_PI) * (coords[k,idx] - RAx) * gauss_list[i,k] * sas_sum[k]
            for k in range(ndim):
                ddepsdRA[k] += line1[k] + line2[k] + line3[k]
            if idx == 2:
                for k in range(ndim):
                    ddepsdRA[k] += -(eps_bulk - eps_sam) / delta1 / delta2 / M_PI * exp(-((coords[k,2] - zmin - stern_sam) / delta1)**2) * (coords[k,idx] - RAx) / dist[i,k] * gauss_list[i,k] * _sas[k]
        er = np.zeros((ndim, 3), dtype=np.float64)
        for idx in range(3):
            for k in prange(ndim, nogil=True, num_threads=OMP_NUM_THREADS):
                er[k,idx] = (coords[k,idx] - r[idx]) / dist[i,k]
                depsdRA[k,idx] = er[k,idx] * (-1.0e0) * (eps_z[k] - 1.0e0) / delta2 / sqrt(M_PI) * gauss_list[i,k] * _sas[k]
                _nabla[k,idx] = (dphi_opt[k,idx] + grad_bc[k,idx]) * ddepsdRA[k] - 4.0e0*M_PI*depsdRA[k,idx]*rho[k]
        placeholder = -0.125e0 / M_PI * np.dot(phi_tot, _nabla) * spacing**3
        for idx in range(3):
            Fdb[i,idx] = placeholder[idx]
    
    return np.asarray(Fdb)