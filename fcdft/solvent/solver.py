import numpy
import scipy
import os
import fcdft
from fcdft.solvent import calculus_helper as ch
from pyscf import lib
from pyscf.lib import logger
import ctypes

PI = numpy.pi
TWO_PI = 2.0e0*PI
nproc = lib.num_threads()
libpbe = lib.load_library(os.path.join(fcdft.__path__[0], 'lib', 'libpbe'))

def fft2d_solve(solver, rho, L, ngrids, spacing, kpts):
    """Poisson equation solver using FFT

    Args:
        solver (_type_): solver object
        rho (_type_): Charge density in real space
        L (numpy.ndarray): Positive definite Laplacian without grid spacing.
        ngrids (int): Grid points along each axis.
        spacing (float): Grid spacing
        kpts (numpy.ndarray): k-points
    """
    lap = -L / spacing**2
    # Charge density in the Fourier space
    rhok = scipy.fft.fftn(rho.reshape((ngrids,)*3), axes=(0,1), workers=nproc)
    # The sign convention below corresponds converting 4pi (make_phi) to -4pi
    rhok = -rhok.flatten()
    # Electrostatic potential in the Fourier space
    phik = numpy.empty(ngrids**3, dtype=numpy.complex128, order='C')
    info = ctypes.c_int(0)

    drv = solver.drv

    c_rhok = rhok.ctypes.data_as(ctypes.c_void_p)
    c_kpts = kpts.ctypes.data_as(ctypes.c_void_p)
    c_phik = phik.ctypes.data_as(ctypes.c_void_p)
    c_info = ctypes.byref(info)
    c_lap = lap.ctypes.data_as(ctypes.c_void_p)
    c_spacing = ctypes.c_double(spacing)
    c_ngrids = ctypes.c_int(ngrids)

    drv(c_rhok, c_lap, c_ngrids, c_spacing, c_kpts, c_phik, c_info)

    if info.value != 0:
        raise RuntimeError('LAPACKE_dsyev failed.')
    phi = scipy.fft.ifftn(phik.reshape((ngrids,)*3), axes=(0,1), workers=nproc)

    return phi.real.flatten(), phik

def nabla_reciprocal(kpts, spacing):
    k = -1.0j*(1/140*numpy.sin(2.0*PI*4*kpts) - 8/105*numpy.sin(2.0*PI*3*kpts) + 2/5*numpy.sin(2.0*PI*2*kpts) - 8/5*numpy.sin(2.0*PI*kpts)) / spacing
    return k

def laplacian_reciprocal(kpts, spacing):
    k = -(1/280*numpy.cos(TWO_PI*4*kpts) - 16/315*numpy.cos(TWO_PI*3*kpts) + 2/5*numpy.cos(TWO_PI*2*kpts) - 16/5*numpy.cos(TWO_PI*kpts) + 205/72) / spacing**2
    return k
    
class multigrid(lib.StreamObject):
    def __init__(self, ngrids, spacing, verbose, stdout=None):
        self.ngrids = ngrids
        self.spacing = spacing
        self.verbose = verbose
        self.stdout = stdout
        self.drv = None
        self.L = None

    def build(self, ngrids=None):
        if ngrids is None:
            ngrids = self.ngrids
        prm_precond = {'coarsening': 'smoothed_aggregation', 'max_levels': 8, 'ncycle': 2}
        prm_solver = {'type': 'cg', 'tol': 1.0e-8, 'maxiter': 1000}
        self.L= ch.poisson((ngrids,)*3, format='csr')
        import pyamgcl
        self.drv = pyamgcl.solver(pyamgcl.amgcl(self.L, prm=prm_precond), prm=prm_solver)

    def gradient(self, phi, phik, ngrids=None, spacing=None):
        """Gradient under Dirichlet boundary conditions

        Args:
            phi (_type_): _description_
            ngrids (_type_, optional): _description_. Defaults to None.
            spacing (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if ngrids is None:
            ngrids = self.ngrids
        if spacing is None:
            spacing = self.spacing
        dphi = ch.vectorize_grad(ch.gradient(phi.reshape((ngrids,)*3))) / spacing
        return dphi
    
    def laplacian(self, phi, phik, ngrids=None, spacing=None, L=None):
        if ngrids is None:
            ngrids = self.ngrids
        if spacing is None:
            spacing = self.spacing
        if L is None:
            L = self.L
        return -L.dot(phi) / spacing**2

    
    def solve(self, rho, ngrids=None, spacing=None):
        if ngrids is None:
            ngrids = self.ngrids
        if spacing is None:
            spacing = self.spacing
        phi = self.drv(rho*spacing**2)
        return phi, phi 
    
    def whoareyou(self):
        logger.info(self, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
        logger.info(self, ' |  Poisson-Boltzmann Solver with the Multigrid Scheme  |')
        logger.info(self, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')

class fft2d(lib.StreamObject):
    def __init__(self, ngrids, spacing, verbose, stdout=None):
        self.ngrids = ngrids
        self.spacing = spacing
        self.verbose = verbose
        self.stdout = stdout
        self.kpts = None
        self.drv = None
        self.L = None

    def build(self, ngrids=None):
        if ngrids is None:
            ngrids = self.ngrids
        self.kpts = numpy.fft.fftfreq(ngrids)
        self.drv = libpbe.poisson_fft_2d
        self.L = ch.poisson((ngrids,)).toarray()
        logger.info(self, 'cond(L) = %s', numpy.linalg.cond(self.L))

    def gradient(self, phi, phik, ngrids=None, spacing=None):
        """Gradient under 2-dimensional periodic boundary conditions

        Args:
            phi (_type_): _description_
            ngrids (_type_, optional): _description_. Defaults to None.
            spacing (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if ngrids is None:
            ngrids = self.ngrids
        if spacing is None:
            spacing = self.spacing
        kpts = self.kpts

        if phik is None:
            return numpy.zeros((ngrids**3, 3), dtype=numpy.float64)

        k = nabla_reciprocal(kpts, spacing)
        kx, ky = numpy.meshgrid(k, k, indexing='ij')
        _phik = phik.reshape((ngrids,)*3)
        # Gradient in the Fourier space
        grad_kx = kx[:,:,None] * _phik
        grad_ky = ky[:,:,None] * _phik
        grad_kz = ch.gradient(_phik)[2] / spacing

        # Gradient in the real space
        grad_x = scipy.fft.ifftn(grad_kx, axes=(0,1), workers=nproc).real.flatten()
        grad_y = scipy.fft.ifftn(grad_ky, axes=(0,1), workers=nproc).real.flatten()
        grad_z = scipy.fft.ifftn(grad_kz, axes=(0,1), workers=nproc).real.flatten()

        dphi = numpy.hstack([grad_x.reshape(-1,1), grad_y.reshape(-1,1), grad_z.reshape(-1,1)])
        return dphi

    def laplacian(self, phi, phik, ngrids=None, spacing=None, L=None):
        # This function calculates the Laplacian of phi in k-space, not the charge density.
        # To get the charge density, a factor of -0.25 / PI has to be multiplied 
        # after taking inverse Fourier transformation.
        if ngrids is None:
            ngrids = self.ngrids
        if spacing is None:
            spacing = self.spacing
        if L is None:
            L = self.L
        kpts = self.kpts
        lap = -L / spacing**2

        drv = libpbe.laplacian_2d
        rhok = numpy.empty(ngrids**3, dtype=numpy.complex128, order='C')
        c_phik = phik.ctypes.data_as(ctypes.c_void_p)
        c_kpts = kpts.ctypes.data_as(ctypes.c_void_p)
        c_rhok = rhok.ctypes.data_as(ctypes.c_void_p)
        c_lap = lap.ctypes.data_as(ctypes.c_void_p)
        c_spacing = ctypes.c_double(spacing)
        c_ngrids = ctypes.c_int(ngrids)

        drv(c_phik, c_lap, c_ngrids, c_spacing, c_kpts, c_rhok)

        rho = scipy.fft.ifftn(rhok.reshape((ngrids,)*3), axes=(0,1), workers=nproc).real
        return rho.flatten()

    def solve(self, rho, ngrids=None, spacing=None):
        if ngrids is None:
            ngrids = self.ngrids
        if spacing is None:
            spacing = self.spacing
        kpts = self.kpts
        L = self.L
        return fft2d_solve(self, rho, L, ngrids, spacing, kpts)
    
    def whoareyou(self):
        logger.info(self, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
        logger.info(self, ' |  Poisson-Boltzmann Solver with the 2D-FFT Scheme  |')
        logger.info(self, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')