import numpy
import scipy
import fcdft
import os
import ctypes
from fcdft.solvent.pbe import KB2HARTREE
from pyscf.data.nist import BOHR
from pyscf import lib

libpbe = lib.load_library(os.path.join(fcdft.__path__[0], 'lib', 'libpbe'))

PI = numpy.pi
SQRT3 = numpy.sqrt(3)

def make_grad_sas(solvent_obj):
    mol = solvent_obj.mol
    coords = solvent_obj.grids.coords
    ngrids = solvent_obj.grids.ngrids
    atomic_radii = solvent_obj.get_atomic_radii()
    probe = solvent_obj.probe/ BOHR
    delta2 = solvent_obj.delta2 / BOHR
    atom_coords = mol.atom_coords()
    natm = mol.natm

    r = atom_coords[:,None,:]
    rp = coords - r
    dist = scipy.spatial.distance.cdist(atom_coords, coords)
    x = (dist - atomic_radii[:,None] - probe) / delta2
    _erf = scipy.special.erf(x)
    erf_list = 0.5e0 * (1.0e0 + _erf)
    er = rp / dist[:,:,None]
    gauss = numpy.exp(-x**2)
    grad_list = numpy.multiply(er, gauss[:,:,None], out=er) / (delta2 * numpy.sqrt(PI))

    drv = libpbe.grad_sas_drv
    grad_sas = numpy.empty((ngrids**3, 3), dtype=numpy.float64, order='C')
    c_erf_list = erf_list.ctypes.data_as(ctypes.c_void_p)
    c_grad_list = grad_list.ctypes.data_as(ctypes.c_void_p)
    c_delta2 = ctypes.c_double(delta2)
    c_ngrids = ctypes.c_int(ngrids)
    c_natm = ctypes.c_int(natm)
    c_grad_sas = grad_sas.ctypes.data_as(ctypes.c_void_p)
    drv(c_erf_list, c_grad_list, c_delta2, c_ngrids, c_natm, c_grad_sas)

    return grad_sas

def make_lap_sas(solvent_obj):
    mol = solvent_obj.mol
    coords = solvent_obj.grids.coords
    ngrids = solvent_obj.grids.ngrids
    atomic_radii = solvent_obj.get_atomic_radii()
    probe = solvent_obj.probe / BOHR
    delta2 = solvent_obj.delta2 / BOHR
    atom_coords = mol.atom_coords()
    natm = mol.natm

    dist = scipy.spatial.distance.cdist(atom_coords, coords)
    x = (dist - atomic_radii[:,None] - probe) / delta2
    _erf = scipy.special.erf(x)
    erf_list = 0.5e0 * (1.0e0 + _erf)

    r = atom_coords[:,None,:]
    rp = coords - r
    er = rp / dist[:,:,None]
    gauss = numpy.exp(-x**2)
    grad_list = numpy.multiply(er, gauss[:,:,None], out=er) / (delta2 * numpy.sqrt(PI))

    drv = libpbe.lap_sas_drv
    lap_sas = numpy.empty(ngrids**3, dtype=numpy.float64, order='C')
    c_erf_list = erf_list.ctypes.data_as(ctypes.c_void_p)
    c_grad_list = grad_list.ctypes.data_as(ctypes.c_void_p)
    c_x = x.ctypes.data_as(ctypes.c_void_p)
    c_delta2 = ctypes.c_double(delta2)
    c_ngrids = ctypes.c_int(ngrids)
    c_natm = ctypes.c_int(natm)
    c_lap_sas = lap_sas.ctypes.data_as(ctypes.c_void_p)
    
    drv(c_erf_list, c_grad_list, c_x, c_delta2, c_ngrids, c_natm, c_lap_sas)

    return lap_sas

def phi_a_finder(cost_func, jac, bottom):
    # Bisection method for a good initial guess
    phi_a = scipy.optimize.bisect(cost_func, 0.0, bottom, xtol=1e-15, maxiter=20, 
                                  full_output=False, disp=False)
    # Newton method for an accurate result
    phi_a = scipy.optimize.newton(cost_func, phi_a, fprime=jac, tol=1e-15, 
                                  maxiter=1000)
    return phi_a

def one_to_one_bc(solvent_obj, ngrids, spacing, bias, stern_sam, T, eps_sam, eps, sas, pzc, ref_pot, jump_coeff):
    """Boundary condition generator for 1:1 electrolyte by the the Gouy-Chapman-Stern model.

    Args:
        solvent_obj (:class:`PBE`): Solvent object.
        ngrids (int): Number of grid points along each axis.
        spacing (float): Grid spacing.
        bias (float): Bias Potential.
        stern (float): Stern layer by self-assembled monolayer.
        kappa (float): Debye inverse screening length.
        T (float): Temperature.
        eps_sam (float): Dielectric constant of the self-assembled monolayer.
        eps (float): Dielectric constant of the bulk solvent.

    Returns:
        1D numpy.ndarray, 1D numpy.ndarray, float: Boundary values, electrostatic potential 
        before applying solvent-accessible surface, and the potential slope in the Stern layer.
    """
    bottom = jump_coeff * (bias - (ref_pot - pzc))
    phi_z = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
    kappa = solvent_obj.kappa
    if bottom == 0.0e0:
        return phi_z.ravel()*sas, phi_z.ravel(), 0.0e0

    def cost_func(x):
        func = -2.0e0*KB2HARTREE*T*kappa*eps*numpy.sinh(-x/2.0e0/KB2HARTREE/T) - eps_sam*((bottom-x)/stern_sam)
        return func
    
    def jac(x):
        return kappa*eps*numpy.cosh(x/2.0e0/KB2HARTREE/T) + eps_sam/stern_sam

    # Jump boundary condition
    phi_a = phi_a_finder(cost_func, jac, bottom)

    # Continuous potential condition
    slope = (phi_a - bottom) / stern_sam
    z = numpy.arange(ngrids, dtype=numpy.float64) * spacing
    idx = z <= stern_sam
    phi_z[:,:,idx] = slope*z[idx]+bottom
    phi_z[:,:,~idx] = -4.0e0*KB2HARTREE*T*numpy.arctanh(numpy.exp(-kappa*(z[~idx]-stern_sam))
                                                         *numpy.tanh(-phi_a/4.0e0/KB2HARTREE/T))
    phi_z = phi_z.ravel()

    return phi_z*sas, phi_z, slope

def one_to_one_bc_grad(solvent_obj, ngrids, spacing, T, slope, phi_z, sas):
    """Analytic gradient of the boundary conditions for 1:1 Electrolyte.

    Args:
        solvent_obj (:class:`PBE`): Solvent object.
        ngrids (int): Number of grid points along each axis.
        spacing (float): Grid spacing.
        T (float): Temperature.
        slope (float): Negative of electric field inside the Stern layer.
        phi_z (1D numpy.ndarray): Boundary value before applying the solvent-accessible surface.
        sas (1D numpy.ndarray): Solvent-accessible surface.

    Returns:
        2D numpy.ndarray, 2D numpy.ndarray, 2D numpy.ndarray: Analytic gradient of the boundary values, 
        analytic gradient of the electrostatic potential before applying solvent-accessible surface, 
        and analytic gradient of the solvent-accessible surface.
    """
    dphidz = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
    _phi_z = phi_z.reshape((ngrids,)*3)
    stern_sam = solvent_obj.stern_sam / BOHR

    kappa = solvent_obj.kappa
    z = numpy.arange(ngrids, dtype=numpy.float64) * spacing
    idx = z <= stern_sam
    dphidz[:,:,idx] = slope
    dphidz[:,:,~idx] = 2.0e0*KB2HARTREE*T*kappa*numpy.sinh(-_phi_z[:,:,~idx]/2.0e0/KB2HARTREE/T)
    dphidz = dphidz.ravel()

    grad_phi_z = numpy.zeros((ngrids**3,3), dtype=numpy.float64)
    grad_phi_z[:,2] = dphidz

    grad_sas = make_grad_sas(solvent_obj)

    grad_bc = grad_phi_z * sas[:,None]
    grad_bc += grad_sas * phi_z[:,None]
    
    return grad_bc, grad_phi_z, grad_sas

def one_to_one_bc_lap(solvent_obj, ngrids, spacing, T, phi_z, grad_phi_z, sas, grad_sas):
    """Laplacian of the boundary values for 1:1 Electrolyte.

    Args:
        solvent_obj (:class:`PBE`): Solvent object.
        ngrids (int): Number of grid points along each axis.
        spacing (float): Grid spacing.
        T (float): Temperature.
        phi_z (1D numpy.ndarray): Boundary value before applying the solvent-accessible surface.
        grad_phi_z (2D numpy.ndarray): Gradient of the boundary values before applying the solvent accessible surface.
        sas (1D numpy.ndarray): Solvent-accessible surface.
        grad_sas (2D numpy.ndarray): Gradient of the solvent-accessible surface.

    Returns:
        1D numpy.ndarray: Laplacian of the boundary values.
    """
    d2phidz2 = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
    _phi_z = phi_z.reshape((ngrids,)*3)
    _grad_phi_z = grad_phi_z[:,2].reshape((ngrids,)*3)

    # Laplacian of phi(z)
    stern_sam = solvent_obj.stern_sam / BOHR
    kappa = solvent_obj.kappa
    z = numpy.arange(ngrids, dtype=numpy.float64) * spacing
    idx = z > stern_sam
    d2phidz2[:,:,idx] = -kappa*numpy.cosh(-_phi_z[:,:,idx]/2.0e0/KB2HARTREE/T)*_grad_phi_z[:,:,idx]
    d2phidz2 = d2phidz2.ravel()
    lap_sas = make_lap_sas(solvent_obj)

    lap_bc = d2phidz2*sas + phi_z*lap_sas
    lap_bc += 2.0*(grad_phi_z * grad_sas).sum(axis=1)

    return lap_bc

def two_to_one_bc(solvent_obj, ngrids, spacing, bias, stern_sam, T, eps_sam, eps, sas, pzc, ref_pot, jump_coeff):
    bottom = jump_coeff * (bias - (ref_pot - pzc))
    phi_z = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
    kappa = solvent_obj.kappa
    if bottom == 0.0e0:
        return phi_z.ravel()*sas, phi_z.ravel(), 0.0e0

    def cost_func(x):
        t = numpy.exp(-x/KB2HARTREE/T)
        s = kappa * numpy.sqrt(2*t + 1)
        if x == 0:
            return 1e100
        func = (1/(s - SQRT3*kappa) - 1/(s + SQRT3*kappa))*eps_sam*((x - bottom)/stern_sam)
        func -= eps*((SQRT3*s*KB2HARTREE*T)/(kappa*t))
        return func

    def jac(x):
        t = numpy.exp(-x/KB2HARTREE/T)
        s = kappa * numpy.sqrt(2*t + 1)
        dsdt = kappa**2 / s
        dtdx = -t / KB2HARTREE / T
        dsdx = dsdt * dtdx
        grad = -(1/((s - SQRT3*kappa))**2 - 1/(s+SQRT3*kappa)**2)*dsdx*eps_sam*((x-bottom)/stern_sam)
        grad += (1/(s-SQRT3*kappa) - 1/(s+SQRT3*kappa))*eps_sam/stern_sam
        grad -= eps*((SQRT3*KB2HARTREE*T)/(kappa*t))*dsdx
        grad += eps*((SQRT3*s*KB2HARTREE*T)/(kappa*t**2))*dtdx
        return grad
    
    # Jump boundary condition
    phi_a = phi_a_finder(cost_func, jac, bottom)

    # Continuous potential condition
    slope = (phi_a - bottom) / stern_sam
    z = numpy.arange(ngrids, dtype=numpy.float64) * spacing
    idx = z <= stern_sam
    ta = numpy.exp(-phi_a/KB2HARTREE/T)
    sa = kappa*numpy.sqrt(2*ta + 1)
    phi_z[:,:,idx] = slope*z[idx]+bottom
    C = numpy.log(abs(sa - SQRT3*kappa)/(sa + SQRT3*kappa))
    if bottom < 0:
        s = SQRT3*kappa*((1+numpy.exp(-SQRT3*kappa*(z[~idx]-stern_sam)+C))/(1-numpy.exp(-SQRT3*kappa*(z[~idx]-stern_sam)+C)))
    else:
        s = SQRT3*kappa*((1-numpy.exp(-SQRT3*kappa*(z[~idx]-stern_sam)+C))/(1+numpy.exp(-SQRT3*kappa*(z[~idx]-stern_sam)+C)))
    t = 0.5*((s/kappa)**2 - 1)
    phi_z[:,:,~idx] = -KB2HARTREE*T*numpy.log(t)
    phi_z = phi_z.ravel()
    return phi_z*sas, phi_z, slope

def two_to_one_bc_grad(solvent_obj, ngrids, spacing, T, slope, phi_z, sas):
    dphidz = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
    _phi_z = phi_z.reshape((ngrids,)*3)
    stern_sam = solvent_obj.stern_sam / BOHR

    kappa = solvent_obj.kappa
    z = numpy.arange(ngrids, dtype=numpy.float64) * spacing
    idx = z <= stern_sam
    dphidz[:,:,idx] = slope
    t = numpy.exp(-_phi_z[:,:,~idx] / KB2HARTREE / T)
    s = kappa*numpy.sqrt(2*t + 1)
    dphidz[:,:,~idx] = SQRT3*s*KB2HARTREE*T/kappa/t/((s - SQRT3*kappa)**-1 - (s + SQRT3*kappa)**-1)
    dphidz = dphidz.ravel()

    grad_phi_z = numpy.zeros((ngrids**3, 3), dtype=numpy.float64)
    grad_phi_z[:,2] = dphidz

    grad_sas = make_grad_sas(solvent_obj)

    grad_bc = grad_phi_z * sas[:,None]
    grad_bc += grad_sas * phi_z[:,None]

    return grad_bc, grad_phi_z, grad_sas

def two_to_one_bc_lap(solvent_obj, ngrids, spacing, T, phi_z, grad_phi_z, sas, grad_sas):
    """Laplacian of the boundary values.

    Args:
        solvent_obj (:class:`PBE`): Solvent object.
        ngrids (int): Number of grid points along each axis.
        spacing (float): Grid spacing.
        T (float): Temperature.
        phi_z (1D numpy.ndarray): Boundary value before applying the solvent-accessible surface.
        grad_phi_z (2D numpy.ndarray): Gradient of the boundary values before applying the solvent accessible surface.
        sas (1D numpy.ndarray): Solvent-accessible surface.
        grad_sas (2D numpy.ndarray): Gradient of the solvent-accessible surface.

    Returns:
        1D numpy.ndarray: Laplacian of the boundary values.
    """
    d2phidz2 = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
    _phi_z = phi_z.reshape((ngrids,)*3)
    _grad_phi_z = grad_phi_z[:,2].reshape((ngrids,)*3)

    # Laplacian of phi(z)
    stern_sam = solvent_obj.stern_sam / BOHR
    kappa = solvent_obj.kappa
    z = numpy.arange(ngrids, dtype=numpy.float64) * spacing
    idx = z > stern_sam
    t = numpy.exp(-_phi_z[:,:,idx] / KB2HARTREE / T)
    s = kappa*numpy.sqrt(2*t + 1)
    dsdt = kappa**2 / s
    dtdx = -t / KB2HARTREE / T
    dxdz = _grad_phi_z[:,:,idx]
    dsdz = dsdt * dtdx * dxdz
    dtdz = dtdx * dxdz
    
    d2phidz2[:,:,idx] = SQRT3*KB2HARTREE*T/kappa*(dsdz/t-dtdz*s/t**2)/(1/(s-SQRT3*kappa)-1/(s+SQRT3*kappa))
    d2phidz2[:,:,idx] += (1/(s-SQRT3*kappa)+1/(s+SQRT3*kappa))*dsdz*dxdz

    d2phidz2 = d2phidz2.flatten()
    lap_sas = make_lap_sas(solvent_obj)

    lap_bc = d2phidz2*sas + phi_z*lap_sas
    lap_bc += 2.0*(grad_phi_z * grad_sas).sum(axis=1)

    return lap_bc

def one_to_two_bc(solvent_obj, ngrids, spacing, bias, stern_sam, T, eps_sam, eps, sas, pzc, ref_pot, jump_coeff):
    bottom = jump_coeff * (bias - (ref_pot - pzc))
    phi_z = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
    kappa = solvent_obj.kappa
    if bottom == 0.0e0:
        return phi_z.ravel()*sas, phi_z.ravel(), 0.0e0

    def cost_func(x):
        t = numpy.exp(x/KB2HARTREE/T)
        s = kappa * numpy.sqrt(2*t + 1)
        if x == 0:
            return -1e100
        func = (1/(s - SQRT3*kappa) - 1/(s + SQRT3*kappa))*eps_sam*((x - bottom)/stern_sam)
        func += eps*((SQRT3*s*KB2HARTREE*T)/(kappa*t))
        return func

    def jac(x):
        t = numpy.exp(x/KB2HARTREE/T)
        s = kappa * numpy.sqrt(2*t + 1)
        dsdt = kappa**2 / s
        dtdx = t / KB2HARTREE / T
        dsdx = dsdt * dtdx
        grad = -(1/((s - SQRT3*kappa))**2 - 1/(s+SQRT3*kappa)**2)*dsdx*eps_sam*((x-bottom)/stern_sam)
        grad += (1/(s-SQRT3*kappa) - 1/(s+SQRT3*kappa))*eps_sam/stern_sam
        grad += eps*((SQRT3*KB2HARTREE*T)/(kappa*t))*dsdx
        grad -= eps*((SQRT3*s*KB2HARTREE*T)/(kappa*t**2))*dtdx
        return grad
    
    # Jump boundary condition
    phi_a = phi_a_finder(cost_func, jac, bottom)

    # Continuous potential condition
    slope = (phi_a - bottom) / stern_sam
    z = numpy.arange(ngrids, dtype=numpy.float64) * spacing
    idx = z <= stern_sam
    ta = numpy.exp(phi_a/KB2HARTREE/T)
    sa = kappa*numpy.sqrt(2*ta + 1)
    phi_z[:,:,idx] = slope*z[idx]+bottom
    C = numpy.log(abs(sa - SQRT3*kappa)/(sa + SQRT3*kappa))
    if bottom < 0:
        s = SQRT3*kappa*((1-numpy.exp(-SQRT3*kappa*(z[~idx]-stern_sam)+C))/(1+numpy.exp(-SQRT3*kappa*(z[~idx]-stern_sam)+C)))
    else:
        s = SQRT3*kappa*((1+numpy.exp(-SQRT3*kappa*(z[~idx]-stern_sam)+C))/(1-numpy.exp(-SQRT3*kappa*(z[~idx]-stern_sam)+C)))
    t = 0.5*((s/kappa)**2 - 1)
    phi_z[:,:,~idx] = KB2HARTREE*T*numpy.log(t)
    phi_z = phi_z.ravel()
    return phi_z*sas, phi_z, slope

def one_to_two_bc_grad(solvent_obj, ngrids, spacing, T, slope, phi_z, sas):
    dphidz = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
    _phi_z = phi_z.reshape((ngrids,)*3)
    stern_sam = solvent_obj.stern_sam / BOHR

    kappa = solvent_obj.kappa
    z = numpy.arange(ngrids, dtype=numpy.float64) * spacing
    idx = z <= stern_sam
    dphidz[:,:,idx] = slope
    t = numpy.exp(_phi_z[:,:,~idx] / KB2HARTREE / T)
    s = kappa*numpy.sqrt(2*t + 1)
    dphidz[:,:,~idx] = -SQRT3*s*KB2HARTREE*T/kappa/t/((s - SQRT3*kappa)**-1 - (s + SQRT3*kappa)**-1)
    dphidz = dphidz.ravel()

    grad_phi_z = numpy.zeros((ngrids**3, 3), dtype=numpy.float64)
    grad_phi_z[:,2] = dphidz

    grad_sas = make_grad_sas(solvent_obj)

    grad_bc = grad_phi_z * sas[:,None]
    grad_bc += grad_sas * phi_z[:,None]

    return grad_bc, grad_phi_z, grad_sas

def one_to_two_bc_lap(solvent_obj, ngrids, spacing, T, phi_z, grad_phi_z, sas, grad_sas):
    d2phidz2 = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
    _phi_z = phi_z.reshape((ngrids,)*3)
    _grad_phi_z = grad_phi_z[:,2].reshape((ngrids,)*3)

    # Laplacian of phi(z)
    stern_sam = solvent_obj.stern_sam / BOHR
    kappa = solvent_obj.kappa
    z = numpy.arange(ngrids, dtype=numpy.float64) * spacing
    idx = z > stern_sam
    t = numpy.exp(_phi_z[:,:,idx] / KB2HARTREE / T)
    s = kappa*numpy.sqrt(2*t + 1)
    dsdt = kappa**2 / s
    dtdx = t / KB2HARTREE / T
    dxdz = _grad_phi_z[:,:,idx]
    dsdz = dsdt * dtdx * dxdz
    dtdz = dtdx * dxdz
    
    d2phidz2[:,:,idx] = -SQRT3*KB2HARTREE*T/kappa*(dsdz/t-dtdz*s/t**2)/(1/(s-SQRT3*kappa)-1/(s+SQRT3*kappa))
    d2phidz2[:,:,idx] += (1/(s-SQRT3*kappa)+1/(s+SQRT3*kappa))*dsdz*dxdz

    d2phidz2 = d2phidz2.flatten()
    lap_sas = make_lap_sas(solvent_obj)

    lap_bc = d2phidz2*sas + phi_z*lap_sas
    lap_bc += 2.0*(grad_phi_z * grad_sas).sum(axis=1)

    return lap_bc