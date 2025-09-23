import numpy
import scipy
from fcdft.solvent.pbe import HARTREE2EV, KB2HARTREE
from pyscf.data.nist import BOHR
from fcdft.lib import pbe_helper

PI = numpy.pi
SQRT3 = numpy.sqrt(3)

def make_grad_sas(solvent_obj):
    mol = solvent_obj.mol
    coords = solvent_obj.grids.coords
    atomic_radii = solvent_obj.get_atomic_radii()
    probe = solvent_obj.probe/ BOHR
    delta2 = solvent_obj.delta2 / BOHR
    atom_coords = mol.atom_coords()
    grad_sas = numpy.zeros_like(coords)
    r = atom_coords[:,None,:]
    rp = coords - r
    dist = pbe_helper.distance_calculator(coords, atom_coords)
    x = (dist - atomic_radii[:,None] - probe) / delta2
    _erf = scipy.special.erf(x)
    erf_list = 0.5e0 * (1.0e0 + _erf)
    er = pbe_helper.product_atom_vector_scalar(rp, 1.0e0/dist)
    for i in range(mol.natm):
        mask = numpy.ones(mol.natm, dtype=bool)
        mask[i] = False
        erf = numpy.prod(erf_list[mask], axis=0)
        gauss = numpy.exp(-x[i]**2)
        coeff = 1.0e0 / delta2 / numpy.sqrt(PI) * gauss * erf
        grad_sas += pbe_helper.product_vector_scalar(er[i], coeff)
    return grad_sas

def make_lap_sas(solvent_obj):
    mol = solvent_obj.mol
    coords = solvent_obj.grids.coords
    atomic_radii = solvent_obj.get_atomic_radii()
    probe = solvent_obj.probe / BOHR
    delta2 = solvent_obj.delta2 / BOHR
    atom_coords = mol.atom_coords()

    r = atom_coords[:,None,:]
    rp = coords - r
    dist = pbe_helper.distance_calculator(coords, atom_coords)
    x = (dist - atomic_radii[:,None] - probe) / delta2
    _erf = scipy.special.erf(x)
    erf_list = 0.5e0 * (1.0e0 + _erf)
    er = pbe_helper.product_atom_vector_scalar(rp, 1.0e0/dist)
    gauss = numpy.exp(-x**2)
    grad_list = pbe_helper.product_atom_vector_scalar(er, 1.0e0/delta2/numpy.sqrt(PI)*gauss)
    lap_sas = pbe_helper.lap_sas(erf_list, grad_list, x, delta2)

    return lap_sas

def phi_a_finder(cost_func, jac, bottom):
    # Bisection method for a good initial guess
    phi_a = scipy.optimize.bisect(cost_func, 0.0, bottom, xtol=1e-15, maxiter=50, 
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
    idx = numpy.array(range(phi_z.shape[2]))
    z = numpy.asarray(idx*spacing, dtype=numpy.float64)
    idx1 = numpy.argwhere(z<=stern_sam)
    idx2 = numpy.argwhere(z>stern_sam)
    phi_z[:,:,idx1] = slope*z[idx1]+bottom
    phi_z[:,:,idx2] = -4.0e0*KB2HARTREE*T*numpy.arctanh(numpy.exp(-kappa*(z[idx2]-stern_sam))
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
    idx = numpy.array(range(dphidz.shape[2]))
    z = numpy.asarray(idx*spacing, dtype=numpy.float64)
    idx1 = numpy.argwhere(z<=stern_sam)
    idx2 = numpy.argwhere(z>stern_sam)
    dphidz[:,:,idx1] = slope
    dphidz[:,:,idx2] = 2.0e0*KB2HARTREE*T*kappa*numpy.sinh(-_phi_z[:,:,idx2]/2.0e0/KB2HARTREE/T)
    dphidz = dphidz.ravel()

    grad_phi_z = numpy.zeros((ngrids**3,3), dtype=numpy.float64)
    grad_phi_z[:,2] = dphidz

    grad_sas = make_grad_sas(solvent_obj)

    grad_bc = pbe_helper.product_vector_scalar(grad_phi_z, sas) + pbe_helper.product_vector_scalar(grad_sas, phi_z)
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
    idx = numpy.array(range(d2phidz2.shape[2]))
    z = numpy.asarray(idx*spacing, dtype=numpy.float64)
    idx2 = numpy.argwhere(z>stern_sam)
    d2phidz2[:,:,idx2] = -kappa*numpy.cosh(-_phi_z[:,:,idx2]/2.0e0/KB2HARTREE/T)*_grad_phi_z[:,:,idx2]
    d2phidz2 = d2phidz2.ravel()
    lap_sas = make_lap_sas(solvent_obj)
    lap_bc = d2phidz2*sas + phi_z*lap_sas + 2.0e0 * pbe_helper.product_vector_vector(grad_phi_z, grad_sas)

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
        grad = (1/((s - SQRT3*kappa))**2 - 1/(s+SQRT3*kappa)**2)*dsdx*eps_sam*((x-bottom)/stern_sam)
        grad += (1/(s-SQRT3*kappa) - 1/(s+SQRT3*kappa))*eps_sam/stern_sam
        grad -= eps*((SQRT3*KB2HARTREE*T)/(kappa*t))*dsdx
        grad += eps*((SQRT3*s*KB2HARTREE*T)/(kappa*t**2))*dtdx
        return grad
    
    # Jump boundary condition
    phi_a = phi_a_finder(cost_func, jac, bottom)

    # Continuous potential condition
    slope = (phi_a - bottom) / stern_sam
    idx = numpy.array(range(phi_z.shape[2]))
    z = numpy.asarray(idx*spacing, dtype=numpy.float64)
    idx1 = numpy.where(z<=stern_sam)
    idx2 = numpy.where(z>stern_sam)
    ta = numpy.exp(-phi_a/KB2HARTREE/T)
    sa = kappa*numpy.sqrt(2*ta + 1)
    phi_z[:,:,idx1] = slope*z[idx1]+bottom
    if bottom < 0:
        C = numpy.log((sa - SQRT3*kappa)/(sa + SQRT3*kappa))
        s = SQRT3*kappa*((1+numpy.exp(-SQRT3*kappa*z[idx2]+C))/(1-numpy.exp(-SQRT3*kappa*z[idx2]+C)))
    else:
        C = numpy.log((SQRT3*kappa - sa)/(sa + SQRT3*kappa))
        s = SQRT3*kappa*((1-numpy.exp(-SQRT3*kappa*z[idx2]+C))/(1+numpy.exp(-SQRT3*kappa*z[idx2]+C)))
    t = 0.5*((s/kappa)**2 - 1)
    phi_z[:,:,idx2] = -KB2HARTREE*T*numpy.log(t)
    phi_z = phi_z.ravel()
    return phi_z*sas, phi_z, slope

def two_to_one_bc_grad(solvent_obj, ngrids, spacing, T, slope, phi_z, sas):
    dphidz = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
    _phi_z = phi_z.reshape((ngrids,)*3)
    stern_sam = solvent_obj.stern_sam / BOHR

    kappa = solvent_obj.kappa
    idx = numpy.array(range(dphidz.shape[2]))
    z = numpy.asarray(idx*spacing, dtype=numpy.float64)
    idx1 = numpy.where(z<=stern_sam)
    idx2 = numpy.where(z>stern_sam)
    dphidz[:,:,idx1] = slope
    t = numpy.exp(-_phi_z[:,:,idx2] / KB2HARTREE / T)
    s = kappa*numpy.sqrt(2*t + 1)
    dphidz[:,:,idx2] = SQRT3*s*KB2HARTREE*T/kappa/t/((s - SQRT3*kappa)**-1 - (s + SQRT3*kappa)**-1)
    dphidz = dphidz.ravel()

    grad_phi_z = numpy.zeros((ngrids**3, 3), dtype=numpy.float64)
    grad_phi_z[:,2] = dphidz

    grad_sas = make_grad_sas(solvent_obj)

    grad_bc = pbe_helper.product_vector_scalar(grad_phi_z, sas) + pbe_helper.product_vector_scalar(grad_sas, phi_z)
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
    idx = numpy.array(range(d2phidz2.shape[2]))
    z = numpy.asarray(idx*spacing, dtype=numpy.float64)
    idx = numpy.where(z>stern_sam)
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
    lap_bc = d2phidz2*sas + phi_z*lap_sas + 2.0e0 * pbe_helper.product_vector_vector(grad_phi_z, grad_sas)

    return lap_bc