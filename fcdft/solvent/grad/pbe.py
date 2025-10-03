import numpy
import scipy
import fcdft
from fcdft.solvent.pbe import M2HARTREE, KB2HARTREE
from fcdft.lib import pbe_helper
from pyscf.solvent._attach_solvent import _Solvation
from pyscf.grad import rhf as rhf_grad
from pyscf import lib
from pyscf.lib import logger
from pyscf.data.nist import *

PI = numpy.pi

def make_grad_object(grad_method):
    '''For grad_method in vacuum, add nuclear gradients of solvent pcmobj'''

    # Zeroth order method object must be a solvation-enabled method
    assert isinstance(grad_method.base, _Solvation)
    if grad_method.base.with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not available for energy gradients')

    name = (grad_method.base.with_solvent.__class__.__name__
            + grad_method.__class__.__name__)
    return lib.set_class(WithSolventGrad(grad_method),
                         (WithSolventGrad, grad_method.__class__), name)

class WithSolventGrad:
    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventGrad, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj
    
    def kernel(self, *args, dm=None, **kwargs):
        dm = kwargs.pop('dm', None)
        if dm is None:
            dm = self.base.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]

        self.de_solvent = kernel(self.base.with_solvent, dm)
        self.de_solute = super().kernel(*args, **kwargs)
        self.de = self.de_solute + self.de_solvent
        
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s (+%s) gradients ---------------',
                        self.base.__class__.__name__,
                        self.base.with_solvent.__class__.__name__)
            rhf_grad._write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')
        return self.de
    
    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass


def kernel(solvent_obj, dm, verbose=None):
    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]

    Frf = rf_force(solvent_obj, dm) # Reaction field force
    Fdb = db_force(solvent_obj, dm) # Dielectric boundary force
    Fib = ib_force(solvent_obj, dm) # Ion boundary force
    if solvent_obj.verbose >= logger.NOTE:
        logger.note(solvent_obj, '------------------ Reaction field force -----------------')
        rhf_grad._write(solvent_obj, solvent_obj.mol, Frf, range(solvent_obj.mol.natm))
        logger.note(solvent_obj, '---------------------------------------------------------')
        logger.note(solvent_obj, '--------------- Dielectric boundary force ---------------')
        rhf_grad._write(solvent_obj, solvent_obj.mol, Fdb, range(solvent_obj.mol.natm))
        logger.note(solvent_obj, '---------------------------------------------------------')
        logger.note(solvent_obj, '------------------ Ionic boundary force -----------------')
        rhf_grad._write(solvent_obj, solvent_obj.mol, Fib, range(solvent_obj.mol.natm))
        logger.note(solvent_obj, '---------------------------------------------------------')
    de = -Frf - Fdb - Fib # Sign convention
    return de

def rf_force(solvent_obj, dm):
    mol = solvent_obj.mol
    phi_tot = solvent_obj.phi_tot
    phi_sol = solvent_obj.phi_sol

    coords = solvent_obj.grids.coords
    spacing = solvent_obj.grids.spacing
    ngrids = solvent_obj.grids.ngrids

    atom_coords = mol.atom_coords()

    # RESP Section
    esp_mol = mol.copy()
    esp_mol.charge = mol.charge + (mol.nelectron - solvent_obj.nelectron) # Inject fractional charge
    from fcdft.solvent.esp import esp_atomic_charges
    options_dict = {'RESP_MAXITER': 100, 'RESP_TOLERANCE': 1.0e-10}
    qesp = esp_atomic_charges(esp_mol, dm.real, options_dict=options_dict, gpu_accel=solvent_obj.gpu_accel)

    atmlst = range(mol.natm)
    logger.info(solvent_obj, '-------------- RESP Atomic Charge ------------')
    for k, ia in enumerate(atmlst):
        logger.info(solvent_obj, '%d %s  %15.10f', ia, mol.atom_symbol(ia), qesp[k])
    logger.note(solvent_obj, '----------------------------------------------')
    phi_pol = phi_tot - phi_sol
    atom_coords = mol.atom_coords()

    Frf = numpy.zeros((mol.natm, 3))
    origin = coords.min(axis=0)
    for i in range(mol.natm):
        r = atom_coords[i]
        ratio = (r - origin) / spacing
        Nx = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
        Ny = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
        Nz = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
        for dim in range(3):
            x = ratio[dim]
            mylist = numpy.array(list(range(ngrids)))
            idx1 = numpy.argwhere(numpy.logical_and(mylist-1.5e0 <= x, x <= mylist-0.5e0))
            val1 = 0.125e0 + 0.5e0*(x-idx1+1.0e0) + 0.5e0*(x-idx1+1.0e0)**2
            idx2 = numpy.argwhere(numpy.logical_and(mylist-0.5e0 <= x, x <= mylist+0.5e0))
            val2 = 0.75e0 - (x-idx2)**2
            idx3 = numpy.argwhere(numpy.logical_and(mylist+0.5e0 <= x, x <= mylist+1.5e0))
            val3 = 0.125e0 - 0.5e0*(x-idx3-1.0e0) + 0.5e0*(x-idx3-1.0e0)**2
            if dim == 0:
                Nx[idx1,:,:] = val1
                Nx[idx2,:,:] = val2
                Nx[idx3,:,:] = val3
            elif dim == 1:
                Ny[:,idx1,:] = val1
                Ny[:,idx2,:] = val2
                Ny[:,idx3,:] = val3
            elif dim == 2:
                Nz[:,:,idx1] = val1
                Nz[:,:,idx2] = val2
                Nz[:,:,idx3] = val3

        dNxdx = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
        dNydy = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
        dNzdz = numpy.zeros((ngrids,)*3, dtype=numpy.float64)
        for dim in range(3):
            x = ratio[dim]
            mylist = numpy.array(list(range(ngrids)))
            idx1 = numpy.argwhere(numpy.logical_and(mylist-1.5e0 <= x, x <= mylist-0.5e0))
            val1 = 0.5e0 + (x-idx1+1.0e0)
            idx2 = numpy.argwhere(numpy.logical_and(mylist-0.5e0 <= x, x <= mylist+0.5e0))
            val2 = -2.0e0 * (x-idx2)
            idx3 = numpy.argwhere(numpy.logical_and(mylist+0.5e0 <= x, x <= mylist+1.5e0))
            val3 = -0.5e0 + (x-idx3-1.0e0)
            if dim == 0:
                dNxdx[idx1,:,:] = val1
                dNxdx[idx2,:,:] = val2
                dNxdx[idx3,:,:] = val3
            elif dim == 1:
                dNydy[:,idx1,:] = val1
                dNydy[:,idx2,:] = val2
                dNydy[:,idx3,:] = val3
            elif dim == 2:
                dNzdz[:,:,idx1] = val1
                dNzdz[:,:,idx2] = val2
                dNzdz[:,:,idx3] = val3
        Nx, Ny, Nz = Nx.ravel(), Ny.ravel(), Nz.ravel()
        dNxdx, dNydy, dNzdz = dNxdx.ravel(), dNydy.ravel(), dNzdz.ravel()
        dNxdRx, dNydRy, dNzdRz = dNxdx/spacing, dNydy/spacing, dNzdz/spacing
        grad = numpy.column_stack((dNxdRx*Ny*Nz, Nx*dNydRy*Nz, Nx*Ny*dNzdRz))
        dqdR = qesp[i] * grad
        Frf[i] = numpy.dot(phi_pol, dqdR)
    Frf = -Frf
    return Frf

def db_force(solvent_obj, dm):
    mol = solvent_obj.mol
    phi_tot = solvent_obj.phi_tot
    delta1 = solvent_obj.delta1 / BOHR
    delta2 = solvent_obj.delta2 / BOHR
    r_vdw = solvent_obj.get_atomic_radii()
    coords = solvent_obj.grids.coords
    spacing = solvent_obj.grids.spacing
    ngrids = solvent_obj.grids.ngrids
    eps_bulk = solvent_obj.eps
    eps_sam = solvent_obj.eps_sam
    probe = solvent_obj.probe / BOHR
    stern_sam = solvent_obj.stern_sam / BOHR
    T = solvent_obj.T
    atom_coords = mol.atom_coords()

    _intermediates = solvent_obj._intermediates
    sas = _intermediates['sas']
    eps = _intermediates['eps']
    grad_eps = _intermediates['grad_eps']
    bias = solvent_obj.bias / HARTREE2EV
    stern_sam = solvent_obj.stern_sam / BOHR
    rho_tot = solvent_obj.rho_sol + solvent_obj.rho_ions
    rho_pol = solvent_obj.rho_pol
    pzc = solvent_obj.pzc / HARTREE2EV
    ref_pot = solvent_obj.ref_pot / HARTREE2EV
    jump_coeff = solvent_obj.jump_coeff

    impose_bc, bc_grad, _ = solvent_obj._gen_boundary_conditions()
    bc, phi_z, slope = impose_bc(solvent_obj, ngrids, spacing, bias, stern_sam, T, eps_sam, eps_bulk, sas, pzc, ref_pot, jump_coeff)
    grad_bc, _, _ = bc_grad(solvent_obj, ngrids, spacing, T, slope, phi_z, sas)
    phi_opt = phi_tot - bc
    solver = solvent_obj.solver
    nproc = lib.num_threads()
    phi_optk = None
    if isinstance(solver, fcdft.solvent.solver.fft2d):
        phi_optk = scipy.fft.fftn(phi_opt.reshape((ngrids,)*3), axes=(0,1), workers=nproc)
    dphi_opt = solver.gradient(phi_opt, phi_optk, ngrids, spacing)

    grad_lneps = pbe_helper.product_vector_scalar(grad_eps, 1.0e0/eps)
    rho_iter_bc = 0.25e0 / PI * pbe_helper.product_vector_vector(grad_lneps, grad_bc)

    Fdb = pbe_helper.db_force_helper(atom_coords, coords, eps_sam, eps_bulk, probe, stern_sam, delta1, delta2, r_vdw, dphi_opt, grad_bc, rho_tot+rho_pol+rho_iter_bc, phi_tot, spacing, ngrids)

    return Fdb

def ib_force(solvent_obj, dm):
    mol = solvent_obj.mol
    phi_tot = solvent_obj.phi_tot
    delta1 = solvent_obj.delta1 / BOHR
    delta2 = solvent_obj.delta2 / BOHR
    cb = solvent_obj.cb * M2HARTREE
    stern_mol = solvent_obj.stern_mol / BOHR
    stern_sam = solvent_obj.stern_sam / BOHR
    T = solvent_obj.T
    probe = solvent_obj.probe / BOHR

    lambda_r = solvent_obj._intermediates['lambda_r']
    r_vdw = solvent_obj.get_atomic_radii()

    cation_rad = solvent_obj.cation_rad / BOHR
    anion_rad = solvent_obj.anion_rad / BOHR

    coords = solvent_obj.grids.coords
    spacing = solvent_obj.grids.spacing

    if cb == 0.0e0:
        return Fib

    equiv = solvent_obj.equiv
    if equiv == 11:
        Fib = one_to_one_ib_force(mol, coords, spacing, phi_tot, cb, lambda_r, delta1, delta2,
                                  stern_sam, stern_mol, T, probe, r_vdw, cation_rad, anion_rad)
    elif equiv == 21:
        Fib = two_to_one_ib_force(mol, coords, spacing, phi_tot, cb, lambda_r, delta1, delta2,
                                  stern_sam, stern_mol, T, probe, r_vdw, cation_rad, anion_rad)
    else:
        raise NotImplementedError

    return Fib

def one_to_one_ib_force(mol, coords, spacing, phi_tot, cb, lambda_r, delta1, delta2,
                        stern_sam, stern_mol, T, probe, r_vdw, cation_rad, anion_rad):
    zmin = coords[:,2].min()
    x = (coords[:,2] - zmin - stern_sam - stern_mol) / delta1
    _erf = scipy.special.erf(x)
    erf_z = 0.5e0 * (1.0e0 + _erf) # given in ngrids

    atom_coords = mol.atom_coords()
    Fib = numpy.zeros((mol.natm, 3))
    if cb == 0.0e0:
        return Fib
    
    dist = pbe_helper.distance_calculator(coords, atom_coords)
    x = (dist - r_vdw[:,None] - probe - stern_mol) / delta2
    _erf = scipy.special.erf(x)
    erf_list = 0.5e0 * (1.0e0 + _erf)
    gauss_list = numpy.exp(-x**2)

    c12 = 0.74e0 / (4.0e0/3.0e0 * PI * (cation_rad**3 + anion_rad**3))
    lnexp = phi_tot / KB2HARTREE / T
    idx = abs(lnexp) < 230.96
    t = numpy.ones_like(lnexp) * 1.0e100
    t[idx] = numpy.cosh(lnexp[idx])
    
    for i in range(mol.natm):
        r = atom_coords[i]
        rp = coords - r
        er = pbe_helper.product_vector_scalar(rp, 1.0e0/dist[i])
        mask = [False if j == i else True for j in range(mol.natm)]
        erf = numpy.prod(erf_list[mask], axis=0)
        gauss_A = gauss_list[i]
        gauss_A[x[i] < -8.0e0*delta2] = 0.0e0 # To ensure zero contribution inside the cavity.
        dl = -1.0e0 / delta2 / numpy.sqrt(PI) * pbe_helper.product_vector_scalar(er, erf_z*gauss_A*erf)
        Fib[i] = numpy.dot(1.0e0 / ((c12 / cb - 1.0e0) / t + lambda_r), dl)    

    Fib = 2.0*c12*KB2HARTREE*T*Fib*spacing**3
    return Fib

def two_to_one_ib_force(mol, coords, spacing, phi_tot, cb, lambda_r, delta1, delta2,
                        stern_sam, stern_mol, T, probe, r_vdw, cation_rad, anion_rad):
    zmin = coords[:,2].min()
    x = (coords[:,2] - zmin - stern_sam - stern_mol) / delta1
    _erf = scipy.special.erf(x)
    erf_z = 0.5e0 * (1.0e0 + _erf) # given in ngrids

    atom_coords = mol.atom_coords()
    Fib = numpy.zeros((mol.natm, 3))
    if cb == 0.0e0:
        return Fib
    
    dist = pbe_helper.distance_calculator(coords, atom_coords)
    x = (dist - r_vdw[:,None] - probe - stern_mol) / delta2
    _erf = scipy.special.erf(x)
    erf_list = 0.5e0 * (1.0e0 + _erf)
    gauss_list = numpy.exp(-x**2)

    c12 = 0.74e0 / (4.0e0/3.0e0 * PI * (2*cation_rad**3 + anion_rad**3))
    lnexp = phi_tot / KB2HARTREE / T
    idx = abs(1.5*lnexp) < 230.96
    t = numpy.ones_like(lnexp) * 1.0e100
    t[idx] = numpy.exp(0.5*lnexp[idx]) * (2*numpy.exp(-1.5*lnexp[idx]) + numpy.exp(1.5*lnexp[idx])) / 3

    for i in range(mol.natm):
        r = atom_coords[i]
        rp = coords - r
        er = pbe_helper.product_vector_scalar(rp, 1.0e0/dist[i])
        mask = [False if j == i else True for j in range(mol.natm)]
        erf = numpy.prod(erf_list[mask], axis=0)
        gauss_A = gauss_list[i]
        gauss_A[x[i] < -8.0e0*delta2] = 0.0e0 # To ensure zero contribution inside the cavity.
        dl = -1.0e0 / delta2 / numpy.sqrt(PI) * pbe_helper.product_vector_scalar(er, erf_z*gauss_A*erf)
        Fib[i] = numpy.dot(1.0e0 / ((c12 / cb - 1.0e0) / t + lambda_r), dl)

    Fib = 3.0*c12*KB2HARTREE*T*Fib*spacing**3
    return Fib

def one_to_two_ib_force(mol, coords, spacing, phi_tot, cb, lambda_r, delta1, delta2,
                        stern_sam, stern_mol, T, probe, r_vdw, cation_rad, anion_rad):
    zmin = coords[:,2].min()
    x = (coords[:,2] - zmin - stern_sam - stern_mol) / delta1
    _erf = scipy.special.erf(x)
    erf_z = 0.5e0 * (1.0e0 + _erf) # given in ngrids

    atom_coords = mol.atom_coords()
    Fib = numpy.zeros((mol.natm, 3))
    if cb == 0.0e0:
        return Fib
    
    dist = pbe_helper.distance_calculator(coords, atom_coords)
    x = (dist - r_vdw[:,None] - probe - stern_mol) / delta2
    _erf = scipy.special.erf(x)
    erf_list = 0.5e0 * (1.0e0 + _erf)
    gauss_list = numpy.exp(-x**2)

    c12 = 0.74e0 / (4.0e0/3.0e0 * PI * (cation_rad**3 + 2*anion_rad**3))
    lnexp = phi_tot / KB2HARTREE / T
    idx = abs(1.5*lnexp) < 230.96
    t = numpy.ones_like(lnexp) * 1.0e100
    t[idx] = numpy.exp(-0.5*lnexp[idx]) * (numpy.exp(-1.5*lnexp[idx]) + 2*numpy.exp(1.5*lnexp[idx])) / 3

    for i in range(mol.natm):
        r = atom_coords[i]
        rp = coords - r
        er = pbe_helper.product_vector_scalar(rp, 1.0e0/dist[i])
        mask = [False if j == i else True for j in range(mol.natm)]
        erf = numpy.prod(erf_list[mask], axis=0)
        gauss_A = gauss_list[i]
        gauss_A[x[i] < -8.0e0*delta2] = 0.0e0 # To ensure zero contribution inside the cavity.
        dl = -1.0e0 / delta2 / numpy.sqrt(PI) * pbe_helper.product_vector_scalar(er, erf_z*gauss_A*erf)
        Fib[i] = numpy.dot(1.0e0 / ((c12 / cb - 1.0e0) / t + lambda_r), dl)

    Fib = 3.0*c12*KB2HARTREE*T*Fib*spacing**3

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(
        atom='''
C       -1.0593548629      0.0004770015      2.3819666439
C       -1.1227505948      0.0005046591      3.7712685135
C        0.1763262203     -0.0000784303      1.7095427085
H       -2.0867905118      0.0009377059      4.2924557406
C        0.0787297071     -0.0000350288      4.5259272468
C        1.3625460462     -0.0006115258      2.4520088957
H        2.3347698110     -0.0010483968      1.9462013728
C        1.3221246435     -0.0005931916      3.8476681322
H        2.2505789007     -0.0010100507      4.4299723474
H       -1.9867216086      0.0008946167      1.7931461762
C        0.0357615279     -0.0000158902      5.9596145884
N        0.0000000000      0.0000000000      7.1396106167
S        0.0000000000      0.0000000000      0.0000000000
H        1.3222704832     -0.0005811676     -0.3021302181''',
        basis='6-31g**', verbose=5, max_memory=10000)
    from fcdft.wbl.rks import *
    wblmf = WBLMoleculeRKS(mol, xc='b3lyp', broad=0.01, smear=0.2, nelectron=70.00)
    wblmf.pot_cycle=100
    wblmf.max_cycle=300
    wblmf.kernel()
    dm = wblmf.make_rdm1()
    from fcdft.solvent.pbe import *
    cm = PBE(mol, cb=1.0, ngrids=41, length=15, stern_sam=-100)
    cm._dm = dm
    solmf = pbe_for_scf(wblmf, cm)
    from pyscf.geomopt.geometric_solver import optimize
    solmf_opt = optimize(solmf, maxsteps=100)