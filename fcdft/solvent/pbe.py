import numpy
import scipy
import os
import ctypes
import fcdft
import fcdft.solvent.calculus_helper as ch

from fcdft.solvent import _attach_solvent
from pyscf.solvent import ddcosmo
from pyscf.tools import cubegen
from pyscf import lib
from pyscf.lib import logger
from pyscf.lib import pack_tril
from pyscf.data.nist import *
from pyscf.data.radii import VDW
from pyscf import df
from pyscf import gto

libpbe = lib.load_library(os.path.join(fcdft.__path__[0], 'lib', 'libpbe'))

try:
    OMP_NUM_THREADS = os.environ['OMP_NUM_THREADS']
except KeyError:
    OMP_NUM_THREADS = 1
PI = numpy.pi
KB2HARTREE = BOLTZMANN / HARTREE2J
M2HARTREE = AVOGADRO*BOHR**3*1.e-27

def pbe_for_scf(mf, solvent_obj=None, dm=None):
    """
    Attach PBE solvation model to a SCF object.

    Creates a self-consistent-field calculator that includes solvation effects
    via the non-linear Poisson-Boltzmann model.

    Parameters
    ----------
    mf : pyscf.scf.RHF/RKS or pyscf.scf.UHF/UKS
        PySCF SCF mean-field object.
    solvent_obj : PBE, optional
        PBE solvation object. If None, creates a default PBE(mf.mol).
    dm : ndarray, optional
        Initial density matrix for solvation. Default: None (computed from mf).

    Returns
    -------
    solmf : pyscf.solvent.PCMSolver.SCFWithPolarization
        SCF object with solvation effects included via PBE.

    Examples
    --------
    >>> from pyscf import gto, dft
    >>> from fcdft.solvent.pbe import PBE, pbe_for_scf
    >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='6-31g')
    >>> mf = dft.RKS(mol, xc='b3lyp')
    >>> cm = PBE(mol, cb=1.0, length=15, ngrids=41)
    >>> cm.eps = 78.3553  # Water
    >>> solmf = pbe_for_scf(mf, cm)
    >>> solmf.kernel()
    """
    if solvent_obj is None:
        solvent_obj = PBE(mf.mol)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)

def gen_pbe_solver(solvent_obj, verbose=None):
    """
    Get the PBE solver driver function.

    Parameters
    ----------
    solvent_obj : PBE
        PBE solvation object.
    verbose : int, optional
        Logger verbosity.

    Returns
    -------
    get_vind : callable
        Function to compute solvation contribution to effective potential.
    """
    return solvent_obj._get_vind

def make_lambda(solvent_obj, mol, probe, stern_mol, stern_sam, coords, delta1, delta2, atomic_radii):
    """
    Ion-exclusion function for Stern layer region.

    Constructs a smooth function λ(r) that transitions from 0 (no ions near molecule/SAM)
    to 1 (bulk solution). Uses error function to smoothly interpolate across both the
    molecular Stern layer and the SAM Stern layer.

    Parameters
    ----------
    solvent_obj : PBE
        PBE solvation object.
    mol : gto.Mole
        Molecule specification.
    probe : float
        Solvent probe radius in a.u.
    stern_mol : float
        Stern layer thickness at molecule surface in a.u.
    stern_sam : float
        Stern layer thickness at SAM in a.u.
    coords : ndarray, shape (n_grid, 3)
        Grid point coordinates in Cartesian (a.u.).
    delta1 : float
        Broadening width for SAM Stern layer erf in a.u.
    delta2 : float
        Broadening width for molecular Stern layer erf in a.u.
    atomic_radii : ndarray, shape (n_atoms,)
        Atomic van der Waals radii in a.u.

    Returns
    -------
    lambda_r : ndarray, shape (n_grid,)
        Ion-exclusion function (0 in Stern, 1 in bulk).
    """
    atom_coords = mol.atom_coords()
    # Molecular Stern Layer
    dist = scipy.spatial.distance.cdist(atom_coords, coords)
    x = (dist - atomic_radii[:,None] - probe - stern_mol) / delta2
    erf_list = 0.5e0*(1.0e0 + scipy.special.erf(x))
    erf_list[x < -8.0e0*delta2] = 0.0e0
    lambda_r = numpy.prod(erf_list, axis=0)

    # SAM Stern Layer
    zmin = coords[:,2].min()
    x = (coords[:,2] - zmin - (stern_sam + stern_mol)) / delta1
    _erf = scipy.special.erf(x)
    _erf[x < -8.0e0*delta1] = -1.0e0 # Value suppression
    lambda_z = 0.5e0 * (1.0e0 + _erf)
    lambda_r = lambda_z * lambda_r
    return lambda_r

def make_sas(solvent_obj, mol, probe, coords, delta2, atomic_radii):
    """
    Construct solvent-accessible surface (SAS) function.

    Returns a smooth function that is 0 inside the SAS (within atomic radii + probe)
    and 1 in the bulk solvent, with smooth transition via error function.

    Parameters
    ----------
    solvent_obj : PBE
        PBE solvation object.
    mol : gto.Mole
        Molecule specification.
    probe : float
        Solvent probe radius (water: 1.4 Å = 2.64 a.u.) in a.u.
    coords : ndarray, shape (n_grid, 3)
        Grid point coordinates in Cartesian (a.u.).
    delta2 : float
        Broadening width for SAS erf in a.u.
    atomic_radii : ndarray, shape (n_atoms,)
        Atomic van der Waals radii in a.u.

    Returns
    -------
    sas : ndarray, shape (n_grid,)
        Solvent-accessible surface function (0=inside, 1=bulk).
    """
    atom_coords = mol.atom_coords()
    dist = scipy.spatial.distance.cdist(atom_coords, coords)
    x = (dist - atomic_radii[:,None] - probe) / delta2
    _erf = scipy.special.erf(x)
    erf_list = 0.5e0 * (1.0e0 + _erf)
    S = numpy.prod(erf_list, axis=0)
    return S

def make_grad_sas(solvent_obj, mol, probe, coords, delta2, atomic_radii):
    # mol = solvent_obj.mol
    # coords = solvent_obj.grids.coords
    ngrids = solvent_obj.grids.ngrids
    # atomic_radii = solvent_obj.get_atomic_radii()
    # probe = solvent_obj.probe/ BOHR
    # delta2 = solvent_obj.delta2 / BOHR
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

def make_lap_sas(solvent_obj, mol, probe, coords, delta2, atomic_radii):
    # mol = solvent_obj.mol
    # coords = solvent_obj.grids.coords
    ngrids = solvent_obj.grids.ngrids
    # atomic_radii = solvent_obj.get_atomic_radii()
    # probe = solvent_obj.probe / BOHR
    # delta2 = solvent_obj.delta2 / BOHR
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

def make_eps(solvent_obj, coords, eps_sam, eps, stern_sam, delta1, sas):
    """
    Compute position-dependent dielectric function.

    The dielectric varies smoothly from the SAM (ε_SAM) through the Stern layer
    to the bulk solvent (ε_bulk) using an error function transition:

        ε(z) = ε_SAM + (ε_bulk - ε_SAM)/2 * (1 + erf((z - z_stern) / Δ_1))
        ε(r) = ε_0 + (ε(z) - ε_0) * SAS

    Parameters
    ----------
    solvent_obj : PBE
        PBE solvation object.
    coords : ndarray, shape (n_grid, 3)
        Grid point coordinates in Cartesian (a.u.).
    eps_sam : float
        Dielectric constant of the SAM region.
    eps : float
        Dielectric constant of bulk solvent.
    stern_sam : float
        z-coordinate of SAM/Stern boundary in a.u.
    delta1 : float
        Broadening width of erf transition in a.u.
    sas : ndarray, shape (n_grid,)
        Solvent-accessible surface function.

    Returns
    -------
    eps_r : ndarray, shape (n_grid,)
        Position-dependent dielectric function at each grid point.
    """
    zmin = coords[:,2].min()
    x = (coords[:,2] - zmin - stern_sam) / delta1
    _erf = scipy.special.erf(x)
    eps_z = eps_sam + 0.5e0 * (eps - eps_sam) * (1.0e0 + _erf)
    eps_r = 1.0e0 + (eps_z - 1.0e0) * sas
    return eps_r

def make_grad_eps(solvent_obj, mol, coords, eps_sam, eps, probe, stern_sam, delta1, delta2, atomic_radii, sas):
    """Generates analytic gradient of the dielectric function.

    Args:
        mol (pyscf.gto.Mole): Mole object.
        coords (2D numpy.ndarray): Cartesian coordinates.
        eps_sam (float): Dielectric constant of the self-assembled monolayer.
        eps (float): Dielectric constant of the bulk solvent.
        probe (float): Probe radius.
        stern_sam (float): Thickness of the Stern layer formed by the self-assembled monolayer.
        delta1 (float): Width of error function along z-axis.
        delta2 (float): Width of error function used for molecular part.
        atomic_radii (1D numpy.ndarray): Atomic radii.
        sas (1D numpy.ndarray): Solvent-accessible surface.

    Returns:
        2D numpy.ndarray: Gradient of the dielectric function.
    """
    ngrids = solvent_obj.grids.ngrids
    natm = mol.natm
    atom_coords = mol.atom_coords()
    zmin = coords[:,2].min()
    x = (coords[:,2] - zmin - stern_sam) / delta1
    _erf = scipy.special.erf(x)
    eps_z = eps_sam + 0.5e0 * (eps - eps_sam) * (1.0e0 + _erf)
    exp_z = numpy.exp(-x**2)

    dist = scipy.spatial.distance.cdist(atom_coords, coords)
    x = (dist - atomic_radii[:,None] - probe) / delta2
    _erf = scipy.special.erf(x)
    erf_list = 0.5e0 * (1.0e0 + _erf)

    r = atom_coords[:,None,:]
    rp = coords - r
    x = (dist - atomic_radii[:,None] - probe) / delta2
    _erf = scipy.special.erf(x)
    erf_list = 0.5e0 * (1.0e0 + _erf)
    er = rp / dist[:,:,None]
    gauss = numpy.exp(-x**2)
    grad_list = numpy.multiply(er, gauss[:,:,None], out=er) / (delta2 * numpy.sqrt(PI))

    grad_eps = numpy.empty((ngrids**3, 3), dtype=numpy.float64, order='C')

    drv = libpbe.grad_eps_drv
    c_erf_list = erf_list.ctypes.data_as(ctypes.c_void_p)
    c_grad_list = grad_list.ctypes.data_as(ctypes.c_void_p)
    c_exp_z = exp_z.ctypes.data_as(ctypes.c_void_p)
    c_eps_z = eps_z.ctypes.data_as(ctypes.c_void_p)
    c_delta1 = ctypes.c_double(delta1)
    c_delta2 = ctypes.c_double(delta2)
    c_eps = ctypes.c_double(eps)
    c_eps_sam = ctypes.c_double(eps_sam)
    c_ngrids = ctypes.c_int(ngrids)
    c_natm = ctypes.c_int(natm)
    c_grad_eps = grad_eps.ctypes.data_as(ctypes.c_void_p)

    drv(c_erf_list, c_grad_list, c_exp_z, c_eps_z, c_delta1, c_delta2,
        c_eps, c_eps_sam, c_ngrids, c_natm, c_grad_eps)

    return grad_eps

def make_phi_sol(solvent_obj, dm=None, coords=None):
    """
    Compute the solute (molecule) electrostatic potential in vacuum.

    Solves Poisson's equation for the isolated molecule:

        φ_sol = φ_nuc + φ_elec

    where φ_nuc is the nuclear potential and φ_elec is the electronic potential
    computed from the density matrix.

    Parameters
    ----------
    solvent_obj : PBE
        PBE solvation object.
    dm : ndarray, shape (n_ao, n_ao), optional
        Density matrix. If None, uses solvent_obj._dm.
        For UHF, shape (2, n_ao, n_ao); sums α + β.
    coords : ndarray, shape (n_grid, 3), optional
        Grid coordinates. If None, uses solvent_obj.grids.coords.

    Returns
    -------
    phi_sol : ndarray, shape (n_grid,)
        Solute potential at grid points (molecule in vacuum).
        Tagged with attributes: Vnuc (nuclear), Vele (electronic).
    """
    if dm is None: dm = solvent_obj._dm
    if coords is None: coords = solvent_obj.grids.coords

    tot_ngrids = solvent_obj.grids.get_ngrids()
    
    logger.info(solvent_obj, 'Generating the solute electrostatic potential...')
    t0 = (logger.process_clock(), logger.perf_counter())
    mol = solvent_obj.mol

    atom_coords = mol.atom_coords()
    Z = mol.atom_charges()
    dist = scipy.spatial.distance.cdist(atom_coords, coords)
    dist[dist < 1.0e-100] = numpy.inf # Machine precision
    Vnuc = numpy.tensordot(1.0e0 / dist, Z, axes=([0], [0]))

    if dm.ndim == 3: # Spin-unrestricted
        dm = dm[0] + dm[1]

    dms = numpy.asarray(dm.real)
    gpu_accel = solvent_obj.gpu_accel

    if gpu_accel:
        logger.info(solvent_obj, 'Will utilize GPUs for computing the electrostatic potential.')
        import cupy
        nbatch = 256*256
        tot_ngrids = coords.shape[0]
        from gpu4pyscf.gto.int3c1e import int1e_grids
        _dm = cupy.asarray(dms)
        _Vele = cupy.zeros(tot_ngrids, order='C')
        for ibatch in range(0, tot_ngrids, nbatch):
            max_grid = min(ibatch+nbatch, tot_ngrids)
            _Vele[ibatch:max_grid] += int1e_grids(mol, coords[ibatch:max_grid], dm=_dm, direct_scf_tol=1e-14)
        Vele = _Vele.get()
        del _dm, _Vele,  cupy, int1e_grids # Release GPU memory
        lib.num_threads(OMP_NUM_THREADS) # GPU4PySCF sets OMP_NUM_THREADS=4 when running.

    else:
        Vele = numpy.empty(tot_ngrids, order='C')
        nao = mol.nao
        dm_tril = pack_tril(dms + dms.T)
        idx = numpy.arange(nao)
        idx = idx * (idx + 1) // 2 + idx
        dm_tril[idx] *= 0.5
        max_memory = solvent_obj.max_memory - lib.current_memory()[0] - Vele.nbytes*1e-6
        blksize = int(max(max_memory*.9e6/8/idx[-1], 400))
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, 'int3c2e')
        for p0, p1 in lib.prange(0, tot_ngrids, blksize):
            fakemol = gto.fakemol_for_charges(coords[p0:p1])
            ints = df.incore.aux_e2(mol, fakemol, aosym='s2ij', cintopt=cintopt)
            Vele[p0:p1] = dm_tril.dot(ints)
            del ints

    MEP = Vnuc - Vele
    t0 = logger.timer(solvent_obj, 'phi_sol', *t0)
    return lib.tag_array(MEP, Vnuc=Vnuc, Vele=-Vele)

def make_rho_sol(solvent_obj, phi_sol=None, ngrids=None, spacing=None):
    """
    Compute solute charge density from electrostatic potential via Poisson's equation.

    Uses the Laplacian of φ_sol to recover the charge density:

        ρ_sol = -∇²φ_sol / 4π

    Parameters
    ----------
    solvent_obj : PBE
        PBE solvation object.
    phi_sol : ndarray, shape (n_grid,), optional
        Solute potential. If None, uses solvent_obj.phi_sol.
    ngrids : int, optional
        Number of grid points along each axis (cubic grid).
        If None, uses solvent_obj.grids.ngrids.
    spacing : float, optional
        Grid spacing in a.u. If None, uses solvent_obj.grids.spacing.

    Returns
    -------
    rho_sol : ndarray, shape (n_grid,)
        Solute charge density at grid points.
    """
    if phi_sol is None: phi_sol = solvent_obj.phi_sol
    if spacing is None: spacing = solvent_obj.grids.spacing
    if ngrids is None: ngrids = solvent_obj.grids.ngrids
    solver = solvent_obj.solver
    nproc = lib.num_threads()
    if isinstance(solver, fcdft.solvent.solver.fft2d):
        phik = scipy.fft.fftn(phi_sol.reshape((ngrids,)*3), axes=(0,1), workers=nproc)
    else:
        phik = None

    rho_sol = -solver.laplacian(phi_sol, phik) / 4.0e0 / PI

    return rho_sol

def make_phi(solvent_obj, bias=None, phi_sol=None, rho_sol=None):
    """
    Solve the non-linear Poisson-Boltzmann equation self-consistently.

    Iteratively solves:

        ∇·(ε(r) ∇φ_tot) = -4π[ρ_sol(r) + ρ_ions(φ_tot, T)]

    where ρ_ions are computed from Boltzmann statistics with ion exclusion in
    Stern layers. Boundary conditions follow Gouy-Chapman-Stern theory at the
    electrode surface.

    Parameters
    ----------
    solvent_obj : PBE
        PBE solvation object with built grids and intermediates.
    bias : float, optional
        Applied bias potential in a.u. If None, uses solvent_obj.bias.
    phi_sol : ndarray, optional
        Solute potential in vacuum. If None, computes from density matrix.
    rho_sol : ndarray, optional
        Solute charge density. If None, computes from phi_sol.

    Returns
    -------
    phi_tot : ndarray, shape (n_grid,)
        Total electrostatic potential (boundary condition + solution).
    rho_ions : ndarray, shape (n_grid,)
        Ion charge density at convergence.
    rho_pol : ndarray, shape (n_grid,)
        Polarization charge density (dielectric response).

    Raises
    ------
    RuntimeError
        If ion charge density becomes NaN (solver divergence).
    RuntimeError
        If PBE iteration fails to converge within max_cycle.
    """
    if solvent_obj._intermediates is None: solvent_obj.build()
    _intermediates = solvent_obj._intermediates

    ngrids = solvent_obj.grids.ngrids
    tot_ngrids = solvent_obj.grids.get_ngrids()
    T = solvent_obj.T
    spacing = solvent_obj.grids.spacing
    stern_sam = solvent_obj.stern_sam / BOHR
    cb = solvent_obj.cb * M2HARTREE
    pzc = solvent_obj.pzc / HARTREE2EV
    ref_pot = solvent_obj.ref_pot / HARTREE2EV
    jump_coeff = solvent_obj.jump_coeff
    
    eps = _intermediates['eps']
    lambda_r = _intermediates['lambda_r']
    grad_eps = _intermediates['grad_eps']
    sas = _intermediates['sas']

    solver = solvent_obj.solver

    eta = 0.6e0
    kappa = 0.2e0

    t0 = (logger.process_clock(), logger.perf_counter())

    phi_tot = numpy.zeros(tot_ngrids, dtype=numpy.float64)
    impose_bc, bc_grad, bc_lap = solvent_obj._gen_boundary_conditions()
    bc, phi_z, slope= impose_bc(solvent_obj, ngrids, spacing, bias, stern_sam, T, 
                                solvent_obj.eps_sam, solvent_obj.eps, pzc, ref_pot, jump_coeff)
    grad_bc, grad_phi_z = bc_grad(solvent_obj, ngrids, spacing, T, slope, phi_z)
    lap_bc = bc_lap(solvent_obj, ngrids, spacing, T, phi_z, grad_phi_z)

    phi_tot += bc

    t0 = logger.timer(solvent_obj, 'bc', *t0)

    grad_lneps = grad_eps / eps[:,None]
    get_rho_ions = solvent_obj._gen_get_rho_ions()
    rho_ions = get_rho_ions(solvent_obj, phi_tot, cb, lambda_r, T)

    rho_tot = rho_sol + rho_ions
    rho_iter = numpy.zeros(tot_ngrids)
    rho_pol = (1.0e0 - eps) / eps * rho_tot + rho_iter

    rho_iter_bc = 0.25e0 / PI * (grad_lneps * grad_bc).sum(axis=1)

    logger.info(solvent_obj, 'Bias vs. PZC = %.15f V', (bias - (ref_pot - pzc)) * HARTREE2EV)
    solver._initialize()

    max_cycle = solvent_obj.max_cycle
    iter = 0
    phik = None
    while iter < max_cycle:
        phi_old = phi_tot
        rho_iter_old = rho_iter
        rho_ions_old = rho_ions
        rho_pol_old = rho_pol

        phi_opt = phi_old - bc
        dphi_opt = solver.gradient(phi_opt, phik, ngrids, spacing)
        rho_iter = 0.25e0 / PI * (grad_lneps * dphi_opt).sum(axis=1)

        rho_iter = eta * rho_iter + (1.0e0 - eta) * rho_iter_old

        rho_tot = rho_sol + rho_ions_old
        rho_pol = (1.0e0 - eps) / eps * rho_tot + rho_iter

        _rho = 4.0e0*PI*(rho_tot+rho_pol+rho_iter_bc) + lap_bc
        phi_opt, phik = solver.solve(_rho, ngrids, spacing)
        phi_tot = phi_opt + bc

        rho_ions = get_rho_ions(solvent_obj, phi_tot, cb, lambda_r, T)
        if numpy.isnan(rho_ions).any():
            raise RuntimeError('PBE solver encountered infinite ion charge density!')

        rho_ions = kappa * rho_ions + (1.0e0 - kappa) * rho_ions_old

        drho_pol = abs(rho_pol - rho_pol_old)
        drho_ions = abs(rho_ions - rho_ions_old)
        logger.info(solvent_obj, 'PBE Iteration %3d max|drho(pol)| = %4.3e, max|drho(ions)| = %4.3e', 
                    iter+1, drho_pol.max(), drho_ions.max())
        if numpy.all(drho_pol < solvent_obj.thresh_pol) and numpy.all(drho_ions < solvent_obj.thresh_ions) and iter > 0:
            logger.info(solvent_obj, 'PBE Converged, max|drho(pol)| = %4.3e, max|drho(ions)| = %4.3e',
                        drho_pol.max(), drho_ions.max())
            solver._finalize()
            t0 = logger.timer(solvent_obj, 'phi_tot', *t0)
            return phi_tot, rho_ions, rho_pol
        iter += 1
    logger.info(solvent_obj, 'PBE failed to converge.')
    raise RuntimeError('PBE solver failed to converge. ' \
                       'Decreasing grid size might help convergence.')

class PBE(ddcosmo.DDCOSMO):
    """
    Poisson-Boltzmann Electrolyte (PBE) solvation model for electrochemical interfaces.

    This class implements a non-linear PB solver for continuum solvation of molecules
    at electrode/liquid interfaces. It models the electric double layer via:

    1. **Dielectric function**: Position-dependent ε(r) from SAM through Stern layer to bulk
    2. **Ion concentration**: From Boltzmann distribution at the applied potential
    3. **Stern layers**: Excluded region near molecule and at SAM surface
    4. **Boundary conditions**: Gouy-Chapman-Stern conditions at the electrode

    Integrates with PySCF's DFT and wave function methods to compute solvation
    contributions to the effective potential and energy.

    Key Attributes
    ---------------
    mol : gto.Mole
        Molecule specification.
    grids : Grids
        Cubic integration grid (defined in this module).
    cb : float
        Ion concentration (cation + anion) in mol/L. Default: 0.0 (no electrolyte).
    T : float
        Temperature in Kelvin. Default: 298.15 K.
    eps : float
        Bulk solvent dielectric constant. Default: 78.3553 (water).
    eps_sam : float
        SAM dielectric constant. Default: 2.284 (benzene-like).
    stern_mol : float
        Stern layer thickness at molecule surface in Å. Default: 0.44 Å.
    stern_sam : float
        Stern layer thickness at SAM in Å. Default: 8.1 Å.
    probe : float
        Solvent probe radius in Å. Default: 1.4 Å (water).
    delta1, delta2 : float
        Broadening widths for error function transitions in Å. Default: 0.265 Å.
    cation_rad, anion_rad : float
        Hydrated ion radii in Å. Default: 4.3 Å (1:1 electrolyte).
    pzc : float
        Potential of zero charge in eV. Default: -4.8 eV (Au electrode).
    ref_pot : float
        Reference potential (electrode Fermi level) in eV for bias calculation.
    jump_coeff : float
        Jump discontinuity coefficient at electrode. Default: 0.73115 (jellium + 1M salt).
    nelectron : float
        Number of electrons (from SCF). Set by pbe_for_scf().
    bias : float
        Applied bias potential in a.u. (set by WBL or user).

    Parameters
    ----------
    mol : gto.Mole
        Molecule specification.
    cb : float, optional
        Ion concentration in mol/L. Default: 0.0.
    cation_rad : float, optional
        Cation radius in Å. Default: 4.3.
    anion_rad : float, optional
        Anion radius in Å. Default: 4.3.
    T : float, optional
        Temperature in K. Default: 298.15.
    stern_mol : float, optional
        Stern layer at molecule in Å. Default: 0.44.
    stern_sam : float, optional
        Stern layer at SAM in Å. Default: 8.1.
    equiv : int, optional
        Equivalence distance (internal use). Default: 11.
    **kwargs
        Passed to Grids constructor (length, ngrids, spacing).

    Examples
    --------
    Basic solvation with default settings:

    >>> from pyscf import gto, dft
    >>> from fcdft.solvent.pbe import PBE, pbe_for_scf
    >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='6-31g')
    >>> cm = PBE(mol)
    >>> cm.eps = 78.3553  # Water
    >>> mf = dft.RKS(mol, xc='b3lyp')
    >>> solmf = pbe_for_scf(mf, cm)
    >>> solmf.kernel()

    Electrochemical setup with WBL and applied potential:

    >>> from fcdft.wbl.rks import WBLMoleculeRKS
    >>> wbl = WBLMoleculeRKS(mol, xc='pbe', broad=0.01, ref_pot=-4.5)
    >>> wbl.kernel()
    >>> cm = PBE(mol, cb=1.0, stern_sam=3.0)
    >>> cm.eps = 78.3553
    >>> cm.eps_sam = 2.284
    >>> solmf = pbe_for_scf(wbl, cm)
    >>> solmf.kernel()
    """
    _keys = {'cb', 'T', 'bias', 'stern_sam', 'delta1', 'delta2', 'eps_sam', 'probe', 'kappa', 'stern_mol', 'cation_rad', 'anion_rad', 'rho_sol', 'rho_ions', 'rho_pol', 'phi_pol', 'phi_tot', 'phi_sol', 'L', 'nelectron', 'phi_pol', 'thresh_pol', 'thresh_ions', 'thresh_amg', 'gpu_accel', 'cycle', 'atom_bottom', 'pzc', 'jump_coeff', 'ref_pot', 'solver', 'equiv', 'custom_shift'}

    def __init__(self, mol, cb=0.0, cation_rad=4.3, anion_rad=4.3, T=298.15, stern_mol=0.44, stern_sam=8.1, equiv=11, **kwargs):
        """
        Initialize PBE solvation model.

        Parameters
        ----------
        mol : gto.Mole
            Molecule specification.
        cb : float, optional
            Ion concentration in mol/L. Default: 0.0 (no electrolyte).
        cation_rad : float, optional
            Cation hydrated radius in Å. Default: 4.3.
        anion_rad : float, optional
            Anion hydrated radius in Å. Default: 4.3.
        T : float, optional
            Temperature in K. Default: 298.15.
        stern_mol : float, optional
            Stern layer thickness at molecule in Å. Default: 0.44.
        stern_sam : float, optional
            Stern layer thickness at SAM in Å. Default: 8.1.
        equiv : int, optional
            Equivalence distance (internal). Default: 11.
        **kwargs
            Additional arguments for Grids (length, ngrids, spacing, etc.).
        """
        ddcosmo.DDCOSMO.__init__(self, mol)
        self.grids = Grids(mol, **kwargs)
        self.radii_table = VDW # in a.u.
        self.probe = 1.4 # in angstrom. Water (1.4 Å)
        self.stern_mol = stern_mol # in angstrom. Stein et al.
        self.stern_sam = stern_sam # in angstrom. SAM Stern layer length Hammes-Schiffer 2020
        self.eps_sam = 2.284 # Benzene
        self.delta1 = 0.265 # in angstrom. Arias 2005 paper
        self.delta2 = 0.265 # in angstrom. Stein et al.
        self.cb = cb # in mol/L
        self.T = T # Temperature in Kelvin.
        self.cation_rad = cation_rad # in angstrom
        self.anion_rad  = anion_rad # in angstrom
        self.pzc = -4.8 # in eV
        self.jump_coeff = 0.73115e0 # Jellium model with 1M electrolyte solution
        self.equiv = equiv

        self.kappa = None
        self.max_cycle = 200
        self.phi_tot = None
        self.phi_sol = None
        self.phi_pol = None
        self.rho_sol = None
        self.rho_ions = None
        self.rho_pol = None
        self.bias = None # Placeholder <- WBLMolecule
        self.nelectron = None # Placeholder <- WBLMolecule
        self.ref_pot = None # Placeholder <- WBLMolecule
        self.L = None
        self.thresh_pol = 1.0e-5
        self.thresh_ions = 1.0e-6
        self.thresh_amg = 1.0e-8 # Determined through numerical test. Lower threshold loses symmetry
        self.gpu_accel = False
        self.atom_bottom = None
        self.solver = None
        self.custom_shift = None # Should be given in angs
        
    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'Probe radius = %.5f Angstrom', self.probe)
        logger.info(self, 'Dielectric constant of the SAM = %.5f', self.eps_sam)
        logger.info(self, 'Dielectric constant of the solvent = %.5f', self.eps)
        logger.info(self, 'Broadening of the SAM Stern layer = %.5f Angstrom', self.delta1)
        logger.info(self, 'Broadening of the molecular Stern layer = %.5f Angstrom', self.delta2)
        logger.info(self, 'SAM Stern layer length = %.5f Angstrom', self.stern_sam)
        logger.info(self, 'Electrolyte concentration = %.5f mol/L', self.cb)
        logger.info(self, 'Electrolyte type = %d:%d (cation:anion)', self.equiv // 10, self.equiv % 10)
        logger.info(self, 'Temperature = %.5f Kelvin', self.T)
        logger.info(self, 'Potential of zero charge = %.5f V', self.pzc)
        logger.info(self, 'Box length = %.5f Angstrom', self.grids.length * BOHR)
        logger.info(self, 'Total grids = %d', self.grids.get_ngrids())
        logger.info(self, 'Polarization charge density threshold = %4.3e', self.thresh_pol)
        logger.info(self, 'Ion charge density threshold = %4.3e', self.thresh_ions)

    def _get_vind(self, dm):
        if not self._intermediates or self.grids.coords is None:
            self.build()
        if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
            dm = dm[0] + dm[1]

        spacing = self.grids.spacing
        coords = self.grids.coords
        ngrids = self.grids.ngrids
        bias = self.bias / HARTREE2EV # in eV --> a.u.

        phi_sol = self.make_phi_sol(dm, coords)
        self.phi_sol = phi_sol
        rho_sol = self.make_rho_sol(phi_sol, ngrids, spacing)
        self.rho_sol = rho_sol
        phi_tot, rho_ions, rho_pol = self.make_phi(bias, phi_sol, rho_sol)
        self.phi_tot = phi_tot
        self.rho_pol = rho_pol
        self.rho_ions = rho_ions

        phi_pol = phi_tot - phi_sol
        self.phi_pol = phi_pol

        # # Zero out the boundary values to eliminate error
        # ngrids = self.grids.ngrids
        # rho_sol = rho_sol.reshape((ngrids,)*3)
        # idx = numpy.array([-4, -3, -2, -1, 0, 1, 2, 3])
        # rho_sol[idx,:,:] = 0.0e0
        # rho_sol[:,idx,:] = 0.0e0
        # rho_sol[:,:,idx] = 0.0e0
        # rho_sol = rho_sol.flatten()

        # Reaction field contribution
        Gsolv_elst = numpy.dot(rho_sol, phi_pol)*spacing**3

        # Dielectric contribution by Fisicaro
        Gsolv_diel = -0.5e0*(numpy.dot(rho_sol, phi_pol)
                           + numpy.dot(rho_ions, phi_tot))*spacing**3

        # Osmotic pressure contribution
        cb = self.cb * M2HARTREE
        lambda_r = self._intermediates['lambda_r']
        T = self.T
        if self.cb == 0.0e0:
            Gsolv_osm = 0.0e0
        else:
            Gsolv_osm = self.energy_osm(phi_tot, cb, lambda_r, T, spacing)

        logger.info(self, "E_es= %.15g, E_diel= %.15g, E_osm= %.15g", Gsolv_elst, Gsolv_diel, Gsolv_osm)

        Gsolv = Gsolv_elst + Gsolv_diel + Gsolv_osm
        vmat = self._get_vmat(phi_pol)
        return Gsolv, vmat

    def _get_vmat(self, phi_pol):
        logger.info(self, 'Constructing the correction to the Hamiltonian...')
        mol = self.mol
        coords = self.grids.coords
        spacing = self.grids.spacing
        nao = mol.nao
        tot_ngrids = self.grids.get_ngrids()

        vmat = numpy.zeros([nao, nao], order='C')
        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(max(max_memory*.9e6/8/nao, 400))
        for p0, p1 in lib.prange(0, tot_ngrids, blksize):
            ao = mol.eval_gto('GTOval', coords[p0:p1])
            buf = ao * phi_pol[p0:p1, None]
            vmat -= 0.5*numpy.dot(buf.T, ao)
        vmat *= spacing**3
        return vmat
    
    def _get_v(self):
        pass

    def build(self):
        if self.grids.coords is None:
            self.grids.build()
            atom_coords = self.mol.atom_coords()
            coords = self.grids.coords
            box_center = (coords.max(axis=0) + coords.min(axis=0)) / 2.0e0
            bottom_center = box_center.copy()
            bottom_center[2] = self.grids.coords[:,2].min()

            if self.atom_bottom != 'center':
                if self.atom_bottom is None:
                    atom_bottom = numpy.argmin(atom_coords, axis=0)[2]
                else:
                    atom_bottom = self.atom_bottom
                r = atom_coords[atom_bottom]
                atomic_radii = self.get_atomic_radii()
                r_atom_bottom = atomic_radii[atom_bottom]
                shift = r - bottom_center
                coords += shift - numpy.array([0.0e0, 0.0e0, r_atom_bottom])
                self.grids.coords = coords

            if self.custom_shift is not None:
                self.grids.coords -= numpy.asarray(self.custom_shift)

        logger.info(self, 'Grid spacing = %.5f Angstrom', self.grids.spacing * BOHR)

        mol = self.mol
        coords = self.grids.coords
        ngrids = self.grids.ngrids
        spacing = self.grids.spacing
        probe = self.probe / BOHR # angstrom to a.u.
        stern_mol = self.stern_mol / BOHR # angstrom to a.u.
        stern_sam = self.stern_sam / BOHR # angstrom to a.u.
        delta1 = self.delta1 / BOHR # angstrom to a.u.
        delta2 = self.delta2 / BOHR # angstrom to a.u.
        atomic_radii = self.get_atomic_radii()
        eps_sam = self.eps_sam
        eps_bulk = self.eps
        cb = self.cb * M2HARTREE # mol/L to a.u.

        lambda_r = self.make_lambda(mol, probe, stern_mol, stern_sam, coords, delta1, delta2, atomic_radii)
        sas = self.make_sas(mol, probe, coords, delta2, atomic_radii)
        grad_sas = self.make_grad_sas(mol, probe, coords, delta2, atomic_radii)
        lap_sas = self.make_lap_sas(mol, probe, coords, delta2, atomic_radii)
        eps = self.make_eps(coords, eps_sam, eps_bulk, stern_sam, delta1, sas)
        grad_eps = self.make_grad_eps(mol, coords, eps_sam, eps_bulk, probe, stern_sam, delta1, delta2, atomic_radii, sas)

        if self.L is None:
            self.L = ch.poisson((ngrids,)*3, format='csr')

        self.kappa = numpy.sqrt(8.0e0 * PI * cb / self.eps / KB2HARTREE / self.T)
        self._intermediates = {
            'grids': self.grids.coords,
            'lambda_r': lambda_r,
            'eps': eps,
            'grad_eps': grad_eps,
            'sas': sas,
            'grad_sas': grad_sas,
            'lap_sas': lap_sas
        }
        if self.solver == 'fft2d':
            from fcdft.solvent.solver import fft2d
            self.solver = fft2d(ngrids=ngrids, spacing=spacing, verbose=self.verbose, stdout=self.stdout)
        else:
            if self.gpu_accel:
                from fcdft.solvent.solver import multigridGPU
                self.solver = multigridGPU(ngrids=ngrids, spacing=spacing, verbose=self.verbose, stdout=self.stdout)
            else:
                from fcdft.solvent.solver import multigrid
                self.solver = multigrid(ngrids=ngrids, spacing=spacing, verbose=self.verbose, stdout=self.stdout)
        self.solver.build()
    
    def _gen_get_rho_ions(self):
        equiv = self.equiv
        if equiv == 11:
            from fcdft.solvent.ions import _one_to_one
            return _one_to_one
        elif equiv == 21:
            from fcdft.solvent.ions import _two_to_one
            return _two_to_one
        elif equiv == 12:
            from fcdft.solvent.ions import _one_to_two
            return _one_to_two
        else:
            raise NotImplementedError

    def _gen_boundary_conditions(self):
        equiv = self.equiv
        from fcdft.solvent import boundary
        if equiv == 11:
            return boundary.one_to_one_bc, boundary.one_to_one_bc_grad, boundary.one_to_one_bc_lap
        elif equiv == 21:
            return boundary.two_to_one_bc, boundary.two_to_one_bc_grad, boundary.two_to_one_bc_lap
        elif equiv == 12:
            return boundary.one_to_two_bc, boundary.one_to_two_bc_grad, boundary.one_to_two_bc_lap
        else:
            raise NotImplementedError

    def energy_osm(self, phi_tot=None, cb=None, lambda_r=None, T=None, spacing=None, equiv=None):
        if phi_tot is None: phi_tot = self.phi_tot
        if cb is None: cb = self.cb * M2HARTREE
        if lambda_r is None: lambda_r = self._intermediates['lambda_r']
        if T is None: T = self.T
        if spacing is None: spacing = self.grids.spacing
        if equiv is None: equiv = self.equiv
        from fcdft.solvent import ions
        if equiv == 11:
            return ions.one_to_one_energy_osm(self, phi_tot, cb, lambda_r, T, spacing)
        elif equiv == 21:
            return ions.two_to_one_energy_osm(self, phi_tot, cb, lambda_r, T, spacing)
        elif equiv == 12:
            return ions.one_to_two_energy_osm(self, phi_tot, cb, lambda_r, T, spacing)
        else:
            raise NotImplementedError
        
    def __setattr__(self, key, val):
        if key in ('radii_table', 'atom_radii', 'delta1', 'delta2', 'eps', 'stern', 'probe'):
            self.reset()
        super(PBE, self).__setattr__(key, val)


    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
        self._intermediates = None
        return self

    def nuc_grad_method(self, grad_method):
        raise DeprecationWarning('Use the make_grad_object function from '
                                 'pyscf.solvent.grad.ddcosmo_grad or '
                                 'pyscf.solvent._ddcosmo_tdscf_grad instead.')
        
    def grad(self, dm):
        from fcdft.solvent.grad import pbe
        return pbe.kernel(self, dm, self.verbose)

    def to_gpu(self):
        self.gpu_accel = True
        return self
    
    make_lambda = make_lambda
    make_sas = make_sas
    make_grad_sas = make_grad_sas
    make_lap_sas = make_lap_sas
    make_eps = make_eps
    make_grad_eps = make_grad_eps
    make_phi_sol = make_phi_sol
    make_rho_sol = make_rho_sol
    make_phi = make_phi

class Grids(cubegen.Cube):
    def __init__(self, mol, ngrids=97, length=20):
        self.mol = mol
        self.ngrids=ngrids
        self.alignment = 0
        self.length = length / BOHR
        self.spacing = None
        self.coords = None
        self.verbose = mol.verbose
        self.center = None
        super().__init__(mol, nx=ngrids, ny=ngrids, nz=ngrids, margin=self.length/2, extent=[self.length, self.length, self.length])
        
    def get_coords(self):
        atom_coords = self.mol.atom_coords()
        self.center = (atom_coords.max(axis=0) + atom_coords.min(axis=0)) / 2.0e0
        xs, ys, zs = self.xs, self.ys, self.zs
        frac_coords = lib.cartesian_prod([xs, ys, zs])
        box_center = self.box.sum(axis=1) / 2.0e0
        return frac_coords @ self.box + (self.center - box_center)

    def dump_flags(self, verbose=None):
        logger.info(self, 'Grid spacing = %.5f Angstrom', self.grids.spacing * BOHR)

    def build(self, mol=None, *args, **kwargs):
        if mol is None: mol = self.mol
        self.coords = self.get_coords()
        self.boxorig = self.coords[0]
        self.spacing = self.length / (self.nx - 1)
        if self.spacing != self.length / (self.ny - 1) or self.spacing != self.length / (self.nz - 1):
            raise ValueError('Mismatch in ngrids. Current nx, ny, nz = %d, %d, %d' % (self.nx, self.ny, self.nz))
        return self

    def reset(self, mol=None):
        self.coords = None
        self.atom_coords = None
        self.center = None
        return self

if __name__=='__main__':
    from pyscf import gto
    from pyscf.dft import RKS
    mol = gto.M(
        atom='''
C       -1.1367537947      0.1104289172      2.4844663896
C       -1.1385831318      0.1723328088      3.8772156394
C        0.0819843127      0.0788096973      1.7730802291
H       -2.0846565855      0.1966185690      4.4236084687
C        0.0806058727      0.2041086872      4.5921211233
C        1.2993389981      0.1104289172      2.4844663896
H        2.2526138470      0.0865980845      1.9483127672
C        1.2994126658      0.1723829840      3.8783367991
H        2.2453411518      0.1966879024      4.4251589385
H       -2.0869454458      0.0863720324      1.9432143952
C        0.0810980584      0.2676328718      6.0213144069
N        0.0819851974      0.3199013851      7.1972568519
S        0.0000000000      0.0000000000      0.0000000000
H        1.3390319419     -0.0095801980     -0.2157234144''',
        charge=0, basis='6-31g**', verbose=5)
    mf = RKS(mol, xc='pbe')
    from fcdft.wbl.rks import *
    wblmf = WBLMoleculeRKS(mol, xc='pbe', broad=0.01, smear=0.2, nelectron=70.00, ref_pot=5.51)
    # wblmf.pot_cycle=100
    # wblmf.pot_damp=0.7
    # wblmf.conv_tol=1e-7
    wblmf.max_cycle=1
    wblmf.kernel()
    cm = PBE(mol, cb=1.0, length=20, ngrids=41, stern_sam=8.1, equiv=11)
    cm.atom_bottom=12
    cm.solver = 'multigrid'
    solmf = pbe_for_scf(wblmf, cm)
    solmf.kernel()