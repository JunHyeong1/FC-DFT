import numpy
import pyscf.grad.rks as rks_grad
from pyscf.grad.rhf import _write
from pyscf.lib import logger
from pyscf.data.nist import HARTREE2EV

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    """
    Compute electronic contribution to WBL nuclear gradients under frozen-potential approximation.

    Parameters
    ----------
    mf_grad : Gradients
        WBL gradient object.
    mo_energy : ndarray, optional
        MO energies (shape n_mo). If None, uses self.base.mo_energy.
    mo_coeff : ndarray, optional
        MO coefficients (shape n_ao x n_mo). If None, uses self.base.mo_coeff.
    mo_occ : ndarray, optional
        MO occupations (shape n_mo). If None, uses self.base.mo_occ.
    atmlst : list, optional
        List of atom indices for which to compute forces. If None, computes all atoms.

    Returns
    -------
    de : ndarray, shape (len(atmlst), 3)
        Real part only.
    """
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dm0 = mf_grad._tag_rdm1 (dm0, mo_coeff, mo_occ)
    broad = mf.broad / HARTREE2EV
    bias = mf.bias / HARTREE2EV

    ao_labels = mol.ao_labels()
    idx = [i for i, basis in enumerate(ao_labels) if ('S 3p' in basis) or ('S 3s' in basis)]
    S = numpy.zeros_like(s1)
    S[:,idx,idx] = s1[:,idx,idx]

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')
    vhf = mf_grad.get_veff(mol, dm0)

    log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()

    de = numpy.zeros((len(atmlst),3), dtype=numpy.complex128)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        h1ao = hcore_deriv(ia)
        # Hellmann-Feynmann force
        # Kohn-Sham Hamiltonian part
        # One-electron contribution
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm0)
        # Coulomb, exchange, and xc potential contribution
        de[k] += numpy.einsum('xij,ij->x', vhf[:,p0:p1], dm0[p0:p1])*2
        # Self-energy contribution
        de[k] -= 0.5j*broad*numpy.einsum('xij,ij->x', s1[:,p0:p1], dm0[p0:p1])*2
        # Voltage contribution
        de[k] += bias*numpy.einsum('xij,ij->x',S[:,p0:p1], dm0[p0:p1])*2
        # Pulay force
        de[k] -= numpy.einsum('xij,ij->x',s1[:,p0:p1], dme0[p0:p1])*2
        # Extra force contribution
        de[k] += mf_grad.extra_force(ia, locals())

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        _write(log, mol, de.real, atmlst)
    return de.real

def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    """
    Construct energy-weighted density matrix.
    """
    mo0e =  mo_coeff * (mo_energy * mo_occ)
    return numpy.dot(mo0e, mo_coeff.T)


class Gradients(rks_grad.Gradients):
    """
    Nuclear gradients for WBL-Molecule RKS.

    Computes nuclear forces (first derivatives of SCF energy with respect to
    nuclear positions) for molecules at electrode surfaces using the WBL
    approximation. Combines PySCF RKS gradient framework with WBL-specific
    terms (self-energy, voltage correction).

    The computed gradients enable geometry optimization and vibrational analysis
    for electrochemical systems.

    Examples
    --------
    >>> from fcdft.wbl.rks import WBLMoleculeRKS
    >>> mol = gto.M(atom='C 0 0 0; S 0 0 1.5', basis='6-31g**')
    >>> wbl = WBLMoleculeRKS(mol, xc='pbe', broad=0.01, ref_pot=-4.5)
    >>> wbl.kernel()
    >>> grad = wbl.nuc_grad_method()
    >>> forces = grad.kernel()  # shape (n_atoms, 3), in Hartree/Bohr
    >>> print(forces)
    """
    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

    grad_elec = grad_elec