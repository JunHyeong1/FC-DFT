from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import rks
from pyscf.scf import rhf
from pyscf.data.nist import *
from functools import reduce
import numpy
import scipy
import fcdft
import ctypes
import os
from fcdft.dft import numint

libfcdft = lib.load_library(os.path.join(fcdft.__path__[0], 'lib', 'libfcdft'))

def get_veff(mf, mol=None, dm=None, dm_last=0, vhf_last=0, *args, **kwargs):
    """
    Construct the effective potential for the WBLMolecule Hamiltonian.

    """
    if mf.fermi is None:
        for _ in range(mf.inner_cycle):
            h1e = mf.get_hcore()
            vhf = mf._get_veff(dm=dm)
            s1e = mf.get_ovlp()
            fock = h1e + vhf
            e, c = rhf.eig(fock, s1e)
            idx = int(mf.nelectron) // 2
            mo_occ = numpy.zeros_like(e)
            mo_occ[:idx] = 2
            dm = rhf.make_rdm1(c, mo_occ)
        idx = round(mf.nelectron) // 2
        fermi = (e[idx-1] + e[idx]) / 2.0e0
        mf.fermi = fermi * HARTREE2EV # Unit in a.u. -> eV

    if dm is not None: dm = dm
    if dm_last is not None: dm_last = dm_last
    sigmaR= mf.get_sigmaR()
    _vhf = mf._get_veff(mol, dm, dm_last, vhf_last, *args, **kwargs)
    vhf = lib.tag_array(_vhf.real+sigmaR, ecoul=_vhf.ecoul, exc=_vhf.exc, vj=_vhf.vj.real, vk=_vhf.vk)
    if vhf.vk is not None:
        vhf.vk = vhf.vk.real
    return vhf

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None, diis_start_cycle=None, level_shift_factor=None, damp_factor=None, fock_last=None):
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    return rhf.get_fock(mf, h1e, s1e, vhf, dm, cycle, diis, diis_start_cycle, level_shift_factor, damp_factor, fock_last)

def get_sigmaR(mf, s1e=None):
    """WBL-Molecule self-energy + Hamiltonian correction by bias voltage.

    Raises:
        AttributeError: No atom detected to apply the bias voltage.

    Returns:
        2D numpy.ndarray: Self-energy + voltage correction
    """
    if s1e is None:
        s1e = mf.get_ovlp()
    ao_labels = mf.mol.ao_labels()
    mf.bias = -(mf.fermi - mf.ref_pot) # Unit in eV
    logger.info(mf, 'Current bias: %.5f V', mf.bias) # Unit in eV
    idx = [i for i, basis in enumerate(ao_labels) if ('S 3p' in basis) or ('S 3s' in basis)]
    if len(idx) == 0:
        raise AttributeError('No sulfur atom detected.')
    S = numpy.zeros_like(s1e)
    S[idx,idx] = s1e[idx,idx]
    return (mf.bias * S - 0.5e0j * mf.broad * s1e) / HARTREE2EV # Unit in a.u.

def eig(h, s):
    """Eigensolver for nonhermitian Hamiltonian.

    Args:
        h (2D numpy.ndarray): Fock matrix
        s (2D numpy.ndarray): Overlap matrix

    Returns:
        e, c: mo energies and mo coefficients
    """
    e, c = scipy.linalg.eig(h, s)
    idx = numpy.argmax(abs(c.real), axis=0)
    c[:,c[idx,numpy.arange(len(e))].real<0] *= -1

    # Sorting the eigenvectors according to the mo energies
    idx = e.real.argsort()
    e, c = e[idx], c[:, idx]

    # Normalization. Bug in scipy.linalg.eig
    norm = numpy.diag(reduce(numpy.dot, (c.T, s, c)))
    c[:,] = c[:,] / numpy.sqrt(norm[:])

    return e, c

def get_occ(mf, mo_energy=None, mo_coeff=None):
    """Occupation matrix through numerical integration

    Args:
        mo_energy (1D numpy.ndarray, optional): MO energies. Defaults to None.
        mo_coeff (2D numpy.ndarray, optional): MO coefficients. Defaults to None.

    Returns:
        1D numpy.ndarray: MO occupation
    """
    if mo_energy is None: mo_energy = mf.mo_energy
    fermi = mf.fermi / HARTREE2EV # Unit in a.u.
    broad = mf.broad / HARTREE2EV # Unit in a.u.
    nmo = mo_energy.size
    pot_cycle = mf.pot_cycle
    nelec_a = mf.nelectron / 2 # Spin-restricted
    fermi, mo_occ = mf.get_fermi_level(nelec_a, pot_cycle, broad, mo_energy, fermi) # Unit in a.u.
    mo_occ *= 2 # Spin-restricted
    mf.fermi = fermi * HARTREE2EV # Unit in eV.
    
    logger.info(mf, 'mo_occ = \n%s', mo_occ)
    if mf.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmo)
        logger.debug(mf, '  mo_energy =\n%s', mo_energy)
        numpy.set_printoptions(threshold=1000)
    logger.info(mf, 'chemical potential = %.15g eV', fermi * HARTREE2EV)
    return mo_occ

def get_fermi_level(mf, nelec_a, pot_cycle=None, broad=None, mo_energy=None, fermi=None):
    """Calculates the Fermi level by the Newton-Raphson method.

    Args:
        mf (:class:`WBLMolecule`): WBLMolecule object.
        pot_cycle (int, optional): Maximum cycle for the Newton-Raphson method. Defaults to 50.
        broad (float, optional): Broadening parameter. Defaults to None.
        mo_energy (1D numpy.ndarray, optional): MO energies. Defaults to None.
        fermi (float, optional): Guess Fermi level. Defaults to None.

    Raises:
        RuntimeError: Failure of the Gaussian quadrature method.
        RuntimeError: Infinite Fermi level detected.
        RuntimeError: Failed to converge in the maximum cycle.

    Returns:
        float, 1D numpy.ndarray: Fermi level and MO occupancy
    """
    if pot_cycle is None: pot_cycle = mf.pot_cycle
    if broad is None: broad = mf.broad / HARTREE2EV
    if mo_energy is None: mo_energy = mf.mo_energy
    if fermi is None: fermi = mf.fermi / HARTREE2EV

    moe_energy = numpy.asarray(mo_energy.real, order='C')
    nbas = moe_energy.shape[0]
    smear = mf.smear / HARTREE2EV
    window = mf.window * broad
    pot_cycle = mf.pot_cycle
    pot_damp = mf.pot_damp
    abscissas, weights = mf.abscissas, mf.weights
    quad_order = mf.quad_order

    drv = libfcdft.fermi_level_drv
    c_moe_energy = moe_energy.ctypes.data_as(ctypes.c_void_p)
    c_abscissas, c_weights = abscissas.ctypes.data_as(ctypes.c_void_p), weights.ctypes.data_as(ctypes.c_void_p)
    c_broad, c_smear = ctypes.c_double(broad), ctypes.c_double(smear)
    c_windows, c_quad_order = ctypes.c_double(window), ctypes.c_int(quad_order)
    c_nbas = ctypes.c_int(nbas)

    fermi_last = None

    for cycle in range(pot_cycle):
        fermi_last = fermi
        mo_occ = numpy.empty(nbas, order='C')
        mo_grad = ctypes.c_double(0.0)
        drv(c_moe_energy, c_abscissas, c_weights,
            ctypes.c_double(fermi), c_broad, c_smear, c_windows, c_quad_order, c_nbas,
            mo_occ.ctypes.data_as(ctypes.c_void_p), ctypes.byref(mo_grad))
        if numpy.any(mo_occ > 1.0e0):
            raise RuntimeError('Numerical integration failed. Integration window: %s eV' % window*HARTREE2EV)
        nelec_last = mo_occ.sum()
        
        delta = (nelec_a - nelec_last) / mo_grad.value
        if delta > 1.0e0:
            fermi += 10**(numpy.log10(delta) - int(numpy.log10(delta)) - 1)
        elif delta < -1.0e0:
            fermi -= 10**(numpy.log10(-delta) - int(numpy.log10(-delta)) - 1)
        else:
            fermi = fermi_last + delta
        if abs(fermi) == numpy.inf:
            raise RuntimeError('Infinity chemical potential detected. Adjust the damping factor.')
        elif abs(nelec_a - nelec_last) > 5.0e-1:
            fermi = pot_damp*fermi_last + (1.0e0-pot_damp)*fermi
        if isinstance(mf, rks.RKS):
            logger.info(mf, ' cycle=%d fermi, nelectron = %.10g, %.10g', cycle+1, fermi, nelec_last*2)
        else:
            logger.info(mf, ' cycle=%d fermi, nelectron = %.10g, %.10g', cycle+1, fermi, nelec_last)
        if abs(fermi - fermi_last) < 1e-11:
            break
        if cycle == pot_cycle-1:
            raise RuntimeError('Chemical potential failed to converge. Adjust the damping factor.')
    return fermi, mo_occ

def make_rdm1(mo_coeff, mo_occ, **kwargs):
    """Reduced density matrix of non-hermitian Hamiltonian

    Args:
        mo_coeff (numpy.ndarray): MO coefficients
        mo_occ (numpy.ndarray): MO occupations

    Returns:
        numpy.ndarray: Reduced density matrix
    """
    dm = reduce(numpy.dot, (mo_coeff, numpy.diag(mo_occ), mo_coeff.T))
    return lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

def init_guess_by_chkfile(mol, chkfile_name, project=None):
    from pyscf.scf import addons, chkfile
    chk_mol, scf_rec = chkfile.load_scf(chkfile_name)
    if project is None:
        project = not gto.same_basis_set(chk_mol, mol)
 
    # Check whether the two molecules are similar
    im1 = scipy.linalg.eigvalsh(mol.inertia_moment())
    im2 = scipy.linalg.eigvalsh(chk_mol.inertia_moment())
    # im1+1e-7 to avoid 'divide by zero' error
    if abs((im1-im2)/(im1+1e-7)).max() > 0.01:
        logger.warn(mol, "Large deviations found between the input "
                    "molecule and the molecule from chkfile\n"
                    "Initial guess density matrix may have large error.")
 
    if project:
        s = rhf.get_ovlp(mol)
 
    def fproj(mo):
        if project:
            mo = addons.project_mo_nr2nr(chk_mol, mo, mol)
            norm = numpy.einsum('pi,pi->i', mo.conj(), s.dot(mo))
            mo /= numpy.sqrt(norm)
        return mo
 
    mo = scf_rec['mo_coeff']
    mo_occ = scf_rec['mo_occ']
    if getattr(mo[0], 'ndim', None) == 1:  # RHF
        mo_coeff = fproj(mo)
        mo_occa = (mo_occ>1e-8).astype(numpy.complex128)
        mo_occb = mo_occ - mo_occa
        dm = _make_rdm1([mo_coeff,mo_coeff], [mo_occa,mo_occb])
    else:  #UHF
        if getattr(mo[0][0], 'ndim', None) == 2:  # KUHF
            logger.warn(mol, 'k-point UHF results are found.  Density matrix '
                        'at Gamma point is used for the molecular SCF initial guess')
            mo = mo[0]
        dm = _make_rdm1([fproj(mo[0]),fproj(mo[1])], mo_occ)
    return dm[0] + dm[1]

def _make_rdm1(mo_coeff, mo_occ, **kwargs):
    '''One-particle density matrix in AO representation

    Args:
        mo_coeff : tuple of 2D ndarrays
            Orbital coefficients for alpha and beta spins. Each column is one orbital.
        mo_occ : tuple of 1D ndarrays
            Occupancies for alpha and beta spins.
    Returns:
        A list of 2D ndarrays for alpha and beta spins
    '''
    mo_a = mo_coeff[0]
    mo_b = mo_coeff[1]
 
    dm_a = numpy.dot(numpy.dot(mo_a, numpy.diag(mo_occ[0])), mo_a.T)
    dm_b = numpy.dot(numpy.dot(mo_b, numpy.diag(mo_occ[1])), mo_b.T)
 
 # DO NOT make tag_array for DM here because the DM arrays may be modified and
 # passed to functions like get_jk, get_vxc.  These functions may take the tags
 # (mo_coeff, mo_occ) to compute the potential if tags were found in the DM
 # arrays and modifications to DM arrays may be ignored.
    return lib.tag_array((dm_a, dm_b), mo_coeff=mo_coeff, mo_occ=mo_occ)

def get_grad(mo_coeff, mo_occ, fock_ao):
    '''Fractional orbital gradients. Seems that this is meaningless in FC-DFT?
    '''
    occidx = mo_occ > 1e-10
    viridx = ~occidx
    g = reduce(numpy.dot, (mo_coeff[:,viridx].T, fock_ao,
                           mo_coeff[:,occidx]))
    return g.ravel()

class WBLBase:
    _keys = {'broad', 'fermi', 'pot_cycle', 'smear', 'nelectron', 'inner_cycle', 
             'pot_damp', 'bias', 'ref_pot', 'window', 'quad_order', 'abscissas', 'weights'}
    
    def __init__(self, broad=0.0, smear=0.2, inner_cycle=1, ref_pot=5.51, nelectron=None):
        self.broad = broad # Unit in eV
        self.ref_pot = ref_pot # Unit in eV
        self.smear = smear # Unit in eV
        self.nelectron = nelectron
        self.fermi = None # Unit in eV

        # It is not recommended to change the following variables.
        self.pot_cycle = 1000
        self.inner_cycle = inner_cycle
        self.pot_damp = 0.5e0
        self.bias = None
        self.window = 2000e0
        self.quad_order = 20001
        self._numint = numint.NumInt()

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'Level broadening = %.5f eV', self.broad)
        logger.info(self, 'Fermi smearing = %.2f eV', self.smear)
        logger.info(self, 'Target number of electrons = %.2f', self.nelectron)
        logger.info(self, 'Reference potential = %.5f eV', self.ref_pot)
        logger.info(self, 'Integration range = %.5f eV', self.window*self.broad)
        logger.info(self, 'Number of Abscissas = %5d points', self.quad_order)
    
    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)

    def nuc_grad_method(self):
        raise NotImplementedError
    
    def _eig(self, h, s):
        return eig(h, s)
    
    def eig(self, h, s):
        return self._eig(h, s)
    
    def get_fermi_level(self, nelec=None, pot_cycle=None, broad=None, mo_energy=None, fermi=None):
        if pot_cycle is None: pot_cycle = self.pot_cycle
        if broad is None: broad = self.broad / HARTREE2EV
        if mo_energy is None: mo_energy = self.mo_energy
        if fermi is None: fermi = self.fermi / HARTREE2EV
        if nelec is None: nelec = self.nelectron / 2
        return get_fermi_level(self, nelec, pot_cycle, broad, mo_energy, fermi)
    
    def build(self, mol=None):
        super().build(mol)
        quad_order = self.quad_order
        if quad_order == 20001:
            path = os.path.join(fcdft.__path__[0], 'wbl')
            self.abscissas = numpy.load(os.path.join(path, 'abscissas.npy'))
            self.weights = numpy.load(os.path.join(path, 'weights.npy'))
        else:
            drv = libfcdft.roots_legendre
            c_quad_order = ctypes.c_int(quad_order)
            abscissas, weights = numpy.empty(quad_order, order='C'), numpy.empty(quad_order, order='C')
            drv(c_quad_order, abscissas.ctypes.data_as(ctypes.c_void_p), weights.ctypes.data_as(ctypes.c_void_p))
            self.abscissas, self.weights = abscissas, weights
        return self
    
    def init_guess_by_chkfile(self, chkfile=None, project=None):
        if chkfile is None: chkfile = self.chkfile
        return init_guess_by_chkfile(self.mol, chkfile, project=project)
    
class WBLMoleculeRKS(WBLBase, rks.RKS):
    def __init__(self, mol, xc='LDA,VWN', broad=0.0, smear=0.2, inner_cycle=1, ref_pot=5.51, nelectron=None):
        rks.RKS.__init__(self, mol, xc=xc)
        WBLBase.__init__(self, broad, smear, inner_cycle, ref_pot, nelectron)
        self._numint = numint.NumInt()

    def dump_flags(self, verbose=None):
        rks.RKS.dump_flags(self, verbose)
        return WBLBase.dump_flags(self, verbose)
    
    def get_sigmaR(self, s1e=None):
        if s1e is None: s1e = self.get_ovlp()
        return get_sigmaR(self, s1e)

    def nuc_grad_method(self):
        from fcdft.grad import rks as wblrks_grad
        return wblrks_grad.Gradients(self)
    
    def _get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, *args, **kwargs):
        """ A hooker to call get_veff of the parant class."""
        return super().get_veff(mol, dm, dm_last, vhf_last, *args, **kwargs)

    def _finalize(self):
        super()._finalize()
        dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
        s1e = self.get_ovlp()
        nelectron = numpy.trace(numpy.dot(dm, s1e)).real
        logger.info(self, 'number of electrons = %.15g', nelectron)
        logger.info(self, 'optimized chemical potential = %.15g eV', self.fermi) # Unit in eV
        return self
    
    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        import fcdft.df.df_jk
        return fcdft.df.df_jk.density_fit(self, auxbasis, with_df, only_dfj)

    get_veff = get_veff
    get_fock = get_fock
    get_occ = get_occ
    make_rdm1 = lib.module_method(make_rdm1, absences=['mo_coeff', 'mo_occ'])

if __name__ == '__main__':
    from pyscf import gto
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
    wblmf = WBLMoleculeRKS(mol, xc='b3lyp', broad=0.01, smear=0.2, nelectron=70.00)
    wblmf.conv_tol = 1e-8
    wblmf.kernel()