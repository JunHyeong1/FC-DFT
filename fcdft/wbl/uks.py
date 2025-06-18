# Untouched version
import numpy
from functools import reduce
from pyscf.scf import rhf
from pyscf.dft import uks
from pyscf.scf import uhf
from pyscf import lib
from pyscf.lib import logger
from pyscf.data.nist import *
from fcdft.wbl import rks as wblrks 

def get_veff(mf, mol=None, dm=None, dm_last=0, vhf_last=0, *args, **kwargs):
    if mf.fermi is None:
        fermi = numpy.zeros(2, dtype=numpy.float64, order='C')
        nelec_a, nelec_b = mf.nelec
        for _ in range(mf.inner_cycle):
            h1e = mf.get_hcore()
            vhf = mf._get_veff(dm=dm)
            s1e = mf.get_ovlp()
            fock = h1e + vhf
            ea, ca = rhf.eig(fock[0], s1e)
            eb, cb = rhf.eig(fock[1], s1e)
            e, c = numpy.array((ea, eb)), numpy.array((ca, cb))
            mo_occ = numpy.zeros((2, e.shape[1]), dtype=numpy.float64, order='C')
            mo_occ[0,:nelec_a] = 1
            mo_occ[1,:nelec_b] = 1
            dm = uhf.make_rdm1(c, mo_occ)
        for spin, nelec in zip(range(2), mf.nelec):
            fermi[spin] = (e[spin][nelec-1] + e[spin][nelec]) / 2.0e0
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
    return uhf.get_fock(mf, h1e, s1e, vhf, dm, cycle, diis, diis_start_cycle, level_shift_factor, damp_factor, fock_last)

def get_sigmaR(mf, s1e=None):
    if s1e is None:
        s1e = mf.get_ovlp()
    ao_labels = mf.mol.ao_labels()
    # mf.bias = -(mf.fermi - mf.ref_pot) # Unit in eV
    mf.bias = -(mf.fermi.sum()/2 - mf.ref_pot) # Unit in eV
    logger.info(mf, 'Current bias: %.5f V', mf.bias)
    idx = [i for i, basis in enumerate(ao_labels) if ('S 3p' in basis) or ('S 3s' in basis)]
    if len(idx) == 0:
        raise AttributeError('No sulfur atom detected.')
    S = numpy.zeros_like(s1e)
    S[idx,idx] = s1e[idx,idx]
    return (mf.bias * S - 0.5e0j * mf.broad * s1e) / HARTREE2EV # Unit in a.u.

def get_occ(mf, mo_energy=None, mo_coeff=None):
    if mo_energy is None: mo_energy = mf.mo_energy
    fermi = mf.fermi / HARTREE2EV # Unit in a.u.
    broad = mf.broad / HARTREE2EV # Unit in a.u.
    nmo = mo_energy.size
    pot_cycle = mf.pot_cycle
    nelec_a = (mf.nelectron + mf.mol.spin) / 2
    nelec_b = mf.nelectron - nelec_a
    fermi_a, mo_occ_a = mf.get_fermi_level(nelec_a, pot_cycle, broad, mo_energy[0], fermi[0]) # Unit in a.u.
    fermi_b, mo_occ_b = mf.get_fermi_level(nelec_b, pot_cycle, broad, mo_energy[1], fermi[1]) # Unit in a.u.
    mf.fermi = numpy.array((fermi_a, fermi_b)) * HARTREE2EV
    mo_occ = numpy.array((mo_occ_a, mo_occ_b))
    mf.mo_occ = mo_occ
    
    logger.info(mf, 'mo_occ (alpha)= \n%s', mo_occ_a)
    logger.info(mf, 'mo_occ  (beta)= \n%s', mo_occ_b)
    if mf.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmo)
        logger.debug(mf, '  alpha mo_energy =\n%s', mo_energy[0])
        logger.debug(mf, '  beta  mo_energy =\n%s', mo_energy[1])
        numpy.set_printoptions(threshold=1000)
    logger.info(mf, 'chemical potential (alpha, beta)= %.15g, %.15g eV', fermi[0], fermi[1])
    if mo_coeff is not None and mf.verbose >= logger.DEBUG:
        ss, s = mf.spin_square(mo_coeff, mf.get_ovlp())
        logger.debug(mf, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
    return mo_occ

def make_rdm1(mo_coeff, mo_occ, **kwargs):
    mo_a = mo_coeff[0]
    mo_b = mo_coeff[1]
    dm_a = reduce(numpy.dot, (mo_a, numpy.diag(mo_occ[0]), mo_a.T))
    dm_b = reduce(numpy.dot, (mo_b, numpy.diag(mo_occ[1]), mo_b.T))
    return lib.tag_array((dm_a, dm_b), mo_coeff=mo_coeff, mo_occ=mo_occ)

def spin_square(dm, s1e=1):
    mo_a, mo_b = dm.mo_coeff
    mo_occa, mo_occb = dm.mo_occ
    nocc_a = mo_occa.sum()
    nocc_b = mo_occb.sum()
    s = reduce(numpy.dot, (numpy.diag(mo_occa), mo_a.T.conj(), s1e, mo_b, numpy.diag(mo_occb)))
    ssxy = (nocc_a + nocc_b)*.5 - numpy.einsum('ij,ij->', s.conj(), s).real
    ssz = (nocc_b - nocc_a)**2 * .25
    ss = (ssxy + ssz).real
    s = numpy.sqrt(ss+.25) - .5
    return ss, s*2+1

def get_grad(mo_coeff, mo_occ, fock_ao):
    '''Fractional orbital gradients. Seems that this is meaningless for our purpose?
    '''
    occidxa = mo_occ[0] > 1e-10
    occidxb = mo_occ[1] > 1e-10
    viridxa = ~occidxa
    viridxb = ~occidxb
    ga = reduce(numpy.dot, (mo_coeff[0][:,viridxa].T, fock_ao[0], mo_coeff[0][:,occidxa]))
    gb = reduce(numpy.dot, (mo_coeff[1][:,viridxb].T, fock_ao[1], mo_coeff[1][:,occidxb]))
    return numpy.hstack((ga.ravel(), gb.ravel()))

class WBLMoleculeUKS(wblrks.WBLBase, uks.UKS):
    def __init__(self, mol, xc='LDA,VWN', broad=0.0, smear=0.2, inner_cycle=1, ref_pot=5.51, nelectron=None):
        uks.UKS.__init__(self, mol, xc=xc)
        wblrks.WBLBase.__init__(self, broad, smear, inner_cycle, ref_pot, nelectron)

    def get_sigmaR(self, s1e=None):
        if s1e is None: s1e = self.get_ovlp()
        return get_sigmaR(self, s1e)

    def nuc_grad_method(self):
        from fcdft.grad import uks as wbluks_grad
        return wbluks_grad.Gradients(self)
    
    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)
    
    def _get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, *args, **kwargs):
        """ A hooker to call get_veff of the parant class."""
        return super().get_veff(mol, dm, dm_last, vhf_last, *args, **kwargs)

    def _finalize(self):
        super()._finalize()
        dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
        s1e = self.get_ovlp()
        nelec_a = numpy.trace(numpy.dot(dm[0], s1e)).real
        nelec_b = numpy.trace(numpy.dot(dm[1], s1e)).real
        logger.info(self, 'number of electrons (alpha, beta)= %.15g, %.15g', nelec_a, nelec_b)
        logger.info(self, 'number of electrons (alpha+beta) = %.15g', nelec_a + nelec_b)
        logger.info(self, 'optimized chemical potential (alpha, beta)= %.15g, %.15g eV', self.fermi[0], self.fermi[1])
        return self

    def eig(self, fock, s):
        e_a, c_a = self._eig(fock[0], s)
        e_b, c_b = self._eig(fock[1], s)
        return numpy.array((e_a, e_b)), numpy.array((c_a, c_b))
    
    def spin_square(self, mo_coeff=None, s1e=None):
        if s1e is None:
            s1e = self.get_ovlp()
        dm = self.make_rdm1(mo_coeff, self.mo_occ)
        return spin_square(dm, s1e)
    
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
C        0.000000000      0.000000000      1.820000000
C       -1.209256000      0.000000000      2.518164000
C        1.209256000      0.000000000      2.518164000
H       -2.150061000      0.000000000      1.974991000
H        2.150061000      0.000000000      1.974991000
C       -1.209256000      0.000000000      3.914494000
C        1.209256000      0.000000000      3.914494000
H       -2.150061000      0.000000000      4.457667000
H        2.150061000      0.000000000      4.457667000
C        0.000000000      0.000000000      4.612658000
C        0.000000000      0.000000000      6.152658000
N        0.000000000      0.000000000      7.315997000
S        0.000000000      0.000000000      0.000000000''',
        charge=-1, basis='6-31g**', verbose=5, spin=2)
    wblmf = WBLMoleculeUKS(mol, xc='pbe', broad=0.01, smear=0.2, nelectron=69.95)
    wblmf.conv_tol=1e-8
    wblmf.kernel()