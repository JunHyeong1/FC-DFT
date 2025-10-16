import numpy
import scipy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import rhf
from pyscf.data.nist import *
from fcdft.wbl.rks import WBLMoleculeRKS
import fcdft
import os
import ctypes
from pyscf.scf.hf import SCF

libfcdft = lib.load_library(os.path.join(fcdft.__path__[0], 'lib', 'libfcdft'))

def scf_for_equil(mf, atom_slice=0, cover=0.0, **kwargs):
    eqmf = Equilibrium(mf, atom_slice=atom_slice, cover=cover, **kwargs)
    name = Equilibrium.__name__ + mf.__class__.__name__
    return lib.set_class(eqmf, (Equilibrium, mf.__class__), name)

def decompose_block_diagonal(mf, mat):
    """
    Decomposition of a block-diagonal matrix
    """
    _nao = mf._nao
    return mat[:_nao,:_nao], mat[_nao:,_nao:]

def decompose_block_column(mf, col):
    _nao = mf._nao
    return col[:_nao], col[_nao:]

def get_veff(mf, mol=None, dm=None, dm_last=0, vhf_last=0, *args, **kwargs):
    mol1, mol2 = mf.mol1, mf.mol2
    nao1, nao2 = mol1.nao, mol2.nao
    cover = mf.cover

    if mf.fermi is None:
        for _ in range(mf.inner_cycle):
            h1e = mf.get_hcore()
            vhf = mf._get_veff(dm=dm)
            s1e = mf.get_ovlp()
            fock = h1e + vhf
            e, c = _eig(mf, fock, s1e)
            idx1 = mol1.nelectron // 2
            idx2 = mol2.nelectron // 2
            mo_occ1, mo_occ2 = numpy.zeros(nao1), numpy.zeros(nao2)
            mo_occ1[:idx1], mo_occ2[:idx2] = 2, 2
            mo_occ = numpy.concatenate((mo_occ1, mo_occ2))
            dm = rhf.make_rdm1(c, mo_occ)
        e1, e2 = decompose_block_column(mf, e)
        fermi = ((1 - cover)*(e1[idx1-1] + e1[idx1]) + cover*(e2[idx2-1] + e2[idx2])) / 2.0e0
        mf.fermi = fermi * HARTREE2EV

    sigmaR = mf.get_sigmaR()
    _vhf = mf._get_veff(mol, dm, dm_last, vhf_last, *args, **kwargs)
    vhf = lib.tag_array(_vhf.real+sigmaR, ecoul=_vhf.ecoul, exc=_vhf.exc, vj=_vhf.vj.real, vk=_vhf.vk,
                        ecoul1=_vhf.ecoul1, exc1=_vhf.exc1,
                        ecoul2=_vhf.ecoul2, exc2=_vhf.exc2)
    if vhf.vk is not None:
        vhf.vk = vhf.vk.real
    return vhf

def get_occ(mf, mo_energy=None, mo_coeff=None):
    if mo_energy is None: mo_energy = mf.mo_energy
    fermi = mf.fermi / HARTREE2EV # Unit in a.u.
    broad = mf.broad / HARTREE2EV # Unit in a.u.
    nmo = mo_energy.size
    moe1, moe2 = decompose_block_column(mf, mo_energy)
    nelectron = mf.nelectron
    nelec = nelectron / 2.0e0
    cover = mf.cover
    nelectron1 = mf.nelectron1
    pot_cycle = mf.pot_cycle
    nelec_a1 = nelectron1 / 2.0e0
    max_cycle = mf.elec_max_cycle
    elec_damp = mf.elec_damp
    logger.info(mf, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
    logger.info(mf, ' | Constrained Search for Equilibrium Chemical Potential |')
    logger.info(mf, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
    for cycle in range(max_cycle):
        nelec_a1_old = nelec_a1
        fermi, mo_occ1 = mf.get_fermi_level(nelec_a1, pot_cycle, broad, moe1, fermi, mf._verbose) # Unit in a.u. verbose=0
        mf.fermi = fermi * HARTREE2EV # Unit in eV.
        if cover == 0:
            mo_occ2 = numpy.zeros(moe2.shape)
        else:
            mo_occ2 = get_electron_number(mf, fermi, broad, moe2)

        nelec1, nelec2 = mo_occ1.sum(), mo_occ2.sum()
        nelec_a12 = (1.0e0 - cover) * nelec1 + cover * nelec2
        dN = abs(nelec - nelec_a12)
        logger.info(mf, 'Macrocycle %2d nelec1, nelec2, dN = %12.10f %12.10f %12.10f',
                    cycle, 2*nelec1, 2*nelec2, 2*dN)
        
        if 2*dN < 5e-11: break

        occ_grad1 = get_occ_grad(mf, fermi, broad, moe1)
        occ_grad2 = get_occ_grad(mf, fermi, broad, moe2)
        grad = (1.0e0 - cover) + cover * occ_grad2.sum() / occ_grad1.sum()
        nelec_a1 = nelec_a1_old + (nelec - nelec_a12) / grad
        nelec_a1 = elec_damp*nelec_a1_old + (1.0e0 - elec_damp)*nelec_a1

        if cycle == max_cycle-1:
            raise RuntimeError

    mf.nelectron1, mf.nelectron2 = 2*nelec1, 2*nelec2
    mf.fermi = fermi * HARTREE2EV
    mo_occ = numpy.concatenate((2*mo_occ1, 2*mo_occ2))

    logger.info(mf, 'mo_occ1 = \n%s', 2*mo_occ1)
    logger.info(mf, 'mo_occ2 = \n%s', 2*mo_occ2)
    if mf.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmo)
        logger.debug(mf, '  mo_energy1 =\n%s', moe1)
        logger.debug(mf, '  mo_energy2 =\n%s', moe2)
        numpy.set_printoptions(threshold=1000)
    logger.info(mf, 'chemical potential = %.15g eV', fermi * HARTREE2EV)
    return mo_occ

def get_occ_grad(mf, fermi, broad=None, mo_energy=None):
    moe_energy = numpy.asarray(mo_energy.real, order='C')
    nbas = moe_energy.shape[0]
    smear = mf.smear / HARTREE2EV
    window = mf.window * broad
    abscissas, weights = mf.abscissas, mf.weights
    quad_order = mf.quad_order

    drv = libfcdft.occupation_grad_drv
    c_moe_energy = moe_energy.ctypes.data_as(ctypes.c_void_p)
    c_abscissas = abscissas.ctypes.data_as(ctypes.c_void_p)
    c_weights = weights.ctypes.data_as(ctypes.c_void_p)
    c_quad_order = ctypes.c_int(quad_order)
    c_window = ctypes.c_double(window)
    c_fermi = ctypes.c_double(fermi)
    c_broad = ctypes.c_double(broad)
    c_smear = ctypes.c_double(smear)
    c_nbas = ctypes.c_int(nbas)
    occ_grad = numpy.empty(nbas, order='C')

    drv(c_moe_energy, c_abscissas, c_weights, c_fermi,
        c_broad, c_smear, c_window, c_quad_order, c_nbas,
        occ_grad.ctypes.data_as(ctypes.c_void_p))

    return occ_grad

def get_electron_number(mf, fermi, broad=None, mo_energy=None):
    moe_energy = numpy.asarray(mo_energy.real, order='C')
    nbas = moe_energy.shape[0]
    smear = mf.smear / HARTREE2EV
    window = mf.window * broad
    abscissas, weights = mf.abscissas, mf.weights
    quad_order = mf.quad_order

    mo_occ = numpy.empty(nbas, order='C')
    drv = libfcdft.occupation_drv
    c_moe_energy = moe_energy.ctypes.data_as(ctypes.c_void_p)
    c_abscissas = abscissas.ctypes.data_as(ctypes.c_void_p)
    c_weights = weights.ctypes.data_as(ctypes.c_void_p)
    c_quad_order = ctypes.c_int(quad_order)
    c_window = ctypes.c_double(window)
    c_fermi = ctypes.c_double(fermi)
    c_broad = ctypes.c_double(broad)
    c_smear = ctypes.c_double(smear)
    c_nbas = ctypes.c_int(nbas)

    drv(c_moe_energy, c_abscissas, c_weights, c_fermi,
        c_broad, c_smear, c_window, c_quad_order, c_nbas,
        mo_occ.ctypes.data_as(ctypes.c_void_p))

    return mo_occ

def energy_elec(mf, dm=None, h1e=None, vhf=None):
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.mol, dm)

    cover = mf.cover
    dm = list(decompose_block_diagonal(mf, dm))
    h1e = list(decompose_block_diagonal(mf, h1e))
    _e1 = []
    for i in range(2):
        _e1.append(numpy.einsum('ij,ji->', h1e[i], dm[i]).real)
    e1 = (1 - cover) * _e1[0] + cover * _e1[1]
    ecoul = vhf.ecoul.real
    exc = vhf.exc.real
    e2 = ecoul + exc
    _e2 = [(vhf.ecoul1+vhf.exc1).real, (vhf.ecoul2+vhf.exc2).real]

    mf.scf_summary['e1'] = e1
    mf.scf_summary['coul'] = ecoul
    mf.scf_summary['exc'] = exc
    logger.debug(mf, 'mol1 E1 = %s  Ecoul = %s  Exc = %s', _e1[0], vhf.ecoul1, vhf.exc1)
    logger.debug(mf, 'mol2 E1 = %s  Ecoul = %s  Exc = %s', _e1[1], vhf.ecoul2, vhf.exc2)
    logger.debug(mf, 'E1 = %s  Ecoul = %s  Exc = %s', e1, ecoul, exc)
    return e1+e2, e2, _e1[0]+_e2[0], _e1[1]+_e2[1]

def energy_tot(mf, dm=None, h1e=None, vhf=None):
    cover = mf.cover
    nuc1, nuc2 = mf.energy_nuc()
    nuc = (1 - cover) * nuc1 + cover * nuc2
    mf.scf_summary['nuc'] = nuc.real

    # e_tot = mf.energy_elec(dm, h1e, vhf)[0] + nuc
    e_ele, _, e_ele1, e_ele2 = mf.energy_elec(dm, h1e, vhf)
    e_tot = e_ele + nuc
    e_tot1 = e_ele1 + nuc1
    e_tot2 = e_ele2 + nuc2
    logger.info(mf, 'E(mol1)= %.15g', e_tot1)
    logger.info(mf, 'E(mol2)= %.15g', e_tot2)
    
    if mf.do_disp():
        raise NotImplementedError
    
    return e_tot

def _eig(mf, fock, s1e):
    """Eigensolver for Hermitian block-diagonal matrix

    Args:
        mf (_type_): _description_
        fock (_type_): _description_
        s1e (_type_): _description_
    """
    _nao = mf._nao
    fock1, fock2 = fock[:_nao,:_nao], fock[_nao:,_nao:]
    s1e1, s1e2 = s1e[:_nao,:_nao], s1e[_nao:,_nao:]
    e1, c1 = scipy.linalg.eigh(fock1, s1e1)
    e2, c2 = scipy.linalg.eigh(fock2, s1e2)
    idx = numpy.argmax(abs(c1.real), axis=0)
    c1[:,c1[idx,numpy.arange(len(e1))].real<0] *= -1
    idx = numpy.argmax(abs(c2.real), axis=0)
    c2[:,c2[idx,numpy.arange(len(e2))].real<0] *= -1
    e = numpy.concatenate((e1, e2))
    c = scipy.linalg.block_diag(c1, c2)
    return e, c

class _Equilibrium:
    pass

class Equilibrium(_Equilibrium):
    _keys = {'atom_slice', 'cover', 'mol1', 'mol2', 'nelectron1', 'nelectron2', '_nao',
            'grids1', 'grids2', 'elec_damp', 'elec_max_cycle'}
    def __init__(self, mf, atom_slice=0, cover=0.0, **kwargs):
        # WBLMoleculeRKS.__init__(self, mol, **kwargs)
        self.__dict__.update(mf.__dict__)
        self.atom_slice = atom_slice # Index of the second molecule
        self.cover = cover # Surface coverage
        self.elec_damp = 0.5e0
        self.elec_max_cycle=100
        self._verbose = 0 # verbose level for get_fermi_level
        # Keywords below are not input
        self.mol1 = None
        self.mol2 = None
        self.nelectron1 = None
        self.nelectron2 = None
        self._nao = None
        self.grids1 = None
        self.grids2 = None

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.mol1 is None or self.mol2 is None:
            from fcdft.gto.mole import split_mol
            self.mol1, self.mol2 = split_mol(mol, self.atom_slice)
        super().build(mol)
        self._nao = self.mol1.nao_nr()
        from pyscf.dft import gen_grid
        self.grids1 = gen_grid.Grids(self.mol1)
        self.grids2 = gen_grid.Grids(self.mol2)
        if self.nelectron1 is None:
            self.nelectron1 = self.mol1.nelectron
        cover = self.cover
        if cover > 1 or cover < 0:
            raise RuntimeError
        if cover == 0:
            self.nelectron2 = 0.0
        else:
            self.nelectron2 = (self.nelectron - (1 - cover) * self.nelectron1) / cover
        return self
    
    def get_hcore(self, mol=None):
        mol1 = self.mol1
        mol2 = self.mol2
        hcore1 = super().get_hcore(mol1)
        hcore2 = super().get_hcore(mol2)
        return scipy.linalg.block_diag(hcore1, hcore2)

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        mol1 = self.mol1
        mol2 = self.mol2
        s1 = super().get_ovlp(mol1)
        s2 = super().get_ovlp(mol2)
        return scipy.linalg.block_diag(s1, s2)
    
    def eig(self, h, s):
        h1, h2 = decompose_block_diagonal(self, h)
        s1, s2 = decompose_block_diagonal(self, s)
        e1, c1 = super().eig(h1, s1)
        e2, c2 = super().eig(h2, s2)
        return numpy.concatenate((e1, e2)), scipy.linalg.block_diag(c1, c2)
    
    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        c1, c2 = decompose_block_diagonal(self, mo_coeff)
        mo_occ1, mo_occ2 = decompose_block_column(self, mo_occ)
        dm1 = super().make_rdm1(c1, mo_occ1)
        dm2 = super().make_rdm1(c2, mo_occ2)
        dm = scipy.linalg.block_diag(dm1, dm2)
        return lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    
    def energy_nuc(self):
        return self.mol1.enuc, self.mol2.enuc
    
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        # Equilibrium object does not store eri on memory.
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        vj, vk = SCF.get_jk(self, mol, dm, hermi, with_j, with_k, omega)
        return vj, vk

    def _get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, *args, **kwargs):
        if dm is None: dm = self.make_rdm1()
        cover = self.cover
        _mol = [self.mol1, self.mol2]
        _grids = [self.grids1, self.grids2]
        _dm = list(decompose_block_diagonal(self, dm))

        if hasattr(dm, 'init_slice'):
            init_slice = dm.init_slice
            mo_coeff = dm.mo_coeff
            mo_coeff1 = mo_coeff[:self._nao,:init_slice[0]]
            mo_coeff2 = mo_coeff[self._nao:,init_slice[0]:]
            _mo_coeff = [mo_coeff1, mo_coeff2]
            mo_occ1, mo_occ2 = dm.mo_occ[:init_slice[0]], dm.mo_occ[init_slice[0]:]
            _mo_occ = [mo_occ1, mo_occ2]
        else:
            _mo_coeff = list(decompose_block_diagonal(self, dm.mo_coeff))
            _mo_occ = list(decompose_block_column(self, dm.mo_occ))

        ni = self._numint
        omega, _, _ = ni.rsh_and_hybrid_coeff(self.xc)
        if omega == 0: omega = None
        if self.direct_scf and self._opt.get(omega) is None:
            vhf = []
            for i in range(2):
                with _mol[i].with_range_coulomb(omega):
                    vhf.append(self.init_direct_scf(_mol[i]))
            self._opt[omega] = vhf

        for i in range(2):
            _dm[i] = lib.tag_array(_dm[i], mo_coeff=_mo_coeff[i], mo_occ=_mo_occ[i])
        if isinstance(dm_last, numpy.ndarray):
            _dm_last = list(decompose_block_diagonal(self, dm_last))
        else: _dm_last = [0, 0]
        if isinstance(vhf_last, numpy.ndarray):
            _vhf_last = list(decompose_block_diagonal(self, vhf_last))
        else: _vhf_last = [0, 0]
        _vxc = []
        # Save the keys
        mol = self.mol
        grids = self.grids
        _opt = self._opt

        for i in range(2):
            self.mol, self.grids, self._opt = _mol[i], _grids[i], {omega: _opt[omega][i]}
            _vxc.append(super()._get_veff(_mol[i], _dm[i], _dm_last[i], _vhf_last[i], *args, **kwargs))

        self.mol = mol
        self.grids = grids
        self._opt = _opt

        ecoul = (1 - cover) * _vxc[0].ecoul + cover * _vxc[1].ecoul
        exc = (1 - cover) * _vxc[0].exc + cover * _vxc[1].exc
        vxc = scipy.linalg.block_diag(_vxc[0], _vxc[1])
        vj = scipy.linalg.block_diag(_vxc[0].vj, _vxc[1].vj)
        vk = scipy.linalg.block_diag(_vxc[0].vk, _vxc[1].vk)
        vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk,
                            ecoul1=_vxc[0].ecoul, exc1=_vxc[0].exc,
                            ecoul2=_vxc[1].ecoul, exc2=_vxc[1].exc)
        return vxc
    
    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'Atom index for slicing molecule = %d', self.atom_slice)
        logger.info(self, 'Surface coverage = %.5f', self.cover)
        return self

    def check_sanity(self):
        s1e = self.get_ovlp()
        s1e1, s1e2 = decompose_block_diagonal(self, s1e)
        cond = lib.cond([s1e1, s1e2])
        logger.debug(self, 'cond(S) = %s', cond)
        if numpy.max(cond)*1e-17 > self.conv_tol:
            logger.warn(self, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                        'SCF may be inaccurate and hard to converge.', numpy.max(cond))
        return super().check_sanity()

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        mol1 = self.mol1
        mol2 = self.mol2
        dm1 = super().get_init_guess(mol=mol1, key=key, **kwargs)
        dm2 = super().get_init_guess(mol=mol2, key=key, **kwargs)
        dm = scipy.linalg.block_diag(dm1, dm2)
        mo_coeff = scipy.linalg.block_diag(dm1.mo_coeff, dm2.mo_coeff)
        mo_occ = numpy.concatenate((dm1.mo_occ, dm2.mo_occ))
        return lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ,
                             init_slice=numpy.array([dm1.mo_coeff.shape[1], dm2.mo_coeff.shape[1]]))
    
    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self.mol1 = self.mol2 = None
        self.grids1 = self.grids2 = None
        return super().reset(mol)
        
    def _finalize(self):
        if self.converged:
            logger.note(self, 'converged SCF energy = %.15g', self.e_tot)
        else:
            logger.note(self, 'SCF not converged.')
            logger.note(self, 'SCF energy = %.15g', self.e_tot)
        dm = list(decompose_block_diagonal(self, self.make_rdm1(self.mo_coeff, self.mo_occ)))
        s1e = list(decompose_block_diagonal(self, self.get_ovlp()))
        nelec = []
        for i in range(2):
            nelec.append(numpy.trace(numpy.dot(dm[i], s1e[i])).real)
        logger.info(self, 'mol1 number of electrons = %.15g', nelec[0])
        logger.info(self, 'mol2 number of electrons = %.15g', nelec[1])
        cover = self.cover
        logger.info(self, 'average number of electrons = %.15g', (1-cover)*nelec[0] + cover*nelec[1])
        logger.info(self, 'optimized chemical potential = %.15g eV', self.fermi)
        return self

    def nuc_grad_method(self):
        from fcdft.equil.grad import equil as equil_grad
        return equil_grad.Gradients(self)
    
    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        import fcdft.df.df_jk_equil
        return fcdft.df.df_jk_equil.equil_density_fit(self, auxbasis, with_df, only_dfj)

    get_occ = get_occ
    get_veff = get_veff
    energy_elec = energy_elec
    energy_tot = energy_tot

if __name__ == '__main__':
    from pyscf import gto
    mol1 = gto.M(
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
        charge=0, basis='6-31++g**', verbose=5, max_memory=8000)
    mol2 = gto.M(
        atom='''
C       -0.8476079365    0.0026700202   -3.5281035617
C       -2.0218283577   -0.2604683627   -2.8073958219
C        0.3353012600    0.2642432684   -2.8231212050
H       -2.9520951875   -0.4549635668   -3.3338745954
H        1.2611570861    0.4600128780   -3.3568350109
C       -2.0073245682   -0.2604462093   -1.4139657047
C        0.3383704802    0.2619859179   -1.4285927349
H       -2.9349818456   -0.4420746761   -0.8791308006
H        1.2732326220    0.4430128273   -0.9060391242
C       -0.8293545128    0.0002996322   -0.6916543989
C       -0.8192397451   -0.0006201775    0.7937732451
C       -1.5616839183   -0.9482725602    1.5218647171
C       -0.0671411632    0.9462610490    1.5129186979
C       -1.5541603792   -0.9481485004    2.9180008413
C       -0.0558307025    0.9446630867    2.9090260355
C       -0.8001707945   -0.0022152441    3.6187748549
H       -2.1295491966   -1.7060626237    0.9896596517
H        0.4930453218    1.7050509252    0.9740007250
H       -2.1296311282   -1.6947234290    3.4583587491
H        0.5267024257    1.6907508988    3.4424644381
H       -0.7929373074   -0.0027907562    4.7049223527
S       -0.9353266992   -0.0162143726   -5.3112847893
H        0.3533301599    0.3018611373   -5.5386460312''',
charge=0, basis='6-31++g**', verbose=5, max_memory=8000)
    from fcdft.gto.mole import conc_mol
    mol = conc_mol(mol1, mol2)
    from fcdft.wbl.rks import WBLMoleculeRKS
    wblmf = WBLMoleculeRKS(mol, xc='B3LYP', broad=0.08, smear=0.2, nelectron=84.20)
    eqmf = scf_for_equil(wblmf, atom_slice=14, cover=0.5).density_fit()
    eqmf.fermi = -4.0
    eqmf.conv_tol=1e-1
    # eqmf.kernel()
    from pyscf.geomopt.geometric_solver import optimize
    optimize(eqmf)
    
    # mf = mol2.RKS(xc='b3lyp')
    # mf.kernel()