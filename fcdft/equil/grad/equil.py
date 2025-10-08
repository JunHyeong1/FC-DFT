import numpy
import scipy
from pyscf import gto
from pyscf.grad.rhf import _write
from pyscf.lib import logger
from fcdft.grad import rks as wblrks_grad
from fcdft.equil.equil import decompose_block_diagonal, decompose_block_column

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    _mol = [mf_grad.base.mol1, mf_grad.base.mol2]
    hcore_deriv = []
    for i in range(2):
        hcore_deriv.append(mf_grad.hcore_generator(_mol[i]))

    ovlp = mf_grad.get_ovlp(mol)
    _nao = mf._nao
    s1 = [ovlp[:,:_nao,:_nao], ovlp[:,_nao:,_nao:]]
    dm0 = list(decompose_block_diagonal(mf, mf.make_rdm1(mo_coeff, mo_occ)))
    _mo_coeff = list(decompose_block_diagonal(mf, mo_coeff))
    _mo_occ = list(decompose_block_column(mf, mo_occ))
    dm0[0] = mf_grad._tag_rdm1 (dm0[0], _mo_coeff[0], _mo_occ[0])
    dm0[1] = mf_grad._tag_rdm1 (dm0[1], _mo_coeff[1], _mo_occ[1])
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')
    vhf = []
    grids =  [mf.grids1, mf.grids2]
    _grids = mf.grids
    for i in range(2):
        mf_grad.mol = _mol[i]
        mf_grad.grids = grids[i]
        vhf.append(mf_grad.get_veff(_mol[i], dm0[i]))
    mf_grad.grids = _grids
    mf_grad.mol = mol
    log.timer('gradients of 2e part', *t0)
    dme0 = list(decompose_block_diagonal(mf, mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)))

    de = []
    for i in range(2):
        atmlst = range(_mol[i].natm)
        aoslices = _mol[i].aoslice_by_atom()
        _de = numpy.zeros((len(atmlst),3), dtype=numpy.complex128)
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices [ia,2:]
            h1ao = hcore_deriv[i](ia)
            _de[k] += numpy.tensordot(h1ao, dm0[i], axes=([1,2], [0,1]))
    # nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
            _de[k] += numpy.tensordot(vhf[i][:,p0:p1], dm0[i][p0:p1], axes=([1,2], [0,1])) * 2
            _de[k] -= numpy.tensordot(s1[i][:,p0:p1], dme0[i][p0:p1], axes=([1,2], [0,1])) * 2

            _de[k] += mf_grad.extra_force(ia, locals()) ### Needs to be fixed
        de.append(_de)
    
    cover = mf.cover
    # E = (1 - c)*E_A + c*E_B
    de = numpy.concatenate(((1 - cover) * de[0], cover * de[1]), axis=0)
    atmlst = range(mol.natm)
    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        _write(log, mol, de.real, atmlst)
    return de.real

def make_rdm1e(mf, mo_energy, mo_coeff, mo_occ):
    mo_energy1, mo_energy2 = decompose_block_column(mf, mo_energy)
    mo_occ1, mo_occ2 = decompose_block_column(mf, mo_occ)
    mo_coeff1, mo_coeff2 = decompose_block_diagonal(mf, mo_coeff)
    dm1e1 = wblrks_grad.make_rdm1e(mo_energy1, mo_coeff1, mo_occ1)
    dm1e2 = wblrks_grad.make_rdm1e(mo_energy2, mo_coeff2, mo_occ2)
    return scipy.linalg.block_diag(dm1e1, dm1e2)


class Gradients(wblrks_grad.Gradients):
    def __init__(self, mf):
        wblrks_grad.Gradients.__init__(self, mf)
    
    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        return make_rdm1e(self.base, mo_energy, mo_coeff, mo_occ)
    
    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        mol1, mol2 = self.base.mol1, self.base.mol2
        s1e1 = super().get_ovlp(mol1)
        s1e2 = super().get_ovlp(mol2)
        xyz = [scipy.linalg.block_diag(a, b) for a, b in zip(s1e1, s1e2)]
        return numpy.stack(xyz, axis=0)
    
    def grad_nuc(self, mol=None, atmlst=None):
        mol1, mol2 = self.base.mol1, self.base.mol2
        cover = self.base.cover
        grad_nuc1 = super().grad_nuc(mol1, atmlst=atmlst) # atmlst <- None
        grad_nuc2 = super().grad_nuc(mol2, atmlst=atmlst) # atmlst <- None
        # E = (1 - c)*E_A + c*E_B
        return numpy.concatenate(((1 - cover) * grad_nuc1, cover * grad_nuc2), axis=0)
    
    def extra_force(self, atom_id, envs):
        if self.grid_response:
            vhf = envs['vhf']
            log = envs['log']
            log.debug('grids response for atom %d %s',
                      atom_id, vhf.exc1_grid[atom_id])
            return vhf.exc1_grid[atom_id]            
        else:
            return 0
    
    grad_elec = grad_elec
    