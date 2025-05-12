import numpy
import pyscf.grad.rks as rks_grad
import pyscf.grad.uks as uks_grad
import fcdft.grad.rks as wblrks_grad
from pyscf.grad.rhf import _write
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import numint

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of UHF/UKS gradients

    Args:
        mf_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dm0 = mf_grad._tag_rdm1 (dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-UHF Coulomb repulsion')
    vhf = mf_grad.get_veff(mol, dm0)
    log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    dm0_sf = dm0[0] + dm0[1]
    dme0_sf = dme0[0] + dme0[1]

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3), dtype=numpy.complex128)
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm0_sf)
# s1, vhf are \nabla <i|h|j>, the nuclear gradients = -\nabla
        de[k] += numpy.einsum('sxij,sij->x', vhf[:,:,p0:p1], dm0[:,p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0_sf[p0:p1]) * 2

        de[k] += mf_grad.extra_force(ia, locals())

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        _write(log, mol, de.real, atmlst)
    return de.real

def get_veff(ks_grad, mol=None, dm=None):
    '''
    First order derivative of DFT effective potential matrix (wrt electron coordinates)

    Args:
        ks_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

    ni = mf._numint
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = uks_grad.get_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc_full_response(
                ni, mol, nlcgrids, xc, dm[0]+dm[1],
                max_memory=max_memory, verbose=ks_grad.verbose)
            exc += enlc
            vxc += vnlc
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, vxc = get_vxc(ni, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = wblrks_grad.get_nlc_vxc(
                ni, mol, nlcgrids, xc, dm[0]+dm[1],
                max_memory=max_memory, verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(mol, dm)
        vxc += vj[0] + vj[1]
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        vk *= hyb
        if omega != 0:
            vk += ks_grad.get_k(mol, dm, omega=omega) * (alpha - hyb)
        vxc += vj[0] + vj[1] - vk

    return lib.tag_array(vxc, exc1_grid=exc)    

def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    return numpy.asarray((wblrks_grad.make_rdm1e(mo_energy[0], mo_coeff[0], mo_occ[0]),
                          wblrks_grad.make_rdm1e(mo_energy[1], mo_coeff[1], mo_occ[1])))

def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, numpy.asarray(dms), hermi, False, grids)
    ao_loc = mol.ao_loc_nr()
    vmat = numpy.zeros((2,3,nao,nao), dtype=numpy.complex128)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho_a = make_rho(0, ao[0], mask, xctype)
            rho_b = make_rho(1, ao[0], mask, xctype)
            vxc = ni.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[1]
            wv = weight * vxc[:,0]
            aow = numint._scale_ao(ao[0], wv[0])
            rks_grad._d1_dot_(vmat[0], mol, ao[1:4], aow, mask, ao_loc, True)
            aow = numint._scale_ao(ao[0], wv[1])
            rks_grad._d1_dot_(vmat[1], mol, ao[1:4], aow, mask, ao_loc, True)

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho_a = make_rho(0, ao[:4], mask, xctype)
            rho_b = make_rho(1, ao[:4], mask, xctype)
            vxc = ni.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[1]
            wv = weight * vxc
            wv[:,0] *= .5
            rks_grad._gga_grad_sum_(vmat[0], mol, ao, wv[0], mask, ao_loc)
            rks_grad._gga_grad_sum_(vmat[1], mol, ao, wv[1], mask, ao_loc)

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')

    elif xctype == 'MGGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho_a = make_rho(0, ao[:10], mask, xctype)
            rho_b = make_rho(1, ao[:10], mask, xctype)
            vxc = ni.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[1]
            wv = weight * vxc
            wv[:,0] *= .5
            wv[:,4] *= .5
            rks_grad._gga_grad_sum_(vmat[0], mol, ao, wv[0], mask, ao_loc)
            rks_grad._gga_grad_sum_(vmat[1], mol, ao, wv[1], mask, ao_loc)
            rks_grad._tau_grad_dot_(vmat[0], mol, ao, wv[0,4], mask, ao_loc, True)
            rks_grad._tau_grad_dot_(vmat[1], mol, ao, wv[1,4], mask, ao_loc, True)

    exc = numpy.zeros((mol.natm,3))
    # - sign because nabla_X = -nabla_x
    return exc, -vmat

class Gradients(uks_grad.Gradients, wblrks_grad.GradientsBase):
    def __init__(self, mf):
        uks_grad.Gradients.__init__(self, mf)
        wblrks_grad.GradientsBase.__init__(self, mf)

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_occ is None:    mo_occ = self.base.mo_occ
        if mo_coeff is None:  mo_coeff = self.base.mo_coeff
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

    get_veff = get_veff
    grad_elec = grad_elec