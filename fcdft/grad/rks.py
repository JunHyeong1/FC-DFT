import numpy
import ctypes
import pyscf.grad.rhf as rhf_grad
import pyscf.grad.rks as rks_grad
from pyscf.grad.rhf import _write
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import numint, xc_deriv
from pyscf.scf import _vhf

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of RHF/RKS gradients

    Args:
        mf_grad : grad.rhf.Gradients or grad.rks.Gradients object
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
    dm0 = mf_grad._tag_rdm1 (dm0, mo_coeff, mo_occ)

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
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm0)
# nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
        de[k] += numpy.einsum('xij,ij->x', vhf[:,p0:p1], dm0[p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2

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

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)

    if ks_grad.grid_response:
        exc, vxc = rks_grad.get_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose) # Needs to be confirmed
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc_full_response(
                ni, mol, nlcgrids, xc, dm,
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
            enlc, vnlc = get_nlc_vxc(
                ni, mol, nlcgrids, xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(mol, dm)
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        vk *= hyb
        if omega != 0:
            vk += ks_grad.get_k(mol, dm, omega=omega) * (alpha - hyb)

        vxc += vj - vk * .5

    return lib.tag_array(vxc, exc1_grid=exc)

def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Complex energy weighted density matrix'''
    mo0e =  mo_coeff * (mo_energy * mo_occ)
    return numpy.dot(mo0e, mo_coeff.T)

def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, numpy.asarray(dms), hermi, False, grids)
    ao_loc = mol.ao_loc_nr()
    vmat = numpy.zeros((nset,3,nao,nao), dtype=numpy.complex128)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[0], mask, xctype)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                wv = weight * vxc[0]
                aow = numint._scale_ao(ao[0], wv)
                rks_grad._d1_dot_(vmat[idm], mol, ao[1:4], aow, mask, ao_loc, True)

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[:4], mask, xctype)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                wv = weight * vxc
                wv[0] *= .5
                rks_grad._gga_grad_sum_(vmat[idm], mol, ao, wv, mask, ao_loc)

    elif xctype == 'MGGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[:10], mask, xctype)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                wv = weight * vxc
                wv[0] *= .5
                wv[4] *= .5  # for the factor 1/2 in tau
                rks_grad._gga_grad_sum_(vmat[idm], mol, ao, wv, mask, ao_loc)
                rks_grad._tau_grad_dot_(vmat[idm], mol, ao, wv[4], mask, ao_loc, True)

    exc = None
    if nset == 1:
        vmat = vmat[0]
    # - sign because nabla_X = -nabla_x
    return exc, -vmat

def get_nlc_vxc(ni, mol, grids, xc_code, dm, relativity=0, hermi=1,
                max_memory=2000, verbose=None):
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi, False, grids)
    assert nset == 1
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((3,nao,nao), dtype=numpy.complex128)
    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]
    ao_deriv = 2
    vvrho = []
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        vvrho.append(make_rho(0, ao[:4], mask, 'GGA'))
    rho = numpy.hstack(vvrho)

    vxc = numint._vv10nlc(rho, grids.coords, rho, grids.weights,
                          grids.coords, nlc_pars)[1]
    vv_vxc = xc_deriv.transform_vxc(rho, vxc, 'GGA', spin=0)

    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        p0, p1 = p1, p1 + weight.size
        wv = vv_vxc[:,p0:p1] * weight
        wv[0] *= .5  # *.5 because vmat + vmat.T at the end
        rks_grad._gga_grad_sum_(vmat, mol, ao, wv, mask, ao_loc)

    exc = None
    # - sign because nabla_X = -nabla_x
    return exc, -vmat

def get_jk(mol, dm):
    '''J = ((-nabla i) j| kl) D_lk
    K = ((-nabla i) j| kl) D_jk
    '''
    libcvhf = _vhf.libcvhf
    vhfopt = _vhf._VHFOpt(mol, 'int2e_ip1', 'CVHFgrad_jk_prescreen',
                          dmcondname='CVHFnr_dm_cond1')
    ao_loc = mol.ao_loc_nr()
    nbas = mol.nbas
    q_cond = numpy.empty((2, nbas, nbas))
    with mol.with_integral_screen(vhfopt.direct_scf_tol**2):
        libcvhf.CVHFnr_int2e_pp_q_cond(
            getattr(libcvhf, mol._add_suffix('int2e_ip1ip2')),
            lib.c_null_ptr(), q_cond[0].ctypes,
            ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)
        libcvhf.CVHFnr_int2e_q_cond(
            getattr(libcvhf, mol._add_suffix('int2e')),
            lib.c_null_ptr(), q_cond[1].ctypes,
            ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)
    vhfopt.q_cond = q_cond

    intor = mol._add_suffix('int2e_ip1')
    vjre, vkre = _vhf.direct_mapdm(intor,  # (nabla i,j|k,l)
                                        's2kl', # ip1_sph has k>=l,
                                        ('lk->s1ij', 'jk->s1il'),
                                        dm.real, 3, # xyz, 3 components
                                        mol._atm, mol._bas, mol._env, vhfopt=vhfopt)
    vjim, vkim = _vhf.direct_mapdm(intor,  # (nabla i,j|k,l)
                                        's2kl', # ip1_sph has k>=l,
                                        ('lk->s1ij', 'jk->s1il'),
                                        dm.imag, 3, # xyz, 3 components
                                        mol._atm, mol._bas, mol._env, vhfopt=vhfopt)
    return -(vjre + vjim*1.0j), -(vkre + vkim*1.0j)

class GradientsBase(rhf_grad.GradientsBase):
    def __init__(self, method):
        rhf_grad.GradientsBase.__init__(self, method)

    def get_jk(self, mol=None, dm=None, hermi=0, omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        cpu0 = (logger.process_clock(), logger.perf_counter())
        if omega is None:
            vj, vk = get_jk(mol, dm)
        else:
            with mol.with_range_coulomb(omega):
                vj, vk = get_jk(mol, dm)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        intor = mol._add_suffix('int2e_ip1')
        if omega is None:
            vjre = _vhf.direct_mapdm(intor, 's2kl', 'lk->s1ij', dm.real, 3,
                                     mol._atm, mol._bas, mol._env)
            vjim = _vhf.direct_mapdm(intor, 's2kl', 'lk->s1ij', dm.imag, 3,
                                     mol._atm, mol._bas, mol._env)
            return -(vjre + vjim*1.0j)
        with mol.with_range_coulomb(omega):
            vjre = _vhf.direct_mapdm(intor, 's2kl', 'lk->s1ij', dm.real, 3,
                                     mol._atm, mol._bas, mol._env)
            vjim = _vhf.direct_mapdm(intor, 's2kl', 'lk->s1ij', dm.imag, 3,
                                     mol._atm, mol._bas, mol._env)
            return -(vjre + vjim*1.0j)

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):    
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        intor = mol._add_suffix('int2e_ip1')
        if omega is None:
            vkre = _vhf.direct_mapdm(intor, 's2kl', 'jk->s1il', dm.real, 3,
                                     mol._atm, mol._bas, mol._env)
            vkim = _vhf.direct_mapdm(intor, 's2kl', 'jk->s1il', dm.imag, 3,
                                     mol._atm, mol._bas, mol._env)
            return -(vkre + vkim*1.0j)
        with mol.with_range_coulomb(omega):
            vkre = _vhf.direct_mapdm(intor, 's2kl', 'jk->s1il', dm.real, 3,
                                     mol._atm, mol._bas, mol._env)
            vkim = _vhf.direct_mapdm(intor, 's2kl', 'jk->s1il', dm.imag, 3,
                                     mol._atm, mol._bas, mol._env)
            return -(vkre + vkim*1.0j)        

class Gradients(rks_grad.Gradients, GradientsBase):
    def __init__(self, mf):
        rks_grad.Gradients.__init__(self, mf)
        GradientsBase.__init__(self, mf)

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)
    
    get_veff = get_veff
    grad_elec = grad_elec
    