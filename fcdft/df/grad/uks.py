import numpy
from fcdft.grad import uks as uks_grad
from fcdft.df.grad import rks as df_rks_grad
from pyscf.grad import uks as _uks_grad
from pyscf.lib import logger
from pyscf.grad import rks as rks_grad
from pyscf import lib

def get_veff(ks_grad, mol=None, dm=None):
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = _uks_grad.get_vxc_full_response(
                ni, mol, grids, mf.xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
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
        exc, vxc = uks_grad.get_vxc(
                ni, mol, grids, mf.xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc(
                ni, mol, nlcgrids, xc, dm[0]+dm[1],
                max_memory=max_memory, verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(mol, dm)
        vxc += vj[0] + vj[1]
        if ks_grad.auxbasis_response:
            e1_aux = vj.aux.sum ((0,1))
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        if ks_grad.auxbasis_response:
            vk.aux = vk.aux * hyb
        vk[:] *= hyb # inplace * for vk[:] to keep the .aux tag
        if omega != 0:  # For range separated Coulomb operator
            vk_lr = ks_grad.get_k(mol, dm, omega=omega)
            vk[:] += vk_lr * (alpha - hyb)
            if ks_grad.auxbasis_response:
                vk.aux[:] += vk_lr.aux * (alpha - hyb)
        vxc += vj[0] + vj[1] - vk
        if ks_grad.auxbasis_response:
            e1_aux = vj.aux.sum ((0,1))
            e1_aux -= numpy.trace (vk.aux, axis1=0, axis2=1)

    if ks_grad.auxbasis_response:
        logger.debug1(ks_grad, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
        vxc = lib.tag_array(vxc, exc1_grid=exc, aux=e1_aux)
    else:
        vxc = lib.tag_array(vxc, exc1_grid=exc)

class Gradients(uks_grad.Gradients):
    def __init__(self, mf):
        uks_grad.Gradients.__init__(self, mf)

    auxbasis_response = True

    get_jk = df_rks_grad.Gradients.get_jk
    get_j = df_rks_grad.Gradients.get_j
    get_k = df_rks_grad.Gradients.get_k
    get_veff = get_veff

Grad = Gradients