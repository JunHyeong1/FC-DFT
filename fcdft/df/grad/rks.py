import numpy
import ctypes
from pyscf import df
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from fcdft.grad import rks as rks_grad
from pyscf.df.grad import rhf as df_rhf_grad
from pyscf.df.grad.rks import get_veff
from pyscf.df.grad.rhf import (LINEAR_DEP_THRESHOLD, _gen_metric_solver,
                               _int3c_wrapper, balance_partition)

def get_jk(mf_grad, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
           decompose_j2c='CD', lindep=LINEAR_DEP_THRESHOLD):

    assert (with_j or with_k)
    if not with_k:
        return df_rhf_grad.get_jk(mf_grad, mol=mol, dm=dm, hermi=hermi,
                                  with_j=with_j, with_k=False)

    if hasattr(mf_grad.base, 'only_dfj') and mf_grad.base.only_dfj:
        return df_rhf_grad.get_jk(mf_grad, mol=mol, dm=dm, hermi=hermi,
                                  with_j=with_j, with_k=with_k)

    t0 = (logger.process_clock(), logger.perf_counter())
    if mol is None: mol = mf_grad.mol
    if dm is None: dm = mf_grad.base.make_rdm1()
    with_df = mf_grad.base.with_df
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
    nbas, nao, naux = mol.nbas, mol.nao, auxmol.nao
    nao_pair = nao * (nao + 1) // 2
    aux_loc = auxmol.ao_loc
    aoslice = mol.aoslice_by_atom()
    auxslice = auxmol.aoslice_by_atom()

    # Density matrix preprocessing
    dms = numpy.asarray(dm)
    out_shape = dms.shape[:-2] + (3,) + dms.shape[-2:]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    # For j
    idx = numpy.arange(nao)
    idx = idx * (idx+1) // 2 + idx
    dm_tril = dms + dms.transpose(0,2,1)
    dm_tril = lib.pack_tril(dm_tril)
    dm_tril[:,idx] *= .5

    memory_need = naux*nao_pair*8/1e6
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    if with_k and (memory_need > 0.5*max_memory):
        logger.info(mf_grad, 'Turning on the defaulk get_jk')
        return df_rhf_grad.get_jk(mf_grad, mol, dm, hermi, with_j, with_k,
                                  decompose_j2c, lindep)

    t1 = (logger.process_clock(), logger.perf_counter())
    # Prepare RI-J
    int2c = auxmol.intor('int2c2e', aosym='s1')
    solve_j2c = _gen_metric_solver(int2c, decompose_j2c, lindep)
    int2c = None

    get_int3c_s2 = _int3c_wrapper(mol, auxmol, intor='int3c2e', aosym='s2ij')
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = int(min(max(max_memory*.5e6/8 / nao_pair, 20), naux))

    rhoj = numpy.zeros((nset,naux))
    j3c = numpy.empty((naux,nao_pair))

    for shl0, shl1, nL in balance_partition(aux_loc, blksize):
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        int3c = get_int3c_s2((0, nbas, 0, nbas, shl0, shl1))
        j3c[p0:p1] = int3c.T
        rhoj[:,p0:p1] = dm_tril.dot(int3c)
        int3c = None
    t1 = logger.timer_debug1(mf_grad, 'df grad intor (P|mn)', *t1)

    rhoj = solve_j2c(rhoj.T).T # returns (naux, nset)
    j3c = numpy.asarray(solve_j2c(j3c), order='C') # returns (naux, nao_pair)
    t1 = logger.timer_debug1(mf_grad, 'df grad cho_solve (P|Q) D_Qmn = (P|mn)', *t1)

    vj = numpy.zeros((nset,3,nao,nao))
    vk = numpy.zeros((nset,3,nao,nao))

    get_int3c_ip1 = _int3c_wrapper(mol, auxmol, intor='int3c2e_ip1', aosym='s1')   # (nabla,|)
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = int(min(max(max_memory * .5e6/8 / (nao**2*5), 20), naux, 240))
    fmmm_s2 = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans_s2 = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    null = lib.c_null_ptr()
    # (d/dX i,j|P)
    for shl0, shl1, nL in balance_partition(aux_loc, blksize):
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        int3c = get_int3c_ip1((0, nbas, 0, nbas, shl0, shl1)).transpose(0,3,2,1)
        for i in range(nset):
            vj[i,0] += numpy.dot(rhoj[i,p0:p1], int3c[0].reshape(p1-p0,-1)).reshape(nao,nao).T
            vj[i,1] += numpy.dot(rhoj[i,p0:p1], int3c[1].reshape(p1-p0,-1)).reshape(nao,nao).T
            vj[i,2] += numpy.dot(rhoj[i,p0:p1], int3c[2].reshape(p1-p0,-1)).reshape(nao,nao).T

        if with_k:
            rhok = numpy.empty((p1-p0,nao,nao))
            for i in range(nset):
                fdrv(ftrans_s2, fmmm_s2,
                     rhok.ctypes.data_as(ctypes.c_void_p),
                     j3c[p0:p1].ctypes.data_as(ctypes.c_void_p),
                     dms[i].ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(p1-p0), ctypes.c_int(nao),
                     (ctypes.c_int*4)(0, nao, 0, nao), null, ctypes.c_int(0))
                vk[i] += lib.einsum('xPmn,Pmk->xnk', int3c, rhok)
            rhok = None
        int3c = None
    t1 = logger.timer_debug1(mf_grad, 'df grad einsum (P|mn) D_Pmn = v_ij', *t1)

    if not mf_grad.auxbasis_response:
        vj = -vj.reshape(out_shape)
        vk = -vk.reshape(out_shape)
        logger.timer (mf_grad, 'df grad vj and vk', *t0)
        if with_j: return vj, vk
        else: return None, vk

    ####### BEGIN AUXBASIS PART #######
    vjaux = numpy.zeros((nset,nset,3,naux))
    vkaux = numpy.zeros((nset,nset,3,naux))
    rhok_pq = numpy.zeros((nset,naux,naux))

    # (i,j|d/dX P)
    get_int3c_ip2 = _int3c_wrapper(mol, auxmol, intor='int3c2e_ip2', aosym='s2ij') # (,|nabla)
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = int(min(max(max_memory * .5e6/8 / (nao**2*3), 20), naux, 240))
    for shl0, shl1, nL in balance_partition(aux_loc, blksize):
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        int3c = get_int3c_ip2((0, nbas, 0, nbas, shl0, shl1))
        drhoj = lib.dot (int3c.transpose (0,2,1).reshape (3*(p1-p0), -1),
            dm_tril.T).reshape (3, p1-p0, -1) # xpij,mij->xpm
        vjaux[:,:,:,p0:p1] = lib.einsum('xpm,np->mnxp', drhoj, rhoj[:,p0:p1])
        drhoj = None
        if with_k:
            for i in range(nset):
                rhok = _ao2mo.nr_e2(j3c[p0:p1], dms[i], (0,nao,0,nao),
                                    aosym='s2', mosym='s2') * 2
                rhok[:,idx] *= .5
                vkaux[i,i,:,p0:p1] += numpy.einsum('xkp,pk->xp', int3c, rhok)
                rhok_pq[i][:,p0:p1] = lib.dot(j3c, rhok.T)
                rhok = None
        int3c = None
    t1 = logger.timer_debug1(mf_grad, "df grad vj and vk aux (P'|mn) eval", *t1)

    # (d/dX P|Q)
    int2c_e1 = auxmol.intor('int2c2e_ip1')
    vjaux -= lib.einsum('xpq,mp,nq->mnxp', int2c_e1, rhoj, rhoj)
    if with_k:
        for i in range(nset):
            vkaux[i,i] -= lib.einsum('xpq,pq->xp', int2c_e1, rhok_pq[i])
    int2c_e1 = rhok_pq = j3c = None
    t1 = logger.timer_debug1(mf_grad, "df grad vj and vk aux (P'|Q) eval", *t1)

    vjaux = numpy.array([-vjaux[:,:,:,p0:p1].sum(axis=3) for p0, p1 in auxslice[:,2:]])
    vkaux = numpy.array([-vkaux[:,:,:,p0:p1].sum(axis=3) for p0, p1 in auxslice[:,2:]])

    vjaux = numpy.ascontiguousarray(vjaux.transpose(1,2,0,3))
    vkaux = numpy.ascontiguousarray(vkaux.transpose(1,2,0,3))

    vj = lib.tag_array(-vj.reshape(out_shape), aux=numpy.array(vjaux))
    vk = lib.tag_array(-vk.reshape(out_shape), aux=numpy.array(vkaux))
    logger.timer (mf_grad, 'df grad vj and vk', *t0)
    if with_j: return vj, vk
    else: return None, vk

class Gradients(rks_grad.Gradients):

    _keys = {'with_df', 'auxbasis_response'}

    def __init__(self, mf):
        rks_grad.Gradients.__init__(self, mf)

    auxbasis_response = True

    get_j = df_rhf_grad.Gradients.get_j
    get_k = df_rhf_grad.Gradients.get_k
    get_jk = get_jk    
    get_veff = get_veff

    def extra_force(self, atom_id, envs):
        e1 = rks_grad.Gradients.extra_force(self, atom_id, envs)
        if self.auxbasis_response:
            e1 += envs['vhf'].aux[atom_id]
        return e1

Grad = Gradients