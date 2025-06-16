import scipy
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import df
from pyscf.df.grad import rhf as df_rhf_grad
from pyscf.df.incore import LINEAR_DEP_THR
from pyscf.ao2mo import _ao2mo
from pyscf.grad import rks as _rks_grad
from fcdft.grad import rks as rks_grad
from itertools import product
import ctypes

LINEAR_DEP_THRESHOLD = LINEAR_DEP_THR

def get_jk(mf_grad, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
           decompose_j2c='CD', lindep=LINEAR_DEP_THRESHOLD):
    """Copied from pyscf.df.grad.rhf and modified some parts to allow complex density matrix."""
    assert (with_j or with_k)
    if not with_k:
        return get_j (mf_grad, mol=mol, dm=dm, hermi=hermi), None
    t0 = (logger.process_clock (), logger.perf_counter ())
    if mol is None: mol = mf_grad.mol
    if dm is None: dm = mf_grad.base.make_rdm1()
    with_df = mf_grad.base.with_df
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
    nbas, nao, naux = mol.nbas, mol.nao, auxmol.nao
    aux_loc = auxmol.ao_loc

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

    # For k
    orbol, orbor = df_rhf_grad._decompose_rdm1_svd (mf_grad, mol, dm)
    nocc = [o.shape[-1] for o in orbor]

    # Coulomb: (P|Q) D_Q = (P|uv) D_uv for D_Q ("rhoj")
    # Exchange: (P|Q) D_Qui = (P|uv) C_vi n_i for D_Qui ("rhok")
    rhoj, get_rhok = _cho_solve_rhojk (mf_grad, mol, auxmol, orbol, orbor,
                                       decompose_j2c, lindep)

    # (d/dX i,j|P)
    t1 = (logger.process_clock (), logger.perf_counter ())
    vj = numpy.zeros((nset,3,nao,nao), dtype=numpy.complex128)
    vk = numpy.zeros((nset,3,nao,nao), dtype=numpy.complex128)
    get_int3c_ip1 = df_rhf_grad._int3c_wrapper(mol, auxmol, 'int3c2e_ip1', 's1')
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = int(min(max(max_memory * .5e6/8 / (nao**2*3), 20), naux, 240))
    ao_ranges = df_rhf_grad.balance_partition(aux_loc, blksize)
    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s1 # MO output index slower than AO output index; input AOs are asymmetric
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv # comp and aux indices are slower
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s1 # input is not tril_packed
    null = lib.c_null_ptr()
    t2 = t1
    for shl0, shl1, nL in ao_ranges:
        int3c = get_int3c_ip1((0, nbas, 0, nbas, shl0, shl1)).transpose (0,3,2,1)  # (P|mn'), row-major order
        t2 = logger.timer_debug1 (mf_grad, "df grad intor (P|mn')", *t2)
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        for i in range(nset):
            # MRH 05/21/2020: De-vectorize this because array contiguity -> multithread efficiency
            vj[i,0] += numpy.dot (rhoj[i,p0:p1], int3c[0].reshape (p1-p0, -1)).reshape (nao, nao).T
            vj[i,1] += numpy.dot (rhoj[i,p0:p1], int3c[1].reshape (p1-p0, -1)).reshape (nao, nao).T
            vj[i,2] += numpy.dot (rhoj[i,p0:p1], int3c[2].reshape (p1-p0, -1)).reshape (nao, nao).T
            t2 = logger.timer_debug1 (mf_grad, "df grad einsum rho_P (P|mn') rho_P", *t2)
            tmpRe = numpy.empty ((3,p1-p0,nocc[i],nao), dtype=numpy.float64)
            tmpIm = numpy.empty ((3,p1-p0,nocc[i],nao), dtype=numpy.float64)
            orbolRe = numpy.asarray(orbol[i].real, order='F')
            orbolIm = numpy.asarray(orbol[i].imag, order='F')            
            fdrv(ftrans, fmmm, # lib.einsum ('xpmn,mi->xpin', int3c, orbol[i])
                 tmpRe.ctypes.data_as(ctypes.c_void_p),
                 int3c.ctypes.data_as(ctypes.c_void_p),
                 orbolRe.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int (3*(p1-p0)), ctypes.c_int (nao),
                 (ctypes.c_int*4)(0, nocc[i], 0, nao),
                 null, ctypes.c_int(0))
            fdrv(ftrans, fmmm, # lib.einsum ('xpmn,mi->xpin', int3c, orbol[i])
                 tmpIm.ctypes.data_as(ctypes.c_void_p),
                 int3c.ctypes.data_as(ctypes.c_void_p),
                 orbolIm.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int (3*(p1-p0)), ctypes.c_int (nao),
                 (ctypes.c_int*4)(0, nocc[i], 0, nao),
                 null, ctypes.c_int(0))
            tmp = tmpRe + 1.0j * tmpIm
            t2 = logger.timer_debug1 (mf_grad, "df grad einsum (P|mn') u_mi = dg_Pin", *t2)
            rhok = get_rhok (i, p0, p1)
            vk[i] += lib.einsum('xpoi,pok->xik', tmp, rhok)
            t2 = logger.timer_debug1 (mf_grad, "df grad einsum D_Pim dg_Pin = v_ij", *t2)
            rhok = tmp = None
        int3c = None
    t1 = logger.timer_debug1 (mf_grad, 'df grad vj and vk AO (P|mn) D_P eval', *t1)

    if not mf_grad.auxbasis_response:
        vj = -vj.reshape(out_shape)
        vk = -vk.reshape(out_shape)
        logger.timer (mf_grad, 'df grad vj and vk', *t0)
        if with_j: return vj, vk
        else: return None, vk

    ####### BEGIN AUXBASIS PART #######

    # ao2mo the final AO index of rhok and store in "rhok_oo":
    # dPiu C_uj -> dPij. *Not* symmetric i<->j: "i" has an occupancy
    # factor and "j" must not.
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = int(min(max(max_memory * .5e6/8 / (nao*max (nocc)), 20), naux))
    rhok_oo = []
    for i, j in product (range (nset), repeat=2):
        tmp = numpy.empty ((naux,nocc[i],nocc[j]), numpy.complex128)
        for p0, p1 in lib.prange(0, naux, blksize):
            rhok = get_rhok (i, p0, p1).reshape ((p1-p0)*nocc[i], nao)
            tmp[p0:p1] = lib.dot (rhok, orbol[j]).reshape (p1-p0, nocc[i], nocc[j])
        rhok_oo.append(tmp)
        rhok = tmp = None
    t1 = logger.timer_debug1 (mf_grad, 'df grad vj and vk aux d_Pim u_mj = d_Pij eval', *t1)

    vjaux = numpy.zeros((nset,nset,3,naux), dtype=numpy.complex128)
    vkaux = numpy.zeros((nset,nset,3,naux), dtype=numpy.complex128)
    # (i,j|d/dX P)
    t2 = t1
    get_int3c_ip2 = df_rhf_grad._int3c_wrapper(mol, auxmol, 'int3c2e_ip2', 's2ij')
    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2 # MO output index slower than AO output index; input AOs are symmetric
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv # comp and aux indices are slower
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2 # input is tril_packed
    null = lib.c_null_ptr()
    for shl0, shl1, nL in ao_ranges:
        int3c = get_int3c_ip2((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
        t2 = logger.timer_debug1 (mf_grad, "df grad intor (P'|mn)", *t2)
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        drhoj = lib.dot (int3c.transpose (0,2,1).reshape (3*(p1-p0), -1),
            dm_tril.T).reshape (3, p1-p0, -1) # xpij,mij->xpm
        vjaux[:,:,:,p0:p1] = lib.einsum ('xpm,np->mnxp', drhoj, rhoj[:,p0:p1])
        t2 = logger.timer_debug1 (mf_grad, "df grad vj aux (P'|mn) eval", *t2)
        # MRH, 09/19/2022: This is a different order of operations than PySCF v2.1.0. There,
        #                  the dense matrix rhok_oo is transformed into the larger AO basis.
        #                  Here, the sparse matrix int3c is transformed into the smaller MO
        #                  basis. The latter approach is obviously more performant.
        for i in range (nset):
            bufRe = numpy.empty ((3, p1-p0, nocc[i], nao), dtype=numpy.float64)
            bufIm = numpy.empty ((3, p1-p0, nocc[i], nao), dtype=numpy.float64)
            orbolRe = numpy.asarray(orbol[i].real, order='F')
            orbolIm = numpy.asarray(orbol[i].imag, order='F')
            fdrv(ftrans, fmmm, # lib.einsum ('pmn,ni->pim', int3c, orbol[i])
                    bufRe.ctypes.data_as(ctypes.c_void_p),
                    int3c.ctypes.data_as(ctypes.c_void_p),
                    orbolRe.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int (3*(p1-p0)), ctypes.c_int (nao),
                    (ctypes.c_int*4)(0, nocc[i], 0, nao),
                    null, ctypes.c_int(0))
            fdrv(ftrans, fmmm, # lib.einsum ('pmn,ni->pim', int3c, orbol[i])
                    bufIm.ctypes.data_as(ctypes.c_void_p),
                    int3c.ctypes.data_as(ctypes.c_void_p),
                    orbolIm.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int (3*(p1-p0)), ctypes.c_int (nao),
                    (ctypes.c_int*4)(0, nocc[i], 0, nao),
                    null, ctypes.c_int(0))
            buf = bufRe + 1.0j * bufIm
            for j in range (nset): # lib.einsum ('pim,mj->pij', buf, orbor[j])
                int3c_ij = lib.dot (buf.reshape (-1, nao), orbor[j])
                int3c_ij = int3c_ij.reshape (3, p1-p0, nocc[i], nocc[j])
                rhok_oo_ij = rhok_oo[(i*nset)+j][p0:p1]
                vkaux[i,j,:,p0:p1] += lib.einsum('xpij,pij->xp', int3c_ij,
                                                 rhok_oo_ij)
        t2 = logger.timer_debug1 (mf_grad, "df grad vk aux (P'|mn) eval", *t2)
    int3c = tmp = None
    t1 = logger.timer_debug1 (mf_grad, "df grad vj and vk aux (P'|mn) eval", *t1)

    # (d/dX P|Q)
    int2c_e1 = auxmol.intor('int2c2e_ip1')
    vjaux -= lib.einsum('xpq,mp,nq->mnxp', int2c_e1, rhoj, rhoj)
    for i, j in product (range (nset), repeat=2):
        k = (i*nset) + j
        l = (j*nset) + i
        tmp = lib.einsum('pij,qji->pq', rhok_oo[k], rhok_oo[l])
        vkaux[i,j] -= lib.einsum('xpq,pq->xp', int2c_e1, tmp)
    t1 = logger.timer_debug1 (mf_grad, "df grad vj and vk aux (P'|Q) eval", *t1)

    auxslices = auxmol.aoslice_by_atom()
    vjaux = numpy.array ([-vjaux[:,:,:,p0:p1].sum(axis=3) for p0, p1 in auxslices[:,2:]])
    vkaux = numpy.array ([-vkaux[:,:,:,p0:p1].sum(axis=3) for p0, p1 in auxslices[:,2:]])

    vjaux = numpy.ascontiguousarray (vjaux.transpose (1,2,0,3))
    vkaux = numpy.ascontiguousarray (vkaux.transpose (1,2,0,3))

    vj = lib.tag_array(-vj.reshape(out_shape), aux=numpy.array(vjaux))
    vk = lib.tag_array(-vk.reshape(out_shape), aux=numpy.array(vkaux))
    logger.timer (mf_grad, 'df grad vj and vk', *t0)
    if with_j: return vj, vk
    else: return None, vk

def get_j(mf_grad, mol=None, dm=None, hermi=0):
    # Hints:
    # i,j,k,l: AO indices
    # p,q: Auxiliary basis indices
    # a,b: Spin indices
    # x: Cartesian index (x=0,1,2 for x,y,z)
    if mol is None: mol = mf_grad.mol
    if dm is None: dm = mf_grad.base.make_rdm1()
    t0 = (logger.process_clock (), logger.perf_counter ())

    with_df = mf_grad.base.with_df
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
    nbas = mol.nbas

    # Computing speed has not been optimized.
    get_int3c_s1 = df_rhf_grad._int3c_wrapper(mol, auxmol, 'int3c2e', 's1')
    get_int3c_ip1 = df_rhf_grad._int3c_wrapper(mol, auxmol, 'int3c2e_ip1', 's1')
    get_int3c_ip2 = df_rhf_grad._int3c_wrapper(mol, auxmol, 'int3c2e_ip2', 's1')

    nao = mol.nao
    naux = auxmol.nao
    dms = numpy.asarray(dm)
    out_shape = dms.shape[:-2] + (3,) + dms.shape[-2:]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    aux_loc = auxmol.ao_loc
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = int(min(max(max_memory * .5e6/8 / (nao**2*3), 20), naux, 240))
    ao_ranges = df_rhf_grad.balance_partition(aux_loc, blksize)    

    # (ij|P), (nao, nao, naux)
    rhoj = numpy.empty((nset,naux), dtype=numpy.complex128)
    for shl0, shl1, nL in ao_ranges:
        int3c = get_int3c_s1((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        rhoj[:,p0:p1] = lib.einsum('ijp,aij->ap', int3c, dms)
        int3c = None

    # (P|Q), (naux, naux)
    int2c = auxmol.intor('int2c2e', aosym='s1')
    rhoj = scipy.linalg.solve(int2c, rhoj.T, assume_a='pos').T
    int2c = None

    # (\nabla ij|P), (3, nao, nao, naux)
    vj = numpy.zeros((nset,3,nao,nao), dtype=numpy.complex128)
    for shl0, shl1, nL in ao_ranges:
        int3c = get_int3c_ip1((0, nbas, 0, nbas, shl0, shl1))
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        vj += lib.einsum('xijp,ap->axij', int3c, rhoj[:,p0:p1])
        int3c = None
    
    if mf_grad.auxbasis_response:
        vjaux = numpy.empty((nset,nset,3,naux), dtype=numpy.complex128)
        #(ij|\nabla P)
        for shl0, shl1, nL in ao_ranges:
            int3c = get_int3c_ip2((0, nbas, 0, nbas, shl0, shl1))
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            vjaux[:,:,:,p0:p1] = lib.einsum('xijp,aij,bp->abxp', int3c, dms, rhoj[:,p0:p1])
            int3c = None
        
        # (\nabla P|Q)
        int2c_e1 = auxmol.intor('int2c2e_ip1', aosym='s1')
        vjaux -= lib.einsum('xpq,ap,bq->abxp', int2c_e1, rhoj, rhoj)
        int2c_e1 = None

        auxslices = auxmol.aoslice_by_atom()
        vjaux = numpy.array([-vjaux[:,:,:,p0:p1].sum(axis=3) for p0, p1 in auxslices[:,2:]])
        vjaux = numpy.ascontiguousarray(vjaux.transpose (1,2,0,3))
        vj = lib.tag_array(-vj.reshape(out_shape), aux=numpy.array(vjaux))
    else:
        vj = -vj.reshape(out_shape)
    logger.timer (mf_grad, 'df vj', *t0)
    return vj

def _cho_solve_rhojk (mf_grad, mol, auxmol, orbol, orbor,
                      decompose_j2c='CD', lindep=LINEAR_DEP_THRESHOLD):
    nset = len (orbol)
    nao, naux = mol.nao, auxmol.nao
    nbas, nauxbas = mol.nbas, auxmol.nbas
    ao_loc = mol.ao_loc
    nocc = [o.shape[-1] for o in orbor]

    int2c = auxmol.intor('int2c2e', aosym='s1')
    solve_j2c = df_rhf_grad._gen_metric_solver(int2c, decompose_j2c, lindep)
    int2c = None
    get_int3c_s1 = df_rhf_grad._int3c_wrapper(mol, auxmol, 'int3c2e', 's1')
    rhoj = numpy.zeros((nset,naux), dtype=numpy.complex128) # Modified part
    f_rhok = lib.H5TmpFile()
    t1 = (logger.process_clock (), logger.perf_counter ())
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = max_memory * .5e6/8 / (naux*nao)
    mol_ao_ranges = df_rhf_grad.balance_partition(ao_loc, blksize)
    nsteps = len(mol_ao_ranges)
    t2 = t1
    for istep, (shl0, shl1, nd) in enumerate(mol_ao_ranges):
        int3c = get_int3c_s1((0, nbas, shl0, shl1, 0, nauxbas))
        t2 = logger.timer_debug1 (mf_grad, 'df grad intor (P|mn)', *t2)
        p0, p1 = ao_loc[shl0], ao_loc[shl1]
        for i in range(nset):
            # MRH 05/21/2020: De-vectorize this because array contiguity -> multithread efficiency
            v = lib.dot(int3c.reshape (nao, -1, order='F').T, orbor[i]).reshape (naux, (p1-p0)*nocc[i])
            t2 = logger.timer_debug1 (mf_grad, 'df grad einsum (P|mn) u_ni N_i = v_Pmi', *t2)
            rhoj[i] += numpy.dot (v, orbol[i][p0:p1].ravel ())
            t2 = logger.timer_debug1 (mf_grad, 'df grad einsum v_Pmi u_mi = rho_P', *t2)
            v = solve_j2c(v)
            t2 = logger.timer_debug1 (mf_grad, 'df grad cho_solve (P|Q) D_Qmi = v_Pmi', *t2)
            f_rhok['%s/%s'%(i,istep)] = v.reshape(naux,p1-p0,-1)
            t2 = logger.timer_debug1 (mf_grad, 'df grad cache D_Pmi (m <-> i transpose upon retrieval)', *t2)
        int3c = v = None
    rhoj = solve_j2c(rhoj.T).T
    int2c = None
    t1 = logger.timer_debug1 (mf_grad, 'df grad vj and vk AO (P|Q) D_Q = (P|mn) D_mn solve', *t1)
    class get_rhok_class :
        def __init__(self, my_f):
            self.f_rhok = my_f
        def __call__(self, set_id, p0, p1):
            buf = numpy.empty((p1-p0,nocc[set_id],nao), dtype=numpy.complex128)
            col1 = 0
            for istep in range(nsteps):
                dat = self.f_rhok['%s/%s'%(set_id,istep)][p0:p1]
                col0, col1 = col1, col1 + dat.shape[1]
                buf[:p1-p0,:,col0:col1] = dat.transpose(0,2,1)
            return buf
    get_rhok = get_rhok_class (f_rhok)
    return rhoj, get_rhok

def get_veff(ks_grad, mol=None, dm=None):
    '''Coulomb + XC functional
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = _rks_grad._initialize_grids(ks_grad)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = _rks_grad.get_vxc_full_response(
            ni, mol, grids, mf.xc, dm,
            max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = _rks_grad.get_nlc_vxc_full_response(
                ni, mol, nlcgrids, xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
            exc += enlc
            vxc += vnlc
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, vxc = rks_grad.get_vxc(
            ni, mol, grids, mf.xc, dm,
            max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc(
                ni, mol, nlcgrids, xc, dm,
                max_memory=max_memory, verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(mol, dm)
        vxc += vj
        if ks_grad.auxbasis_response:
            e1_aux = vj.aux.sum ((0,1))
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        if ks_grad.auxbasis_response:
            vk.aux *= hyb
        vk[:] *= hyb # Don't erase the .aux tags!
        if omega != 0:  # For range separated Coulomb operator
            # TODO: replaced with vk_sr which is numerically more stable for
            # inv(int2c2e)
            vk_lr = ks_grad.get_k(mol, dm, omega=omega)
            vk[:] += vk_lr * (alpha - hyb)
            if ks_grad.auxbasis_response:
                vk.aux[:] += vk_lr.aux * (alpha - hyb)
        vxc += vj - vk * .5
        if ks_grad.auxbasis_response:
            e1_aux = (vj.aux - vk.aux * .5).sum ((0,1))

    if ks_grad.auxbasis_response:
        logger.debug1(ks_grad, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
        vxc = lib.tag_array(vxc, exc1_grid=exc, aux=e1_aux)
    else:
        vxc = lib.tag_array(vxc, exc1_grid=exc)
    return vxc


class Gradients(rks_grad.Gradients):
    def __init__(self, mf):
        rks_grad.Gradients.__init__(self, mf)

    auxbasis_response = True

    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
               omega=None):
        if omega is None:
            return get_jk(self, mol, dm, hermi, with_j, with_k)

        with self.base.with_df.range_coulomb(omega):
            return get_jk(self, mol, dm, hermi, with_j, with_k)

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        if omega is None:
            return get_j(self, mol, dm, hermi)

        with self.base.with_df.range_coulomb(omega):
            return get_j(self, mol, dm, hermi)

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):
        return self.get_jk(mol, dm, with_j=False, omega=omega)[1]

    get_veff = get_veff

Grad = Gradients