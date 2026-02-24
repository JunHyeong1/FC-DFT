import numpy
import scipy
from pyscf.df import df_jk
from pyscf import lib
from pyscf.lib import logger
from fcdft import wbl
import ctypes
import os
import fcdft
from pyscf.lib.numpy_helper import _dgemm

libdf = lib.load_library(os.path.join(fcdft.__path__[0], 'lib', 'libdf'))

def density_fit(mf, auxbasis=None, with_df=None, only_dfj=False):
    from pyscf import df
    from pyscf.scf import dhf
    assert (isinstance(mf, wbl.rks.WBLBase))

    if with_df is None:
        with_df = df.DF(mf.mol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    if isinstance(mf, _DFHF):
        if mf.with_df is None:
            mf.with_df = with_df
        elif getattr(mf.with_df, 'auxbasis', None) != auxbasis:
            #logger.warn(mf, 'DF might have been initialized twice.')
            mf = mf.copy()
            mf.with_df = with_df
            mf.only_dfj = only_dfj
        return mf

    dfmf = _DFHF(mf, with_df, only_dfj)
    return lib.set_class(dfmf, (_DFHF, mf.__class__))

def get_j(dfobj, dm, hermi=0, direct_scf_tol=1e-13):
    """ 
    PySCF does not support complex density matrix for vj.
    Only the real part of dm is taken for constructing vj matrix.
    """
    from pyscf.scf import _vhf
    from pyscf.scf import jk
    from pyscf.df import addons
    t0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = dfobj.mol
    if dfobj._vjopt is None:
        dfobj.auxmol = auxmol = addons.make_auxmol(mol, dfobj.auxbasis)
        opt = _vhf._VHFOpt(mol, 'int3c2e', 'CVHFnr3c2e_schwarz_cond',
                           dmcondname='CVHFnr_dm_cond',
                           direct_scf_tol=direct_scf_tol)

        # q_cond part 1: the regular int2e (ij|ij) for mol's basis
        opt.init_cvhf_direct(mol, 'int2e', 'CVHFnr_int2e_q_cond')

        # Update q_cond to include the 2e-integrals (auxmol|auxmol)
        j2c = auxmol.intor('int2c2e', hermi=1)
        j2c_diag = numpy.sqrt(abs(j2c.diagonal()))
        aux_loc = auxmol.ao_loc
        aux_q_cond = [j2c_diag[i0:i1].max()
                      for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
        q_cond = numpy.hstack((opt.q_cond.ravel(), aux_q_cond))
        opt.q_cond = q_cond

        try:
            opt.j2c = j2c = scipy.linalg.cho_factor(j2c, lower=True)
            opt.j2c_type = 'cd'
        except scipy.linalg.LinAlgError:
            opt.j2c = j2c
            opt.j2c_type = 'regular'

        # jk.get_jk function supports 4-index integrals. Use bas_placeholder
        # (l=0, nctr=1, 1 function) to hold the last index.
        bas_placeholder = numpy.array([0, 0, 1, 1, 0, 0, 0, 0],
                                      dtype=numpy.int32)
        fakemol = mol + auxmol
        fakemol._bas = numpy.vstack((fakemol._bas, bas_placeholder))
        opt.fakemol = fakemol
        dfobj._vjopt = opt
        t1 = logger.timer_debug1(dfobj, 'df-vj init_direct_scf', *t1)

    opt = dfobj._vjopt
    fakemol = opt.fakemol
    # Modified. vj only depends on the real part.
    dm = numpy.asarray(dm.real, order='C')
    assert dm.dtype == numpy.float64
    dm_shape = dm.shape
    nao = dm_shape[-1]
    dm = dm.reshape(-1,nao,nao)
    n_dm = dm.shape[0]

    # First compute the density in auxiliary basis
    # j3c = fauxe2(mol, auxmol)
    # jaux = numpy.einsum('ijk,ji->k', j3c, dm)
    # rho = numpy.linalg.solve(auxmol.intor('int2c2e'), jaux)
    nbas = mol.nbas
    nbas1 = mol.nbas + dfobj.auxmol.nbas
    shls_slice = (0, nbas, 0, nbas, nbas, nbas1, nbas1, nbas1+1)
    with lib.temporary_env(opt, prescreen='CVHFnr3c2e_vj_pass1_prescreen'):
        jaux = jk.get_jk(fakemol, dm, ['ijkl,ji->kl']*n_dm, 'int3c2e',
                         aosym='s2ij', hermi=0, shls_slice=shls_slice,
                         vhfopt=opt)
    # remove the index corresponding to bas_placeholder
    jaux = numpy.array(jaux)[:,:,0]
    t1 = logger.timer_debug1(dfobj, 'df-vj pass 1', *t1)

    if opt.j2c_type == 'cd':
        rho = scipy.linalg.cho_solve(opt.j2c, jaux.T)
    else:
        rho = scipy.linalg.solve(opt.j2c, jaux.T)
    # transform rho to shape (:,1,naux), to adapt to 3c2e integrals (ij|k)
    rho = rho.T[:,numpy.newaxis,:]
    t1 = logger.timer_debug1(dfobj, 'df-vj solve ', *t1)

    # Next compute the Coulomb matrix
    # j3c = fauxe2(mol, auxmol)
    # vj = numpy.einsum('ijk,k->ij', j3c, rho)
    # temporarily set "_dmcondname=None" to skip the call to set_dm method.
    with lib.temporary_env(opt, prescreen='CVHFnr3c2e_vj_pass2_prescreen',
                           _dmcondname=None):
        # CVHFnr3c2e_vj_pass2_prescreen requires custom dm_cond
        aux_loc = dfobj.auxmol.ao_loc
        dm_cond = [abs(rho[:,:,i0:i1]).max()
                   for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
        opt.dm_cond = numpy.array(dm_cond)
        vj = jk.get_jk(fakemol, rho, ['ijkl,lk->ij']*n_dm, 'int3c2e',
                       aosym='s2ij', hermi=1, shls_slice=shls_slice,
                       vhfopt=opt)

    t1 = logger.timer_debug1(dfobj, 'df-vj pass 2', *t1)
    logger.timer(dfobj, 'df-vj', *t0)
    return numpy.asarray(vj).reshape(dm_shape)

# numpy.einsum replaced by custom dgemm for computational efficiency. 
def get_jk(dfobj, dm, hermi=0, with_j=True, with_k=True, direct_scf_tol=1e-13):
    assert (with_j or with_k)
    if (not with_k and not dfobj.mol.incore_anyway and
        # 3-center integral tensor is not initialized
        dfobj._cderi is None):
        return get_j(dfobj, dm, hermi, direct_scf_tol), None

    t0 = t1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(dfobj.stdout, dfobj.verbose)

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    vj = 0
    vk = numpy.zeros((2,) + dms.shape)
    dmsReIm = numpy.array([dms.real, dms.imag], dtype=numpy.float64, order='C')
    fdrv = libdf.nr_df_contract_k

    if numpy.iscomplexobj(dms):
        if with_j:
            vj = numpy.zeros_like(dms)
        max_memory = dfobj.max_memory - lib.current_memory()[0]
        blksize = max(4, int(min(dfobj.blockdim, max_memory*.22e6/8/nao**2)))
        buf = numpy.empty((blksize,nao,nao))
        buf2 = numpy.empty((naux,nao,nao), dtype=numpy.float64, order='C')
        for eri1 in dfobj.loop(blksize):
            naux, nao_pair = eri1.shape
            eri1 = lib.unpack_tril(eri1, out=buf)
            if with_j:
                tmp = numpy.tensordot(eri1, dms.real, axes=([1,2],[2,1]))
                vj.real += numpy.tensordot(tmp.T, eri1, axes=([1],[0]))
                tmp = numpy.tensordot(eri1, dms.imag, axes=([1,2],[2,1]))
                vj.imag += numpy.tensordot(tmp.T, eri1, axes=([1],[0]))
            for k in range(nset):
                fdrv(vk[:,k].ctypes.data_as(ctypes.c_void_p),
                     buf2.ctypes.data_as(ctypes.c_void_p),
                     eri1.ctypes.data_as(ctypes.c_void_p),
                     dmsReIm[:,k].ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(nao), ctypes.c_int(naux))
            t1 = log.timer_debug1('jk', *t1)
        vk = vk[0] + 1.0j*vk[1]
        if with_j: vj = vj.reshape(dm_shape)
        if with_k: vk = vk.reshape(dm_shape)
        logger.timer(dfobj, 'df vj and vk', *t0)
        return vj, vk

    else:
        return df_jk.get_jk(dfobj, dm, hermi, with_j, with_k, direct_scf_tol)

# def test_get_jk(dfobj, dm, hermi=0, with_j=True, with_k=True, direct_scf_tol=1e-13):
#     assert (with_j or with_k)
#     if (not with_k and not dfobj.mol.incore_anyway and
#         # 3-center integral tensor is not initialized
#         dfobj._cderi is None):
#         return get_j(dfobj, dm, hermi, direct_scf_tol), None

#     t0 = t1 = (logger.process_clock(), logger.perf_counter())
#     log = logger.Logger(dfobj.stdout, dfobj.verbose)

#     dms = numpy.asarray(dm)
#     dm_shape = dms.shape
#     nao = dm_shape[-1]
#     dms = dms.reshape(-1,nao,nao)
#     nset = dms.shape[0]
#     vj = 0
#     vk = numpy.zeros_like(dms)

#     fdrv = libdf.nr_mapdm1

#     if numpy.iscomplexobj(dms):
#         for k in range(nset):
#             assert (scipy.linalg.issymeetric(dms[k], rtol=1e-10))
#         if with_j:
#             idx = numpy.arange(nao)
#             # Different orthogonality condition
#             dmtril = lib.pack_tril(dms + dms.transpose(0,2,1))
#             dmtril[:,idx*(idx+1)//2+idx] *= .5
        
#         if not with_k:
#             for eri1 in dfobj.loop():
#                 vj += dmtril.dot(eri1.T).dot(eri1)

#         elif getattr(dm, 'mo_coeff', None) is not None:
#             mo_coeff = numpy.asarray(dm.mo_coeff, order='F')
#             mo_occ   = numpy.asarray(dm.mo_occ)
#             nmo = mo_occ.shape[-1]
#             mo_coeff = mo_coeff.reshape(-1,nao,nmo)
#             mo_occ   = mo_occ.reshape(-1,nmo)
#             if mo_occ.shape[0] * 2 == nset: # handle ROHF DM
#                 raise NotImplementedError('ROHF not supported.')

#             orbo = []
#             for k in range(nset):
#                 c = numpy.einsum('pi,i->pi', mo_coeff[k], numpy.sqrt(mo_occ[k]))
#                 orbo.append(numpy.asarray(c, order='C'))

#         return vj, vk

#     else:
#         return df_jk.get_jk(dfobj, dm, hermi, with_j, with_k, direct_scf_tol)    

class _DFHF(df_jk._DFHF):
    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True, omega=None):
        if dm is None: dm = self.make_rdm1()
        if not with_k:
            return get_j(self.with_df, dm, hermi, self.direct_scf_tol), None
        else:
            return get_jk(self.with_df, dm, hermi, with_j, with_k, self.direct_scf_tol)

    def nuc_grad_method(self):
        from fcdft.df.grad import rks, uks
        if isinstance(self, wbl.rks.WBLMoleculeRKS):
            return rks.Gradients(self)
        elif isinstance(self, wbl.uks.WBLMoleculeUKS):
            return uks.Gradients(self)
        else:
            raise NotImplementedError

    Gradients = nuc_grad_method

    def Hessian(self):
        raise NotImplementedError
    
    def to_gpu(self):
        raise NotImplementedError

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(atom='''
    C       -1.130409485      0.110509105      2.487624324
    C       -1.133152455      0.172317410      3.876550456
    C        0.083208438      0.079010195      1.779347136
    H       -2.073077371      0.196405844      4.417984513
    C        0.080765072      0.203892233      4.586608119
    C        1.294918790      0.110563494      2.488622797
    H        2.242144342      0.086971817      1.958502899
    C        1.294972435      0.172423808      3.878759077
    H        2.234575318      0.196549301      4.420961499
    H       -2.074264979      0.086638346      1.951309936
    C        0.080417450      0.267549014      6.018388645
    N        0.080518923      0.319251712      7.181712450
    S        0.002877593     -0.000010379      0.004499969
    H        1.331118166     -0.009827639     -0.217185412
    C        3.634787838      3.729719574      1.046675071
    C        4.921304115      3.816141915      1.598818260
    C        2.525002270      3.824067407      1.897656464
    H        5.795404410      3.734570605      0.958740568
    H        1.517466732      3.771555033      1.494543863
    C        5.087886010      3.992640160      2.971026952
    C        2.703287147      4.000397050      3.269460579
    H        6.094436491      4.031213032      3.377171707
    H        1.826960115      4.095521496      3.904127827
    C        3.985308746      4.088859641      3.838437268
    C        4.168269362      4.276749528      5.300566278
    C        5.176071782      5.121260505      5.801087081
    C        3.338892534      3.615132798      6.224581373
    C        5.349757682      5.296463944      7.175273989
    C        3.509250264      3.792830513      7.598862256
    C        4.516356822      4.633894445      8.081264553
    H        5.812406721      5.663617748      5.107628032
    H        2.570932282      2.936880306      5.863729374
    H        6.130250569      5.959719714      7.537834346
    H        2.861223112      3.265171072      8.293257477
    H        4.650275861      4.771150299      9.150374259
    S        3.492564144      3.506912708     -0.719087092
    H        4.820818581      3.497095346     -0.940774787'''
                , basis='6-31++g**', verbose=9, max_memory=8000)
    from fcdft.wbl.rks import WBLMoleculeRKS
    wblmf = WBLMoleculeRKS(mol, xc='b3lyp-d3zero', nelectron=168, broad=0.08)
    wblmf.max_cycle=1
    wblmf = wblmf.density_fit()
    wblmf.kernel()

    with_df = wblmf.with_df
    dm = wblmf.make_rdm1()
    import ipdb
    ipdb.set_trace()