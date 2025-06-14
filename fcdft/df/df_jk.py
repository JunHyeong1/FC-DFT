import numpy
import scipy
from pyscf.df import df_jk
from pyscf import lib
from pyscf.lib import logger
from fcdft import wbl

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

class _DFHF(df_jk._DFHF):
    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True, omega=None):
        if dm is None: dm = self.make_rdm1()
        if not with_k:
            return get_j(self.with_df, dm, hermi, self.direct_scf_tol), None
        else:
            return super().get_jk(mol, dm, hermi, with_j, with_k, omega)

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

