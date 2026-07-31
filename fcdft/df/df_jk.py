from pyscf.df import df_jk
from pyscf import lib
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

class _DFHF(df_jk._DFHF):
    """Inject nuc_grad_method"""
    def nuc_grad_method(self):
        if isinstance(self, wbl.rks.WBLMoleculeRKS):
            from fcdft.df.grad import rks
            return rks.Gradients(self)
        elif isinstance(self, wbl.uks.WBLMoleculeUKS):
            from fcdft.df.grad import uks
            return uks.Gradients(self)
        else:
            raise NotImplementedError

    Gradients = nuc_grad_method

    def Hessian(self):
        raise NotImplementedError
    
    def to_gpu(self):
        raise NotImplementedError