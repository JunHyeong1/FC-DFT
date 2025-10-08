import numpy
import scipy
from pyscf import lib
from pyscf import df
from pyscf.df import df_jk
from fcdft.df import df_jk as fcdft_df_jk
from fcdft.equil.equil import decompose_block_diagonal, decompose_block_column
from fcdft.wbl.rks import _get_veff as wblrks_get_veff

def equil_density_fit(mf, auxbasis=None, with_df=None, only_dfj=False):
    from pyscf import df
    from pyscf.scf import dhf

    if with_df is None:
        from fcdft.gto.mole import split_mol
        mol = mf.mol # Supermol
        if mf.mol1 is None or mf.mol2 is None:
            mol1, mol2 = split_mol(mol, mf.atom_slice)
            mf.mol1, mf.mol2 = mol1, mol2
        else:
            mol1, mol2 = mf.mol1, mf.mol2
        df1, df2 = df.DF(mol1), df.DF(mol2)
        df1.max_memory = df2.max_memory = mf.max_memory
        df1.stdout = df2.stdout = mf.stdout
        df1.verbose = df2.verbose = mf.verbose
        df1.auxbasis = df2.auxbasis = auxbasis
        with_df = [df1, df2]

    # Needs attention!!
    if isinstance(mf, _EqDFHF):
        if mf.with_df is None:
            mf.with_df = with_df
        elif getattr(mf.with_df, 'auxbasis', None) != auxbasis:
            #logger.warn(mf, 'DF might have been initialized twice.')
            mf = mf.copy()
            mf.with_df = with_df
            mf.only_dfj = only_dfj
        return mf
    # Needs attention!!

    dfmf = _EqDFHF(mf, with_df, only_dfj)
    return lib.set_class(dfmf, (_EqDFHF, mf.__class__))

class _EqDFHF(df_jk._DFHF):

    def build(self, mol=None):
        mol = self.mol
        if self.mol1 is None or self.mol2 is None:
            from fcdft.gto.mole import split_mol
            self.mol1, self.mol2 = split_mol(mol, self.atom_slice)
        with_df = self.with_df
        with_df[0].mol, with_df[1].mol = self.mol1, self.mol2
        with_df[0].build()
        with_df[1].build()
        return super(df_jk._DFHF, self).build()

    def _get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, *args, **kwargs):
        if dm is None: dm = self.make_rdm1()
        cover = self.cover
        _mol = [self.mol1, self.mol2]
        _grids = [self.grids1, self.grids2]
        _dm = list(decompose_block_diagonal(self, dm))
        _mo_coeff = list(decompose_block_diagonal(self, dm.mo_coeff))
        _mo_occ = list(decompose_block_column(self, dm.mo_occ))

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
            self.mol, self.grids = _mol[i], _grids[i]
            _vxc.append(wblrks_get_veff(self, _mol[i], _dm[i], _dm_last[i], _vhf_last[i], *args, **kwargs))

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

    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True, omega=None):
        if dm is None: dm = self.make_rdm1()
        with_df = self.with_df
        mol1, mol2 = with_df[0].mol, with_df[1].mol
        if not with_k:
            if mol == mol1:
                return fcdft_df_jk.get_j(self.with_df[0], dm, hermi, self.direct_scf_tol), None
            elif mol == mol2:
                return fcdft_df_jk.get_j(self.with_df[1], dm, hermi, self.direct_scf_tol), None
            else:
                raise RuntimeError
        else:
            with_df = self.with_df
            mol1, mol2 = with_df[0].mol, with_df[1].mol
            if mol == mol1:
                return fcdft_df_jk.get_jk(with_df[0], dm, hermi, with_j, with_k, omega)
                # return df_jk.get_jk(with_df[0], dm, hermi, with_j, with_k, omega)
            elif mol == mol2:
                return fcdft_df_jk.get_jk(with_df[1], dm, hermi, with_j, with_k, omega)
                # return df_jk.get_jk(with_df[1], dm, hermi, with_j, with_k, omega)
            else:
                raise RuntimeError

    def reset(self, mol=None):
        if mol is None: mol = self.mol
        mol1, mol2 = self.mol1, self.mol2
        with_df = self.with_df
        self.with_df = [with_df[0].reset(mol1), with_df[1].reset(mol2)]
        from pyscf.df import df
        return super(df_jk._DFHF, self).reset(mol)

    def nuc_grad_method(self):
        from fcdft.df.grad import equil
        return equil.Gradients(self)

    Gradients = nuc_grad_method

    def Hessian(self):
        raise NotImplementedError
    
    def to_gpu(self):
        raise NotImplementedError