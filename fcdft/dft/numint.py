from pyscf.dft import numint
from pyscf.dft.numint import eval_rho1
import numpy

class NumInt(numint.NumInt):
    """Routing to eval_rho1"""
    def _gen_rho_evaluator(self, mol, dms, hermi=0, with_lapl=True, grids=None):
        if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
            dms = dms[numpy.newaxis]
        nao = dms[0].shape[0]
        ndms = len(dms)
        def make_rho(idm, ao, sindex, xctype):
            return eval_rho1(mol, ao, dms[idm], sindex, xctype, hermi, with_lapl)
        return make_rho, ndms, nao