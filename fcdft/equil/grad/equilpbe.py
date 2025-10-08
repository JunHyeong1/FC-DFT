from fcdft.solvent.grad import pbe
from fcdft.equil.equil import decompose_block_diagonal
from pyscf.grad import rhf as rhf_grad
from pyscf import lib
from pyscf.lib import logger
import numpy

def make_grad_object(base_method):
    from pyscf.solvent._attach_solvent import _Solvation
    if isinstance(base_method, rhf_grad.GradientsBase):
        base_method = base_method.base

    assert isinstance(base_method, _Solvation)
    with_solvent = base_method.with_solvent

    if with_solvent[0].frozen or with_solvent[1].frozen:
        raise RuntimeError('Frozen solvent model is not available for energy gradients')

    vac_grad = base_method.undo_solvent().Gradients()
    vac_grad.base = base_method
    name = (base_method.with_solvent[0].__class__.__name__
            + vac_grad.__class__.__name__)
    return lib.set_class(WithSolventGrad(vac_grad),
                         (WithSolventGrad, vac_grad.__class__), name)

class WithSolventGrad(pbe.WithSolventGrad):
    def kernel(self, *args, dm=None, **kwargs):
        dm = kwargs.pop('dm', None)
        if dm is None:
            dm = self.base.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]

        cover = self.base.cover
        dm1, dm2 = decompose_block_diagonal(self.base, dm)
        de_solvent1 = pbe.kernel(self.base.with_solvent[0], dm1)
        de_solvent2 = pbe.kernel(self.base.with_solvent[1], dm2)
        self.de_solvent = numpy.concatenate(((1 - cover) * de_solvent1, cover * de_solvent2))
        self.de_solute = super(pbe.WithSolventGrad, self).kernel(*args, **kwargs)
        self.de = self.de_solute + self.de_solvent

        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s (+%s) gradients ---------------',
                        self.base.__class__.__name__,
                        self.base.with_solvent[0].__class__.__name__)
            rhf_grad._write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')
        return self.de