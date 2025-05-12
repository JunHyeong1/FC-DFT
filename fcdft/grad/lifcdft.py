from pyscf.grad import uks
from pyscf.lib import logger

class Gradients(uks.Gradients):
    def kernel(self):
        base = self.base
        mf1 = base.mf1
        mf2 = base.mf2
        delta = base.delta
        grad1 = mf1.nuc_grad_method()
        grad2 = mf2.nuc_grad_method()
        de1 = grad1.kernel()
        de2 = grad2.kernel()
        de = (1.0e0 - delta) * de1 + delta * de2
        self.de = de
        self._finalize()
        return self.de
    
    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s gradients ---------------',
                        self.base.__class__.__name__)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')          

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(atom='H 0 0 0; H 0 0 1', verbose=5)
    from fcdft.lifcdft.lifcdft import LIFCDFT
    mf = LIFCDFT(mol, xc='pbe', delta=0.1)
    from pyscf.geomopt.geometric_solver import optimize
    mol_opt = optimize(mf)