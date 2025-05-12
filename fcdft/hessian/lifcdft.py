from pyscf.hessian import rhf as rhf_hess

class Hessian(rhf_hess.HessianBase):
    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        delta = self.base.delta
        mf1 = self.base.mf1
        mf2 = self.base.mf2
        hess1 = mf1.Hessian().kernel()
        hess2 = mf2.Hessian().kernel()
        self.de = (1.0e0 - delta) * hess1 + delta * hess2
        return self.de

