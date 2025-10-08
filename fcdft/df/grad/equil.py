from fcdft.df.grad import rks as rks_grad
from fcdft.grad import equil as eq_grad

class Gradients(eq_grad.Gradients):
    auxbasis_response = True

    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
               omega=None):
        with_df = self.base.with_df
        mol1, mol2 = self.base.mol1, self.base.mol2
        if mol == mol1:
            self.base.with_df = with_df[0]
        elif mol == mol2:
            self.base.with_df = with_df[1]
        else:
            raise RuntimeError

        if omega is None:
            vj, vk = rks_grad.get_jk(self, mol, dm, hermi, with_j, with_k)
            self.base.with_df = with_df
            return vj, vk

        with self.base.with_df.range_coulomb(omega):
            vj, vk = rks_grad.get_jk(self, mol, dm, hermi, with_j, with_k)
            self.base.with_df = with_df
            return vj, vk

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        if omega is None:
            return rks_grad.get_j(self, mol, dm, hermi)

        with self.base.with_df.range_coulomb(omega):
            return rks_grad.get_j(self, mol, dm, hermi)

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):
        return self.get_jk(mol, dm, with_j=False, omega=omega)[1]

Grad = Gradients