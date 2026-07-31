from pyscf.df.grad import rhf as df_uhf_grad
from pyscf.df.grad.uks import get_veff
from fcdft.grad import uks as uks_grad

class Gradients(uks_grad.Gradients):
    def __init__(self, mf):
        uks_grad.Gradients.__init__(self, mf)

    auxbasis_response = True

    get_jk = df_uhf_grad.Gradients.get_jk
    get_j = df_uhf_grad.Gradients.get_j
    get_k = df_uhf_grad.Gradients.get_k
    get_veff = get_veff

    def extra_force(self, atom_id, envs):
        e1 = uks_grad.Gradients.extra_force(self, atom_id, envs)
        if self.auxbasis_response:
            e1 += envs['vhf'].aux[atom_id]
        return e1

Grad = Gradients