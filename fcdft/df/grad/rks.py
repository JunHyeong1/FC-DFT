from pyscf.df.grad import rhf as df_rhf_grad
from pyscf.df.grad.rks import get_veff
from fcdft.grad import rks as rks_grad

class Gradients(rks_grad.Gradients):

    _keys = {'with_df', 'auxbasis_response'}

    def __init__(self, mf):
        rks_grad.Gradients.__init__(self, mf)

    auxbasis_response = True

    get_jk = df_rhf_grad.Gradients.get_jk
    get_j = df_rhf_grad.Gradients.get_j
    get_k = df_rhf_grad.Gradients.get_k
    get_veff = get_veff

    def extra_force(self, atom_id, envs):
        e1 = rks_grad.Gradients.extra_force(self, atom_id, envs)
        if self.auxbasis_response:
            e1 += envs['vhf'].aux[atom_id]
        return e1

Grad = Gradients