from pyscf.dft import uks, UKS
from pyscf import lib
from pyscf.lib import logger

_SCFkeys = {
    'conv_tol', 'conv_tol_grad', 'conv_tol_cpscf', 'max_cycle', 'init_guess',
    'sap_basis', 'diis', 'diis_space', 'diis_damp', 'diis_start_cycle',
    'diis_file', 'diis_space_rollback', 'damp', 'level_shift',
    'direct_scf', 'direct_scf_tol', 'conv_check', 'callback',
    'mo_energy', 'mo_coeff', 'mo_occ',
    'e_tot', 'converged', 'cycles', 'scf_summary', 'opt',
    'disp', 'disp_with_3body',
}

class LIFCDFT(uks.UKS):
    def __init__(self, mol, xc='LDA,VWN', delta=0.0):
        super().__init__(mol, xc=xc)
        self.delta = delta
        self.mol = mol
        self.xc = xc
        self.mf1 = None # N
        self.mf2 = None # N+1
        self.mol1 = None # N
        self.mol2 = None # N+1
        self.dm1 = None
        self.dm2 = None
        self.gpu_accel = False
        self.newton_flag = False

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'delta = %.5f', self.delta)
        logger.info(self, 'Total number of electrons: %.5f', self.mol.nelectron+self.delta)
    
    def build(self):
        if self.delta < 0:
            raise AttributeError('Positive delta allowed.')
        mol = self.mol
        charge = mol.charge
        spin = mol.spin
        mol1 = mol.copy()
        mol2 = mol.copy()
        mol1.charge = charge
        mol1.spin = spin
        mol2.charge = charge-1
        mol2.spin = spin-1
        self.mf1 = UKS(mol1, self.xc)
        self.mf2 = UKS(mol2, self.xc)
        # from pyscf.scf.hf import SCF
        for attr in dir(self):
            if attr in _SCFkeys:
                value = getattr(self, attr)
                setattr(self.mf1, attr, value)
                setattr(self.mf2, attr, value)
        if self.gpu_accel:
            self.mf1 = self.mf1.to_gpu()
            self.mf2 = self.mf2.to_gpu()
        if self.newton_flag:
            self.mf1 = self.mf1.newton()
            self.mf2 = self.mf2.newton()
        self.mol1, self.mol2 = mol1, mol2
        return self

    def nuc_grad_method(self):
        from fcdft.grad import lifcdft
        return lifcdft.Gradients(self)
    
    def reset(self, mol=None):
        super().reset(mol)
        self.mf1 = None
        self.mf2 = None
        return self

    def newton(self):
        self.newton_flag = True
        return self

    def kernel(self, dm0=None):
        self.dump_flags()
        if self.mf1 is None and self.mf2 is None: self.build()
        mf1 = self.mf1
        mf2 = self.mf2
        dm1 = self.dm1
        dm2 = self.dm2
        e1 = mf1.kernel(dm0=dm2)
        dm1 = mf1.make_rdm1()
        e2 = mf2.kernel(dm0=dm1)
        dm2 = mf2.make_rdm1()
        self.dm1, self.dm2 = dm1, dm2
        delta = self.delta
        e_tot = (1.0e0 - delta) * e1 + delta * e2
        self.e_tot = e_tot
        logger.info(self, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
        logger.info(self, ' |           Perdew-Parr-Levy-Balduz condition          |')
        logger.info(self, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
        logger.info(self, ' E(N) = %.15f E(N+1) = %.15f', e1, e2)
        logger.info(self, ' delta = %.5f, E(N+delta) = %.15f', delta, e_tot)
        logger.info(self, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
        if mf1.converged and mf2.converged:
            self.converged = True
        self._finalize()
        return self.e_tot
    
    def _finalize(self):
        if self.converged:
            logger.note(self, 'converged SCF energy = %.15g', self.e_tot)
        else:
            logger.note(self, 'SCF not converged.')
            logger.note(self, 'SCF energy = %.15g', self.e_tot)
        return self
    
from fcdft.hessian.lifcdft import Hessian as lifcdft_Hessian
LIFCDFT.Hessian = lib.class_as_method(lifcdft_Hessian)

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(atom='H 0 0 0; H 0 0 1', verbose=5)
    mf = LIFCDFT(mol, xc='pbe', delta=0.1)
    mf.damp = 0.3
    mf.conv_tol=1.0e-7
    from pyscf.geomopt.geometric_solver import optimize
    optimize(mf)