from fcdft.equil.equil import decompose_block_diagonal, decompose_block_column
from fcdft.equil.equil import Equilibrium
from fcdft.solvent.pbe import PBE
from fcdft.solvent._attach_solvent import SCFWithSolvent
from pyscf.solvent import _attach_solvent
from pyscf import lib
from pyscf.lib import logger
import numpy
import scipy

def pbe_for_equil(mf, solvent_obj=None, dm=None):
    if solvent_obj is None:
        solvent_obj = [PBE(mf.mol1), PBE(mf.mol2)]
    return _for_equil(mf, solvent_obj, dm)

def _for_equil(mf, solvent_obj, dm=None):
    sol_mf = EquilWithSolvent(mf, solvent_obj)
    name = solvent_obj[0].__class__.__name__ + mf.__class__.__name__
    return lib.set_class(sol_mf, (EquilWithSolvent, mf.__class__), name)

class EquilWithSolvent(SCFWithSolvent):
    def dump_flags(self, verbose=None):
        for solvent in self.with_solvent:
            # super().dump_flags(verbose)
            solvent.check_sanity()
            solvent.dump_flags(verbose)

    def reset(self, mol=None):
        from fcdft.gto.mole import split_mol
        mol1, mol2 = split_mol(mol, self.atom_slice)
        with_solvent = self.with_solvent
        with_solvent[0] = with_solvent[0].reset(mol1)
        with_solvent[1] = with_solvent[1].reset(mol2)
        self.with_solvent = with_solvent
        return super(Equilibrium, self).reset(mol)
    
    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        vhf = super(_attach_solvent.SCFWithSolvent, self).get_veff(mol, dm, *args, **kwargs)

        # Update key values if PBE is coupled with WBL
        from fcdft.wbl.rks import WBLBase
        for solvent in self.with_solvent:
            if isinstance(self, WBLBase):
                solvent.bias = self.bias # eV
                solvent.nelectron = self.nelectron
                solvent.ref_pot = self.ref_pot # eV
            # If not, prepare the usual solvent model
            else:
                solvent.bias = 0
                solvent.nelectron = mol.nelectron
                solvent.ref_pot = 0

        with_solvent = self.with_solvent
        dm1, dm2 = decompose_block_diagonal(self, dm)
        # Skipping first iteration
        if len(args) == 0:
            e_solvent1, v_solvent1 = 0.0e0, numpy.zeros_like(dm1)
            e_solvent2, v_solvent2 = 0.0e0, numpy.zeros_like(dm2)
        else:
            if not with_solvent[0].frozen:
                e_solvent1, v_solvent1 = with_solvent[0].kernel(dm1)
            if not with_solvent[1].frozen:
                e_solvent2, v_solvent2 = with_solvent[1].kernel(dm2)

        v_solvent = scipy.linalg.block_diag(v_solvent1, v_solvent2)
        cover = self.cover
        e_solvent = (1 - cover) * e_solvent1 + cover * e_solvent2

        return lib.tag_array(vhf, e_solvent=e_solvent, e_solvent1=e_solvent1, e_solvent2=e_solvent2,
                             v_solvent=v_solvent, v_solvent1=v_solvent1, v_solvent2=v_solvent2)
    
    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        if getattr(vhf, 'e_solvent', None) is None:
            vhf = self.get_veff(self.mol, dm)

        e_tot, e_coul, e_mol1, e_mol2 = super(_attach_solvent.SCFWithSolvent, self).energy_elec(dm, h1e, vhf)
        e_solvent = vhf.e_solvent
        e_solvent1, e_solvent2 = vhf.e_solvent1, vhf.e_solvent2
        e_tot += e_solvent
        self.scf_summary['e_solvent'] = vhf.e_solvent.real

        if (hasattr(self.with_solvent, 'method') and
            self.with_solvent.method.upper() == 'SMD'):
            if self.with_solvent.e_cds is None:
                e_cds = self.with_solvent.get_cds()
                self.with_solvent.e_cds = e_cds
            else:
                e_cds = self.with_solvent.e_cds

            if isinstance(e_cds, numpy.ndarray):
                e_cds = e_cds[0]
            e_tot += e_cds
            self.scf_summary['e_cds'] = e_cds
            logger.info(self, f'CDS correction = {e_cds:.15f}')
        logger.info(self, 'Solvent Energy(mol1) = %.15g', e_solvent1)
        logger.info(self, 'Solvent Energy(mol2) = %.15g', e_solvent2)
        logger.info(self, 'Solvent Energy = %.15g', vhf.e_solvent)

        return e_tot, e_coul, e_mol1+e_solvent1, e_mol2+e_solvent2

    def nuc_grad_method(self):
        from fcdft.equil.grad.equilpbe import make_grad_object
        return make_grad_object(self)
        # raise DeprecationWarning('Use the make_grad_object function from '
        #                     'pyscf.solvent.grad.ddcosmo_grad or '
        #                     'pyscf.solvent._ddcosmo_tdscf_grad instead.')
    
    Gradients = nuc_grad_method
    # def grad(self, dm):
    #     from fcdft.equil.grad import equilpbe
    #     return equilpbe.kernel(self, dm, self.verbose)