from pyscf.solvent import _attach_solvent
from pyscf import lib
from pyscf.lib import logger

# This code is to inject WBLMolecule.fermi to PBE objects.
def _for_scf(mf, solvent_obj, dm=None):
    '''Add solvent model to SCF (HF and DFT) method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(mf, _attach_solvent._Solvation):
        mf.with_solvent = solvent_obj
        return mf

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_mf = SCFWithSolvent(mf, solvent_obj)
    name = solvent_obj.__class__.__name__ + mf.__class__.__name__
    return lib.set_class(sol_mf, (SCFWithSolvent, mf.__class__), name)

class SCFWithSolvent(_attach_solvent.SCFWithSolvent):       
    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        vhf = super(_attach_solvent.SCFWithSolvent, self).get_veff(mol, dm, *args, **kwargs)

        # Update key values if PBE is coupled with WBL
        from fcdft.wbl.rks import WBLBase
        if isinstance(self, WBLBase):
            self.with_solvent.bias = self.bias # eV
            self.with_solvent.nelectron = self.nelectron
            self.with_solvent.ref_pot = self.ref_pot # eV
        # If not, prepare the usual solvent model
        else:
            self.with_solvent.bias = 0
            self.with_solvent.nelectron = mol.nelectron
            self.with_solvent.ref_pot = 0

        with_solvent = self.with_solvent

        # Skipping first iteration
        if len(args) == 0:
            e_solvent, v_solvent = 0.0e0, 0.0e0
        else:
            if not with_solvent.frozen:
                e_solvent, v_solvent = with_solvent.kernel(dm)
        with_solvent.e, with_solvent.v = e_solvent, v_solvent

        return lib.tag_array(vhf, e_solvent=e_solvent, v_solvent=v_solvent)