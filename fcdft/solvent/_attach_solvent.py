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
        from fcdft.wbl.rks import WBLBase
        # Update key values if PBE is coupled with WBL
        if isinstance(self, WBLBase):
            self.with_solvent.bias = self.bias # eV
            self.with_solvent.nelectron = self.nelectron
            self.with_solvent.ref_pot = self.ref_pot # eV
        # If not, prepare the usual solvent model
        else:
            self.with_solvent.bias = 0
            self.with_solvent.nelectron = mol.nelectron
            self.with_solvent.ref_pot = 0

        # Running WBL first is strongly recommended to prevent divergence.
        # Otherwise, mismatch in the electron count causes serious SCF instability.
        if len(args) == 0 and isinstance(self, WBLBase):
            logger.info(self, 'Run FC-DFT first for stable PBE solver...')
            _scf = self.undo_solvent()
            _scf.kernel(dm0=dm)
            # Update key values and the density matrix accordingly.
            self.__dict__.update(_scf.__dict__)
            dm = self.make_rdm1()
            self.with_solvent.bias = self.bias

        # Effective potential from SCF (See pyscf/solvent/_attach_solvent.py)
        # Note that the parent class is _attach_solvent.SCFWithSolvent, not an SCF object.
        vhf = super(_attach_solvent.SCFWithSolvent, self).get_veff(mol, dm, *args, **kwargs)

        with_solvent = self.with_solvent

        if not with_solvent.frozen:
            with_solvent.e, with_solvent.v = with_solvent.kernel(dm)
        e_solvent, v_solvent = with_solvent.e, with_solvent.v

        return lib.tag_array(vhf, e_solvent=e_solvent, v_solvent=v_solvent)