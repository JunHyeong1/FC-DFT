import numpy as np
from pyscf.gto.mole import Mole
from pyscf.scf.hf import SCF

def dump_freq(mf, freq_info, filename):
    """Dump the frequency information to a molden file.

    Args:
        mf (_type_): Either a SCF or Mole object.
        freq_info (_type_): Vibrational frequency information.
        filename (_type_): Output file name.
    """
    if isinstance(mf, Mole):
        mol = mf
    elif isinstance(mf, SCF):
        mol = mf.mol
    freq = freq_info['freq_wavenumber']
    norm_mode = freq_info['norm_mode']
    freq2 = (freq * freq).real
    freq = freq2 / abs(freq2) * np.sqrt(abs(freq2))
    with open(filename, 'w') as f:
        f.write("[Molden Format]\n")
        f.write("PySCF to Molden\n")
        f.write("[Atoms] (AU)\n")
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            x, y, z = mol.atom_coord(ia)
            atom_num = mol.atom_charge(ia)
            f.write("%-4s %4d %4d %12.6f %12.6f %12.6f\n" % (symb, ia+1, atom_num, x, y, z))
        f.write("[FR-COORD]\n")
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            x, y, z = mol.atom_coord(ia)
            f.write("%-4s %12.6f %12.6f %12.6f\n" % (symb, x, y, z))
        f.write("[FREQ]\n")
        for i in range(len(freq)):
            f.write("%12.6f\n" % freq[i])
        f.write("[FR-NORM-COORD]\n")
        # f.write("%5d\n" % len(norm_mode))
        for idx, mode in enumerate(norm_mode):
            f.write("vibration%5d\n" % (idx+1))
            for ia in range(mol.natm):
                x, y, z = mode[ia]
                f.write("%12.6f %12.6f %12.6f\n" % (x, y, z))