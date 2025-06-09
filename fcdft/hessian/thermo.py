import numpy
from pyscf.data import nist
from pyscf.hessian.thermo import rotation_const, _get_rotor_type, rotational_symmetry_number

def quasi_thermo(model, freq, temperature=298.15, pressure=101325, cutoff1=0, cutoff2=0):
    """Extension of thermo.py to support quasi-harmonic approximation.

    Args:
        cutoff1 (float, optional): Uniform shift in cm-1. Defaults to 0.
        cutoff2 (float, optional): Imaginary frequency cutoff in cm-1. Defaults to 0.

    """
    mol = model.mol
    atom_coords = mol.atom_coords()
    mass = mol.atom_mass_list(isotope_avg=True)
    mass_center = numpy.einsum('z,zx->x', mass, atom_coords) / mass.sum()
    atom_coords = atom_coords - mass_center

    kB = nist.BOLTZMANN
    h = nist.PLANCK
    # c = nist.LIGHT_SPEED_SI
    # beta = 1. / (kB * temperature)
    R_Eh = kB*nist.AVOGADRO / (nist.HARTREE2J * nist.AVOGADRO)

    results = {}
    results['temperature'] = (temperature, 'K')
    results['pressure'] = (pressure, 'Pa')

    E0 = model.e_tot
    results['E0'] = (E0, 'Eh')

    # Electronic part
    results['S_elec' ] = (R_Eh * numpy.log(mol.multiplicity), 'Eh/K')
    results['Cv_elec'] = results['Cp_elec'] = (0, 'Eh/K')
    results['E_elec' ] = results['H_elec' ] = (E0, 'Eh')

    # Translational part. See also https://cccbdb.nist.gov/thermo.asp for the
    # partition function q_trans
    mass_tot = mass.sum() * nist.ATOMIC_MASS
    q_trans = ((2.0 * numpy.pi * mass_tot * kB * temperature / h**2)**1.5
               * kB * temperature / pressure)
    results['S_trans' ] = (R_Eh * (2.5 + numpy.log(q_trans)), 'Eh/K')
    results['Cv_trans'] = (1.5 * R_Eh, 'Eh/K')
    results['Cp_trans'] = (2.5 * R_Eh, 'Eh/K')
    results['E_trans' ] = (1.5 * R_Eh * temperature, 'Eh')
    results['H_trans' ] = (2.5 * R_Eh * temperature, 'Eh')

    # Rotational part
    rot_const = rotation_const(mass, atom_coords, 'GHz')
    results['rot_const'] = (rot_const, 'GHz')
    rotor_type = _get_rotor_type(rot_const)

    sym_number = rotational_symmetry_number(mol)
    results['sym_number'] = (sym_number, '')

    # partition function q_rot (https://cccbdb.nist.gov/thermo.asp)
    if rotor_type == 'ATOM':
        results['S_rot' ] = (0, 'Eh/K')
        results['Cv_rot'] = results['Cp_rot'] = (0, 'Eh/K')
        results['E_rot' ] = results['H_rot' ] = (0, 'Eh')
    elif rotor_type == 'LINEAR':
        B = rot_const[1] * 1e9
        q_rot = kB * temperature / (sym_number * h * B)
        results['S_rot' ] = (R_Eh * (1 + numpy.log(q_rot)), 'Eh/K')
        results['Cv_rot'] = results['Cp_rot'] = (R_Eh, 'Eh/K')
        results['E_rot' ] = results['H_rot' ] = (R_Eh * temperature, 'Eh')
    else:
        ABC = rot_const * 1e9
        q_rot = ((kB*temperature/h)**1.5 * numpy.pi**.5
                 / (sym_number * numpy.prod(ABC)**.5))
        results['S_rot' ] = (R_Eh * (1.5 + numpy.log(q_rot)), 'Eh/K')
        results['Cv_rot'] = results['Cp_rot'] = (1.5 * R_Eh, 'Eh/K')
        results['E_rot' ] = results['H_rot' ] = (1.5 * R_Eh * temperature, 'Eh')

    # Vibrational part. Quasi-harmonic approximation
    au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * numpy.pi)
    conv = nist.LIGHT_SPEED_SI * 100 / au2hz
    idx1 = freq.real > 0
    idx2 = numpy.logical_and(freq.imag < cutoff2 * conv, freq.imag > 0)
    _freq = numpy.concatenate((freq.real[idx1], freq.imag[idx2]))
    vib_temperature = _freq * au2hz * h / kB
    idx_cutoff = vib_temperature < cutoff1 * conv * au2hz * h / kB
    vib_temperature[idx_cutoff] = cutoff1 * conv * au2hz * h / kB
    # reduced_temperature
    rt = vib_temperature / max(1e-14, temperature)
    e = numpy.exp(-rt)

    ZPE = R_Eh * .5 * vib_temperature.sum()
    results['ZPE'] = (ZPE, 'Eh')

    results['S_vib' ] = (R_Eh * (rt*e/(1-e) - numpy.log(1-e)).sum(), 'Eh/K')
    results['Cv_vib'] = results['Cp_vib'] = (R_Eh * (e * rt**2/(1-e)**2).sum(), 'Eh/K')
    results['E_vib' ] = results['H_vib' ] = \
            (ZPE + R_Eh * temperature * (rt * e / (1-e)).sum(), 'Eh')

    results['G_elec' ] = (results['H_elec' ][0] - temperature * results['S_elec' ][0], 'Eh')
    results['G_trans'] = (results['H_trans'][0] - temperature * results['S_trans'][0], 'Eh')
    results['G_rot'  ] = (results['H_rot'  ][0] - temperature * results['S_rot'  ][0], 'Eh')
    results['G_vib'  ] = (results['H_vib'  ][0] - temperature * results['S_vib'  ][0], 'Eh')

    def _sum(f):
        keys = ('elec', 'trans', 'rot', 'vib')
        return sum(results.get(f+'_'+key, (0,))[0] for key in keys)
    results['S_tot' ] = (_sum('S' ), 'Eh/K')
    results['Cv_tot'] = (_sum('Cv'), 'Eh/K')
    results['Cp_tot'] = (_sum('Cp'), 'Eh/K')
    results['E_0K' ]  = (E0 + ZPE, 'Eh')
    results['E_tot' ] = (_sum('E'), 'Eh')
    results['H_tot' ] = (_sum('H'), 'Eh')
    results['G_tot' ] = (_sum('G'), 'Eh')

    return results