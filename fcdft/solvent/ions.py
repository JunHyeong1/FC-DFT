import numpy
from pyscf import lib
from pyscf.data.nist import BOHR
from fcdft.solvent.pbe import M2HARTREE, KB2HARTREE

PI = numpy.pi

def _one_to_one(solvent_obj, phi_tot=None, cb=None, lambda_r=None, T=None):
    """Size-modified 1:1 ion charge density.

    Args:
        solvent_obj (:class:`PBE`): Solvent object
        phi_tot (1D numpy.ndarray, optional): Total electrostatic potential. Defaults to None.
        cb (float, optional): Ion concentration in atomic unit. Defaults to None.
        lambda_r (1D numpy.ndarray, optional): Ion-exclusion function. Defaults to None.
        T (float, optional): Temperature. Defaults to None.

    Returns:
        1D numpy.ndarray: Ion charge density
    """
    if phi_tot is None: phi_tot = solvent_obj.phi_tot
    if cb is None: cb = solvent_obj.cb * M2HARTREE
    if lambda_r is None: lambda_r = solvent_obj._intermediates['lambda_r']
    if T is None: T = solvent_obj.T

    cation_rad = solvent_obj.cation_rad / BOHR
    anion_rad = solvent_obj.anion_rad / BOHR
    c12 = 0.74e0 / (4.0e0/3.0e0 * PI * (cation_rad**3 + anion_rad**3))
    _lambda_r = lambda_r.copy()
    _lambda_r[lambda_r < 1.0e-100] = 1.0e-100
    lnlambda = numpy.log(_lambda_r)
    lnlambda[lambda_r < 1.0e-100] = -1.0e100
    if cb == 0.0e0:
        return numpy.zeros_like(phi_tot)
    else:
        lncb = numpy.log(cb)

    x = phi_tot / (KB2HARTREE * T)
    lnA = lnlambda + lncb - x
    lnB = lnlambda + lncb + x
    lnC = numpy.log(0.5e0) + lnlambda - x
    lnD = numpy.log(0.5e0) + lnlambda + x
    rho_ions = (numpy.exp(lnA) - numpy.exp(lnB)) / (1.0e0 - cb/c12 + cb/c12 * (numpy.exp(lnC) + numpy.exp(lnD)))
    return rho_ions

def _two_to_one(solvent_obj, phi_tot=None, cb=None, lambda_r=None, T=None):
    """Size-modified 2(cation):1(anion) ion charge density. z(cation) = +1, z(anion) = -2

    Args:
        solvent_obj (:class:`PBE`): Solvent object
        phi_tot (1D numpy.ndarray, optional): Total electrostatic potential. Defaults to None.
        cb (float, optional): Ion concentration in atomic unit. Defaults to None.
        lambda_r (1D numpy.ndarray, optional): Ion-exclusion function. Defaults to None.
        T (float, optional): Temperature. Defaults to None.

    Returns:
        1D numpy.ndarray: Ion charge density
    """
    if phi_tot is None: phi_tot = solvent_obj.phi_tot
    if cb is None: cb = solvent_obj.cb * M2HARTREE
    if lambda_r is None: lambda_r = solvent_obj._intermediates['lambda_r']
    if T is None: T = solvent_obj.T

    cation_rad = solvent_obj.cation_rad / BOHR
    anion_rad = solvent_obj.anion_rad / BOHR
    c12 = 0.74e0 / (4.0e0/3.0e0 * PI * (2*cation_rad**3 + anion_rad**3))
    _lambda_r = lambda_r.copy()
    _lambda_r[lambda_r < 1.0e-100] = 1.0e-100
    lnlambda = numpy.log(_lambda_r)
    lnlambda[lambda_r < 1.0e-100] = -1.0e100
    if cb == 0.0e0:
        return numpy.zeros_like(phi_tot)
    else:
        lncb = numpy.log(cb)

    x = phi_tot / (KB2HARTREE * T)
    lnA = numpy.log(2) + lnlambda + lncb - x
    lnB = numpy.log(2) + lnlambda + lncb + 2*x
    lnC = numpy.log(2) - numpy.log(3) + lnlambda - x
    lnD = -numpy.log(3) + lnlambda + 2*x
    rho_ions = (numpy.exp(lnA) - numpy.exp(lnB)) / (1.0e0 - cb/c12 + cb/c12*(numpy.exp(lnC) + numpy.exp(lnD)))
    return rho_ions

def _one_to_two(solvent_obj, phi_tot=None, cb=None, lambda_r=None, T=None):
    if phi_tot is None: phi_tot = solvent_obj.phi_tot
    if cb is None: cb = solvent_obj.cb * M2HARTREE
    if lambda_r is None: lambda_r = solvent_obj._intermediates['lambda_r']
    if T is None: T = solvent_obj.T

    cation_rad = solvent_obj.cation_rad / BOHR
    anion_rad = solvent_obj.anion_rad / BOHR
    c12 = 0.74e0 / (4.0e0/3.0e0 * PI * (cation_rad**3 + 2*anion_rad**3))
    _lambda_r = lambda_r.copy()
    _lambda_r[lambda_r < 1.0e-100] = 1.0e-100
    lnlambda = numpy.log(_lambda_r)
    lnlambda[lambda_r < 1.0e-100] = -1.0e100
    if cb == 0.0e0:
        return numpy.zeros_like(phi_tot)
    else:
        lncb = numpy.log(cb)

    x = phi_tot / (KB2HARTREE * T)
    lnA = numpy.log(2) + lnlambda + lncb - 2*x
    lnB = numpy.log(2) + lnlambda + lncb + x
    lnC = -numpy.log(3) + lnlambda - 2*x
    lnD = numpy.log(2) - numpy.log(3) + lnlambda + x
    rho_ions = (numpy.exp(lnA) - numpy.exp(lnB)) / (1.0e0 - cb/c12 + cb/c12*(numpy.exp(lnC) + numpy.exp(lnD)))
    return rho_ions

def one_to_one_energy_osm(solvent_obj, phi_tot, cb, lambda_r, T, spacing):
    _lambda_r = lambda_r.copy()
    _lambda_r[lambda_r < 1.0e-100] = 1.0e-100 # Machine precision
    lnlambda = numpy.log(_lambda_r)
    lnlambda[lambda_r < 1.0e-100] = -1.0e100
    x = phi_tot / KB2HARTREE / T
    lnA = numpy.log(0.5) + lnlambda - x
    lnB = numpy.log(0.5) + lnlambda + x
    cation_rad, anion_rad = solvent_obj.cation_rad / BOHR, solvent_obj.anion_rad / BOHR
    c12 = 0.74e0 / (4.0e0/3.0e0 * PI * (cation_rad**3 + anion_rad**3))
    Gsolv_osm = -2.0*KB2HARTREE*T*c12*(numpy.log(1.0e0 + cb/c12*(numpy.exp(lnA) + numpy.exp(lnB) - 1.0e0))).sum()*spacing**3
    return Gsolv_osm

def two_to_one_energy_osm(solvent_obj, phi_tot, cb, lambda_r, T, spacing):
    _lambda_r = lambda_r.copy()
    _lambda_r[lambda_r < 1.0e-100] = 1.0e-100 # Machine precision
    lnlambda = numpy.log(_lambda_r)
    lnlambda[lambda_r < 1.0e-100] = -1.0e100
    x = phi_tot / KB2HARTREE / T
    lnA = numpy.log(2) - numpy.log(3) + lnlambda - x
    lnB = -numpy.log(3) + lnlambda + 2*x
    cation_rad, anion_rad = solvent_obj.cation_rad / BOHR, solvent_obj.anion_rad / BOHR
    c12 = 0.74e0 / (4.0e0/3.0e0 * PI * (2*cation_rad**3 + anion_rad**3))
    Gsolv_osm = -3.0*KB2HARTREE*T*c12*(numpy.log(1.0e0 + cb/c12*(numpy.exp(lnA) + numpy.exp(lnB) - 1.0e0))).sum()*spacing**3
    return Gsolv_osm

def one_to_two_energy_osm(solvent_obj, phi_tot, cb, lambda_r, T, spacing):
    _lambda_r = lambda_r.copy()
    _lambda_r[lambda_r < 1.0e-100] = 1.0e-100 # Machine precision
    lnlambda = numpy.log(_lambda_r)
    lnlambda[lambda_r < 1.0e-100] = -1.0e100
    x = phi_tot / KB2HARTREE / T
    lnA = -numpy.log(3) + lnlambda - 2*x
    lnB = numpy.log(2) - numpy.log(3) + lnlambda + x
    cation_rad, anion_rad = solvent_obj.cation_rad / BOHR, solvent_obj.anion_rad / BOHR
    c12 = 0.74e0 / (4.0e0/3.0e0 * PI * (cation_rad**3 + 2*anion_rad**3))
    Gsolv_osm = -3.0*KB2HARTREE*T*c12*(numpy.log(1.0e0 + cb/c12*(numpy.exp(lnA) + numpy.exp(lnB) - 1.0e0))).sum()*spacing**3
    return Gsolv_osm

# TODO: OOP for ion charge density, osmotic pressure, and nuclear gradient
class Ions(lib.StreamObject):
    _keys = {'cb', 'cation_rad', 'anion_rad', 'equiv'}
    def __init__(self, cb, cation_rad, anion_rad, equiv):
        self.cb = cb
        self.cation_rad = cation_rad
        self.anion_rad = anion_rad
        self.equiv = equiv

    def build(self):
        pass