from pyscf.hessian import rhf as rhf_hess
import numpy as np
from pyscf.lib import logger

def hess_generator(mol, g_scanner, h, order):
    """
    Generate numerical Hessian matrix from finite differences of gradients.

    Computes the Hessian by displacing each coordinate ±Δ and evaluating
    the gradient at each displaced geometry, then differencing.

    Parameters
    ----------
    mol : gto.Mole
        Molecule specification (used for coordinate displacement).
    g_scanner : callable
        Gradient scanner function that returns (energy, gradient) for a given
        molecule geometry. Typically obtained from as_scanner(nuc_grad_method()).
    h : float, optional
        Finite difference step size in Bohr. If None, defaults to 5.0e-3 Bohr.
    order : int, optional
        Order of finite difference stencil: 3 (2-point) or 5 (4-point).
        If None, defaults to 3. 5th order is more accurate but requires more
        gradient evaluations (4 per coordinate vs. 2).

    Returns
    -------
    hess : ndarray, shape (n_atoms, n_atoms, 3, 3)
        Hessian matrix in PySCF convention: hess[i, j, α, β] = ∂²E/∂R_iα∂R_jβ.
        Units: Hartree/Bohr².

    Also saves: mol.output + ".hessian" file with flattened 2D Hessian.

    Raises
    ------
    ValueError
        If order is not 3 or 5.

    Notes
    -----
    The full Hessian is symmetrized after computation: (H + H^T) / 2.
    For WBL systems with fractional occupations, the numerical approach
    avoids analytical complications of non-Hermitian perturbation theory.
    """
    if h is None: delta = 5.0e-3
    else: delta = h
    if order is None: order = 3
    if order not in [3, 5]:
        raise ValueError("Only 3rd and 5th order finite difference are supported.")
    geom = mol.atom_coords()
    nat = geom.shape[0]
    ndim = 3 * nat
    H = np.zeros([ndim, ndim])

    counter = 0
    for iat in range(nat):
        for icoor in range(3):
            i = 3 * iat + icoor
            _geom = geom.copy()
            Hx = []
            if order == 3:
                for diff in [-1, 1]:
                    logger.info(g_scanner.base, 'Numerical hessian step %d/%d', counter+1, ndim*2)
                    _geom[iat, icoor] = geom[iat, icoor] + diff*delta
                    mol.set_geom_(_geom, unit="Bohr")
                    e, g = g_scanner(mol)
                    Hx.append(g)
                    counter += 1
                H[i, :] = (-Hx[0] + Hx[1]).reshape(-1)
            elif order == 5:
                for diff in [-2, -1, 1, 2]:
                    logger.info(g_scanner.base, 'Numerical hessian step %d/%d', counter+1, ndim*4)
                    _geom[iat, icoor] = geom[iat, icoor] + diff*delta
                    mol.set_geom_(_geom, unit="Bohr")
                    e, g = g_scanner(mol)
                    Hx.append(g)
                    counter += 1
                H[i, :] = (Hx[0] - 8*Hx[1] + 8*Hx[2] - Hx[3]).reshape(-1)

    mol.set_geom_(geom, unit="Bohr")

    if order == 3:
        H = (H + H.T) / (2*delta) / 2
    elif order == 5:
        H = (H + H.T) / (12*delta) / 2

    # Save the Hessian just in case
    np.save('%s.hessian' %mol.output, H)
    
    # Converting 2d hessian to 4d analogue to follow PySCF convention.
    hess = np.zeros((nat, nat, 3, 3))
    for atm1 in range(nat):
        for atm2 in range(nat):
            hess[atm1, atm2] = H[atm1*3:(atm1+1)*3, atm2*3:(atm2+1)*3]

    return hess


class Hessian(rhf_hess.HessianBase):
    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None, h=None, order=None):
        from pyscf.grad.rhf import as_scanner
        g_scanner = as_scanner(self.base.nuc_grad_method())
        self.de = hess_generator(self.mol, g_scanner, h, order)
        if self.base.disp is not None:
            self.de += self.get_dispersion()
        return self.de

    hess = kernel


if __name__ == '__main__':
    from pyscf import gto
    from pyscf.dft import RKS
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74')
    mf = RKS(mol, xc='pbe')
    mf.kernel()
    hess = Hessian(mf)
    hess.kernel()