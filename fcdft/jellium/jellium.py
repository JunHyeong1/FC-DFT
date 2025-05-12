from pyscf.data.nist import *
import numpy as np
import scipy
from scipy.integrate import solve_bvp

PI = np.pi
NREF = 1.0e0 / (BOHR * 1.0e-10)**3 # in /m^3
EPS0 = 8.8541878188e-12 # F/m = C/m/V = C^2 s^2 / kg / m^3 , J = kg m^2 / s^2
XI = HARTREE2J

def gamma(xe, ne):
    """Orbital-Free Local Density Approximation

    Args:
        xe (1D np.array): Input electron density
        ne (1D np.array): Reference electron density

    Returns:
        1D np.array: Electronic energy density referenced to the bulk
    """
    global jellium_obj
    eps_opt = jellium_obj.eps_opt
    ck = 2.87e0
    ce = -0.74e0
    cc = -0.056e0
    return 5.0e0 / 3.0e0 * ck * np.cbrt(xe**2) + 4.0e0 / 3.0e0 * ce / eps_opt * np.cbrt(xe) + cc / eps_opt * (eps_opt * np.cbrt(xe**2) + 4.0e0/3.0e0*0.079e0 * np.cbrt(xe)) / (0.079e0 + eps_opt * np.cbrt(ne))**2

def init_guess(jellium_obj, UM=None, n0=None):
    """Generate the initial guess for the collocation method.

    Args:
        jellium_obj (_type_): Jellium object
        UM (float, optional): Bulk electrostatic potential. Defaults to None.
        n0 (float, optional): Jellium background charge. Defaults to None.

    Returns:
       1D np.array, 1D np.array, 1D np.array, 1D np.array: Electrostatic potential, gradient of the electrostatic potential, electron density, and gradient of the electron density.
    """
    if UM is None: UM = jellium_obj.UM
    if n0 is None: n0 = jellium_obj.n0
    coords = jellium_obj.coords
    D = jellium_obj.D

    beta = np.sqrt(n0 / abs(UM))
    
    _exp = np.zeros_like(coords)
    idx = np.where(coords >= 0.0e0)
    _exp[idx] = np.exp(-beta*coords[idx])
    U = UM*np.heaviside(-(coords+D), 0.0e0) + (UM - 30 + 30*np.sin(PI/2.0e0 + PI*(coords+D)/D)) * np.heaviside(coords+D, 1.0e0) * np.heaviside(-coords, 0.0e0) + (60-UM)*((1.0e0 - _exp)**2 - 1.0e0) * np.heaviside(coords, 1.0e0)
    dUdX = 30*PI/D*np.cos(PI/2.0e0 + PI*((coords+D)/D)) * np.heaviside(coords+D, 1.0e0) * np.heaviside(-coords, 0.0e0) + (60-UM)*2.0e0*(1.0e0 - _exp)*(beta*_exp)* np.heaviside(coords, 1.0e0)
    ne = n0 * np.heaviside(-coords, 0.0e0) + n0 * np.heaviside(coords, 1.0e0) * (1.0e0 - (1.0e0 - _exp)**2)
    dnedX = -n0 * (2.0e0 * (1.0e0 - _exp)) * beta * _exp * np.heaviside(coords, 1.0e0)
    return U, dUdX, ne, dnedX

def eps_s(X, dUdX):
    """Calculate the field-dependent dielectric function and its gradient.

    Args:
        X (1D np.array): Grids
        dUdX (1D np.array): Negative of the electric field.

    Returns:
        1D np.array, 1D np.array: Dielectric function and its gradient.
    """
    global jellium_obj
    P = jellium_obj.P
    D = jellium_obj.D
    delta2 = jellium_obj.delta2
    eps_opt = jellium_obj.eps_opt
    ns = jellium_obj.ns

    t = P * dUdX
    idx1 = np.where(abs(t) < 1.0e-3)
    idx2 = np.logical_and(abs(t) >= 1.0e-3, abs(t) < 690)
    idx3 = np.where(abs(t) >= 690)

    xLx = np.zeros_like(X)
    xLx[idx1] = 1.0e0 / 3.0e0
    xLx[idx2] = (1.0e0 / np.tanh(t[idx2])) / (t[idx2]) - 1.0e0 / (t[idx2])**2
    xLx[idx3] = (1.0e0 / np.tanh(t[idx3])) / (t[idx3]) - 1.0e0 / (t[idx3])**2

    eps = 1.0e0 + 0.5e0 * (1.0e0 + scipy.special.erf((X-D)/delta2)) * ((eps_opt - 1.0e0) + P**2 * ns * xLx)

    d2UdX2 = np.gradient(dUdX) / (X[1] - X[0])
    sinh = np.zeros_like(X)
    sinh[idx2] = np.sinh(t[idx2])
    sinh[idx3] = np.sign(t[idx3]) * 1.0e300
    grad_xLx = np.zeros_like(X)

    grad_xLx[idx1] = 0.0e0
    grad_xLx[idx2] = 2.0e0 / t[idx2]**3 - 1.0e0 / t[idx2] * (1.0e0 / sinh[idx2])**2 - 1.0e0 / t[idx2]**2 / np.tanh(t[idx2])
    grad_xLx[idx3] = 2.0e0 / t[idx3]**3 - 1.0e0 / t[idx3] * (1.0e0 / sinh[idx3])**2 - 1.0e0 / t[idx3]**2 / np.tanh(t[idx3])

    grad_eps = 0.5e0 * (1.0e0 + scipy.special.erf((X-D)/delta2)) * P**3 * ns * d2UdX2 * grad_xLx

    return eps, grad_eps

def fun(x, y):
    """Control equation of the Jellium+PBE model

    Args:
        x (1D np.array): Grids
        y (2D np.array): Input values

    Returns:
        2D np.array: dUdX, d2UdX2, dnedX, d2nedX2
    """
    U, dUdX, ne, dnedX = y[0], y[1], y[2], y[3]
    eps, grad_eps = eps_s(x, dUdX)

    global jellium_obj
    UM = jellium_obj.UM
    n0 = jellium_obj.n0
    nc0 = jellium_obj.nc0
    na0 = jellium_obj.na0
    chi = jellium_obj.chi
    delta1 = jellium_obj.delta1
    delta2 = jellium_obj.delta2
    D = jellium_obj.D
    lamb = jellium_obj.lamb
    zeta = jellium_obj.zeta
    T = jellium_obj.T

    ns = jellium_obj.ns
    Ws = jellium_obj.Ws

    idx = np.where(abs(U) < 690)
    if nc0 == 0.0e0 and na0 == 0.0e0:
        rho_ions = np.zeros_like(U)
    else:
        rho_ions = nc0*np.ones_like(U) * (-2.0e0 / chi) * np.sign(U)
        rho_ions[idx] = -2.0e0 * nc0* np.sinh(U[idx]) / (1.0e0 - chi + chi*np.cosh(U[idx]))

    rho_tot = n0 * 0.5e0 * (1.0e0 - scipy.special.erf(x/delta1)) - ne + rho_ions * 0.5e0 * (1.0e0 + scipy.special.erf((x-D)/delta2))
    rho_pol = grad_eps * dUdX
    _rho = rho_tot + rho_pol

    cg = 0.014e0

    ne_idx = np.where(abs(ne) > 1.0e-5)
    ne_tot = lamb**2 / 2.0e0 / cg / (BOHR*1.0e-10)**2 * (ne * (gamma(ne, ne) - gamma(n0, ne)) - zeta*ne*(U-UM) + zeta*ne*0.5e0 * (1.0e0 + scipy.special.erf((x-D)/delta2))/(BOLTZMANN*T*n0) * (ns*Ws * E_CHARGE))
    ne_tot[ne_idx] += 0.5e0 / ne[ne_idx] * dnedX[ne_idx]**2

    return np.vstack([dUdX, -_rho/eps, dnedX, ne_tot])

def bc(ya, yb):
    """Dirichlet boundary condition.

    Args:
        ya (_type_): _description_
        yb (_type_): _description_

    Returns:
        _type_: _description_
    """
    global jellium_obj
    UM = jellium_obj.UM
    n0 = jellium_obj.n0

    return np.array([ya[0]-UM, yb[0], ya[2]-n0, yb[2]])

def kernel():
    global jellium_obj
    ngrids = jellium_obj.ngrids

    U = jellium_obj.U
    dUdX = jellium_obj.dUdX
    ne = jellium_obj.ne
    dnedX = jellium_obj.dnedX
    coords = jellium_obj.coords

    Y = np.zeros((4, ngrids), dtype=np.float64)
    Y[0], Y[1], Y[2], Y[3] = U, dUdX, ne, dnedX
    sol = solve_bvp(fun, bc, coords, Y, tol=1.0e-6, max_nodes=30000, verbose=2)

    jellium_obj.U = sol.sol(coords)[0]
    jellium_obj.dUdX = sol.sol(coords)[1]
    jellium_obj.ne = sol.sol(coords)[2]
    jellium_obj.dnedX = sol.sol(coords)[3]
    return sol

class Jellium:
    def __init__(self, UM=30, rs=3.01, solvent=55.6, cation=1.0, anion=1.0, Ws=5.0, Wc=0.0, Wa=0.0, stern=1.0):
        self.UM = UM
        self.rs = rs
        self.cation = cation
        self.anion = anion
        self.solvnet = solvent
        self.solvent = 55.6
        self.delta1 = 1.0
        self.delta2 = 1.0
        self.eps_opt = 1.8
        self.p = 1.58e-29 # C m, Electrochimica Acta 2021, 389, 138720
        self.lattice = 3.34e28
        self.Ws = Ws
        self.Wc = Wc
        self.Wa = Wa
        self.stern = stern * 1.0e-10
        self.T = 298.15
        self.zmin = -5.0
        self.zmax = 20.0
        self.ngrids = 1000

        self.n0 = None
        self.lamb = None
        self.P = None
        self.D = None
        self.ns = None
        self.nc0 = None
        self.na0 = None
        self.nl = None
        self.chi = None
        self.U = None
        self.dUdX = None
        self.ne = None
        self.dnedX = None
        self.coords = None
        self.zeta = None
        self.sol = None

    def build(self):
        self.n0 = 1.0e0 / (4.0e0 / 3.0e0 * PI * self.rs**3)
        lamb = np.sqrt(EPS0 * BOLTZMANN * self.T / E_CHARGE**2 / NREF)
        self.lamb = lamb
        self.P = self.p / E_CHARGE / lamb
        self.D = self.stern / lamb
        self.ns = self.solvent * 1000 * AVOGADRO / NREF
        self.nc0 = self.cation * 1000 * AVOGADRO / NREF
        self.na0 = self.anion * 1000 * AVOGADRO / NREF
        self.nl = self.lattice / NREF
        self.chi = (self.nc0 + self.na0) / self.nl
        self.zeta = BOLTZMANN * self.T / XI
        self.coords = np.linspace(self.zmin, self.zmax, self.ngrids, dtype=np.float64) * 1.0e-10 / lamb
        self.U, self.dUdX, self.ne, self.dnedX = self.init_guess()

        global jellium_obj
        jellium_obj = self

    def kernel(self):
        if self.coords is None:
            self.build()
        sol = kernel()
        self.sol = sol
        return sol
    
    init_guess = init_guess

if __name__=='__main__':
    mf = Jellium()
    mf.kernel()