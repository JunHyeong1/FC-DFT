Examples
*********

This chapter introduces practical examples of running FC-DFT and Poisson-Boltzmann solvation calculations with the current implementation.

Wide-Band Limit (WBL) Approximation
------------------------------------

FC-DFT implements the wide-band limit (WBL) approximation through the WBL-Molecule approach, which allows DFT calculations on molecules coupled to electrode contacts. The WBL-Molecule requires you to specify the imaginary part of the self-energy (broadening parameter, Γ_L) and the Fermi level (reference potential).

The self-energy in WBL is defined as:

.. math::

   \Sigma_L = \Lambda_L - \frac{i}{2}\Gamma_L S

where Λ_L shifts the energy levels and Γ_L controls the level broadening (related to the ``broad`` parameter). The applied voltage is incorporated through a potential correction:

.. math::

   [\Delta H_L]_{\mu\nu} = \Delta V_b [S]_{\mu\nu}

where ΔV_b = -(μ - μ_ref) is the chemical potential difference from the reference electrode potential.

Currently, the spin-restricted version (``fcdft.wbl.rks``) is supported. The example below shows how to perform a WBL calculation on methanethiol with a sulfur-electrode contact:

    >>> from pyscf import gto
    >>> from pyscf.dft import RKS
    >>>
    >>> # Define molecular structure
    >>> mol = gto.M(atom='''
    ...     C    -1.718553971    -0.000000250    -0.626147715
    ...     H    -2.739245971    -0.008907250    -0.227127715
    ...     H    -1.200493971    -0.879491250    -0.227127715
    ...     H    -1.215921971     0.888398750    -0.227127715
    ...     S    -1.718553971    -0.000000250    -2.396147715
    ...     H    -2.150082583     0.805150681    -2.710667448''',
    ...     charge=0, basis='6-31g**')
    >>>
    >>> # Standard DFT calculation
    >>> mf = RKS(mol, xc='pbe')
    >>> mf.kernel()
    >>>
    >>> # WBL calculation
    >>> from fcdft.wbl.rks import WBLMoleculeRKS
    >>> wblmf = WBLMoleculeRKS(mol, xc='pbe', broad=0.01, nelectron=25.95)
    >>> wblmf.ref_pot = -4.5  # Reference potential (e.g., electrode Fermi level) in eV
    >>> wblmf.kernel()
    >>> dm = wblmf.make_rdm1()  # Density matrix at electrode contact

Poisson-Boltzmann Solvation Model
----------------------------------

The ``fcdft.solvent.pbe`` module provides a non-linear Poisson-Boltzmann (PB) solver for modeling solvation at electrode/liquid interfaces. Unlike standard PCM models, this implementation can handle complex boundary conditions including Stern layers and is designed for electrochemical systems.

The non-linear Poisson-Boltzmann equation is:

.. math::

   \nabla \cdot (\epsilon(r)\nabla\phi^{\text{tot}}(r)) = -4\pi[\rho^{\text{sol}}(r) + \rho^{\text{ions}}(r)]

where ε(r) is the position-dependent dielectric function, φ^tot(r) is the total electrostatic potential, ρ^sol(r) is the solute charge density, and ρ^ions(r) is the ion charge density. The dielectric function varies smoothly from the SAM (self-assembled monolayer) to the bulk solvent:

.. math::

   \epsilon(z) = \epsilon_{\text{SAM}} + \frac{1}{2}(\epsilon_{\text{bulk}} - \epsilon_{\text{SAM}})\left[1 + \text{erf}\left(\frac{z-a}{\Delta_1}\right)\right]

**Standard Solvation :**

For standard continuum solvation without electrode effects:

    >>> from pyscf import gto
    >>> from pyscf.dft import RKS
    >>> from fcdft.solvent.pbe import PBE, pbe_for_scf
    >>>
    >>> # Water molecule
    >>> mol = gto.M(atom='''
    ...     O    0.152427064    0.959723218   -2.275350162
    ...     H    0.152427064    1.719060218   -1.679307162
    ...     H    0.152427064    0.200386218   -1.679307162''',
    ...     charge=0, basis='6-31g**')
    >>>
    >>> # Standard DFT
    >>> mf = RKS(mol, xc='b3lyp')
    >>>
    >>> # Setup PBE solvation model
    >>> cm = PBE(mol, cb=1.0, length=15, ngrids=41)
    >>> cm.eps = 78.3553  # Dielectric constant of water
    >>> cm.nelectron = mol.nelectron
    >>> cm.atom_bottom = 'center'
    >>>
    >>> # Solvated DFT calculation
    >>> solmf = pbe_for_scf(mf, cm)
    >>> solmf.kernel()

**Electrochemical Solvation:**

For electrochemical systems with a Stern layer and self-assembled monolayer (SAM), combine WBL and PBE solvation. The boundary conditions follow the Gouy-Chapman-Stern theory.

    >>> from pyscf import gto
    >>> from pyscf.dft import RKS
    >>> from fcdft.wbl.rks import WBLMoleculeRKS
    >>> from fcdft.solvent.pbe import PBE, pbe_for_scf
    >>>
    >>> # Methanethiol molecule
    >>> mol = gto.M(atom='''
    ...     C    -1.718553971    -0.000000250    -0.626147715
    ...     H    -2.739245971    -0.008907250    -0.227127715
    ...     H    -1.200493971    -0.879491250    -0.227127715
    ...     H    -1.215921971     0.888398750    -0.227127715
    ...     S    -1.718553971    -0.000000250    -2.396147715
    ...     H    -2.150082583     0.805150681    -2.710667448''',
    ...     charge=0, basis='6-31g**')
    >>>
    >>> # Step 1: WBL calculation at fixed electrode potential
    >>> wblmf = WBLMoleculeRKS(mol, xc='pbe', broad=0.01, nelectron=25.95)
    >>> wblmf.ref_pot = -4.5  # Electrode Fermi level
    >>> wblmf.kernel()
    >>> dm = wblmf.make_rdm1()
    >>>
    >>> # Step 2: Add solvation with SAM and Stern layer
    >>> cm = PBE(mol, cb=1.0, length=15, ngrids=41, stern_sam=3.0)
    >>> cm.eps = 78.3553        # Water dielectric constant
    >>> cm.eps_sam = 2.284      # SAM dielectric constant
    >>> cm.nelectron = mol.nelectron
    >>> cm.bias = wblmf.bias    # Applied potential from WBL calculation
    >>>
    >>> # FC-DFT+ calculation
    >>> solmf = pbe_for_scf(wblmf, cm)
    >>> solmf.kernel(dm0=dm)


Geometry Optimization
----------------------

FC-DFT supports analytic gradients for both WBL and solvation models. Geometry optimization can be performed using external optimizers like GeomeTRIC:

    >>> from pyscf.geomopt.geometric_solver import optimize
    >>>
    >>> # Optimize structure at fixed electrode potential
    >>> moleq = optimize(solmf, maxstep=100)
    >>>
    >>> # Access optimized geometry and energies
    >>> print(moleq.atom_coords())
    >>> print(f"Final energy: {moleq.e_tot:.6f} Hartree")


Vibrational Analysis and Thermochemistry
------------------------------------------

FC-DFT provides numerical Hessian calculations based on analytic forces. This is important for non-Hermitian Hamiltonians (WBL) where analytical Hessian is not readily available. The Hessian is constructed using finite differences:

.. math::

   H_{ij} = \frac{1}{2\Delta x}\left[\frac{\partial^2 E}{\partial R_i \partial R_j}\right]

where the derivatives are obtained from analytic nuclear gradients.

    >>> from fcdft.hessian.numhess import Hessian
    >>> from pyscf.hessian.thermo import harmonic_analysis
    >>>
    >>> # Calculate Hessian using finite differences of analytic gradients
    >>> hessmf = Hessian(solmf)
    >>> hess = hessmf.kernel()
    >>>
    >>> # Analyze vibrational frequencies and thermochemistry
    >>> freq_info = harmonic_analysis(moleq, hess)
    >>>
    >>> # Save to Molden format for visualization
    >>> from fcdft.tools.molden import dump_freq
    >>> dump_freq(moleq, freq_info, 'freq.molden')