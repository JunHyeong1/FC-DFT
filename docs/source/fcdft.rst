fcdft package
==============

FC-DFT is a Python package for DFT calculations at electrochemical interfaces.
It combines the wide-band limit (WBL) approximation for electrode coupling with
continuum solvation models to study molecules at electrode/liquid interfaces under
applied potential.

Core Modules
============

**Wide-Band Limit (WBL)**: Couples molecules to electrode contacts via complex self-energy

- ``fcdft.wbl.rks`` — Restricted Kohn-Sham with WBL (RKS + electrode coupling)
- ``fcdft.wbl.uks`` — Unrestricted Kohn-Sham with WBL

**Solvation**: Non-linear Poisson-Boltzmann for electrode/liquid interfaces

- ``fcdft.solvent.pbe`` — Poisson-Boltzmann electrolyte model
- Stern layers and self-assembled monolayers (SAM) support
- Analytic gradients included

**Gradients**: Analytic nuclear forces for geometry optimization

- ``fcdft.grad.rks`` — Gradients for WBL (RKS)
- ``fcdft.grad.uks`` — Gradients for WBL (UKS)

**Hessian**: Numerical Hessians for vibrational analysis

- ``fcdft.hessian.numhess`` — Numerical second derivatives via finite differences
- ``fcdft.hessian.thermo`` — Thermochemistry analysis from Hessians

**Additional**:

- ``fcdft.lifcdft`` — Linear-interaction FC-DFT for charge analysis
- ``fcdft.jellium`` — Jellium electrode models
- ``fcdft.dft`` — DFT numerics (grids, numerical integration)
- ``fcdft.tools`` — Utilities (Molden export, etc.)

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   fcdft.dft
   fcdft.grad
   fcdft.hessian
   fcdft.jellium
   fcdft.lib
   fcdft.lifcdft
   fcdft.solvent
   fcdft.tools
   fcdft.wbl

Module Contents
---------------

.. automodule:: fcdft
   :members:
   :undoc-members:
   :show-inheritance:
