fcdft.solvent package
======================

Poisson-Boltzmann solvation model for electrode/liquid interfaces.

This package implements continuum solvation for molecules at electrode surfaces using
a non-linear Poisson-Boltzmann (PB) approach. Key features:

- **Position-dependent dielectric**: Smooth transition from SAM through Stern layers to bulk
- **Ion concentration**: From Boltzmann distribution at the applied potential
- **Gouy-Chapman-Stern boundary conditions**: Electrode surface treatment
- **Analytic gradients**: For geometry optimization
- **GPU acceleration**: Optional gpu4pyscf support for large grids

The non-linear PB equation solved is: ∇·(ε(r) ∇φ_tot) = -4π[ρ_sol + ρ_ions(φ_tot, T)]

Submodules
----------

fcdft.solvent.calculus\_helper module
-------------------------------------

.. automodule:: fcdft.solvent.calculus_helper
   :members:
   :undoc-members:
   :show-inheritance:

fcdft.solvent.esp module
------------------------

.. automodule:: fcdft.solvent.esp
   :members:
   :undoc-members:
   :show-inheritance:

fcdft.solvent.pbe module
------------------------

.. automodule:: fcdft.solvent.pbe
   :members:
   :undoc-members:
   :show-inheritance:

fcdft.solvent.pbe\_grad module
------------------------------

.. automodule:: fcdft.solvent.pbe_grad
   :members:
   :undoc-members:
   :show-inheritance:

fcdft.solvent.radii module
--------------------------

.. automodule:: fcdft.solvent.radii
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: fcdft.solvent
   :members:
   :undoc-members:
   :show-inheritance:
