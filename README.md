# Fractional Charge Density Functional Theory (FC-DFT)
This code computes single-point energies, optimized structures, vibrational frequencies, and solvation free energies of open quantum systems using fractional charge density functional theory (FC-DFT).

## Requirements
- [GeomeTRIC](https://github.com/leeping/geomeTRIC)
- [PySCF](https://pyscf.org/)
- [PyAMG](https://github.com/pyamg/pyamg)
- [PyAMGCL](https://github.com/ddemidov/amgcl)
  
## Installation
  - Clone the repository.
  - `cd FC-DFT`
  - `pip install .`
  - Change directory to `$PYTHONPATH/fcdft/lib` and create `build` directory.
  - Go into `build` and compile the C shared libraries by `cmake ..` and `make`.

## Features
  - FC-DFT calculations for all density functional approximations supported by LibXC library.
  - Wide-band limit calculations, namely WBL-Molecule.
  - Non-linear Poisson-Boltzmann solver in real space and its analytic nuclear gradients.
  - Numerical Hessian calculations.

## Notes
Poisson-Boltzmann geometry optimization uses RESP atomic charges, which are computed by the code provided by https://github.com/swillow/pyscf_esp after some modifications for computational efficiency.

## How to Cite
Please cite the paper below if this code was directly or indirectly helpful to your research.

Jun-Hyeong Kim, Dongju Kim, Weitao Yang, and Mu-Hyun Baik. Fractional Charge Density Functional Theory and Its Application to the Electro-inductive Effect. _J. Phys. Chem. Lett._ **2023**, _14_, 3329-3334

Jun-Hyeong Kim and Weitao Yang. Fractional Charge Density Functional Theory Elucidates Electro-Inductive and Electric Field Effects at Electrochemical Interfaces. _Submitted_ ChemRxiv: 10.26434/chemrxiv-2025-6qqfb

## Bug Report and Feature Request
Please open a thread on the [Issues](https://github.com/Yang-Laboratory/FC-DFT/issues) tab.