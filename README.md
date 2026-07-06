# Fractional Charge Density Functional Theory (FC-DFT)

**FC-DFT** is a Python package for FC-DFT calculations for predicting chemical events happening at electrochemical interfaces. It combines the wide-band limit (WBL) approximation for a model electrode with continuum solvation models to voltage-dependent shifts of chemical properties.

The code supports single-point energies, optimized structures, numerical vibrational frequencies, and solvation free energies calculations.

## Requirements
- [GeomeTRIC](https://github.com/leeping/geomeTRIC) (NOTE: FC-DFT has been tested with GeomeTRIC==1.0.2.)
- [PySCF](https://pyscf.org/)
- [PyAMG](https://github.com/pyamg/pyamg)
- [PyAMGCL](https://github.com/ddemidov/amgcl)
- [GPU4PySCF](https://github.com/pyscf/gpu4pyscf) (optional for GPU-accelerated PBE solver)
- [AMGX](https://github.com/NVIDIA/AMGX) (optional for GPU-accelerated PBE solver)
  
## Installation
  * Option 1: Install stable release via PyPI:

        pip install fcdft

  * Option 2: Clone the repository:

        git clone https://github.com/Yang-Laboratory/FC-DFT.git
        cd FC-DFT
        pip install .

  * (Deprecated) Once one of the options is completed, change directory to:
  
        $PYTHONPATH/fcdft/lib
        
  * (Deprecated) Create `build` directory and go into it:

        mkdir build && cd build

  * (Deprecated) Compile C libraries:

        cmake .. && make

### Tips for compilation

  * When linking Intel MKL, make sure to link the sequential one by `cmake -DBLA_VENDOR=Intel10_64lp_seq ..`. The performance of Intel MKL over OpenBLAS has not been tested. One may want to link OpenBLAS instead by `cmake -DBLA_VENDOR=OpenBLAS ..` for easier installation.
  * If CMake fails to find either `lapacke.h` or `cblas.h`, set `C_INCLUDE_PATH` manually.

## Features
  - FC-DFT calculations for all density functional approximations supported by PySCF.
  - Wide-band limit calculations, namely WBL-Molecule.
  - Non-linear Poisson-Boltzmann solver in real space and its analytic nuclear gradients.
  - Numerical Hessian calculations.

## Notes

Poisson-Boltzmann geometry optimization uses RESP atomic charges. These are computed by the code provided by https://github.com/swillow/pyscf_esp after some modifications for computational efficiency.

## How to Cite
Please cite the paper below if this code was directly or indirectly helpful to your research.

Jun-Hyeong Kim, Dongju Kim, Weitao Yang, and Mu-Hyun Baik. Fractional Charge Density Functional Theory and Its Application to the Electro-inductive Effect. _J. Phys. Chem. Lett._ **2023**, _14_, 3329-3334

Jun-Hyeong Kim and Weitao Yang. Fractional Charge Density Functional Theory Elucidates Electro-Inductive and Electric Field Effects at Electrochemical Interfaces. _Proc. Natl. Acad. Sci. U.S.A._ **2026**, _123_, e2602964123

## Bug Report and Feature Request

Please open a thread on the [Issues](https://github.com/Yang-Laboratory/FC-DFT/issues) tab.