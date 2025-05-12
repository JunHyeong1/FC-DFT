Examples
********

This chapter introduces brief exmaples of running FC-DFT and Poisson-Boltzmann solvation calculations

Wide-Band Limit
----------------

FC-DFT employs a special type of the wide-band limit (WBL) approximation, where the self-energy is constructed under the absence of electrodes. This requires users to provide the imaginary part of the self-energy in ``WBLMolecule`` object.
Currently, spin-restricted version of FC-DFT (``fcdft.wbl.rks``) is supported. Below is a sample code, where the self-energy of 0.01 eV is attached to the sulfur atom of methanethiol with 25.95 electrons.

    >>> from pyscf import gto
    >>> from pyscf.dft import RKS
    >>> mol = gto.M(atom='''
                C       -1.718553971     -0.000000250     -0.626147715
                H       -2.739245971     -0.008907250     -0.227127715
                H       -1.200493971     -0.879491250     -0.227127715
                H       -1.215921971      0.888398750     -0.227127715
                S       -1.718553971     -0.000000250     -2.396147715
                H       -2.150082583      0.805150681     -2.710667448''',
            charge=0, basis='6-31g**')
    >>> mf = RKS(mol, xc='pbe')
    >>> mf.kernel()
    >>> from fcdft.wbl.rks import *
    >>> wblmf = WBLMolecule(mf, broad=0.01, nelectron=25.95)
    >>> wblmf.kernel()

Non-Linear Poisson-Boltzmann Solvation Model
---------------------------------------------
We provide the Poisson-Boltzmann solver for general purpose. ``fcdft.solvent.pbe`` module supports usual solvation energy calculations as what polarizable continuum model does.
To do so, a few attributes of ``PBE`` needs to be controlled since it was originally intended to solve the electrostatic potential under the Gouy-Chapman-Stern theory:

    >>> from pyscf import gto
    >>> from pyscf.dft import RKS
    >>> mol = gto.M(atom='''
                O        0.152427064      0.959723218     -2.275350162
                H        0.152427064      1.719060218     -1.679307162
                H        0.152427064      0.200386218     -1.679307162''',
            charge=0, basis='6-31g**')
    >>> mf = RKS(mol, xc='b3lyp')
    >>> from fcdft.solvent.pbe import *
    >>> cm = PBE(mol, cb=1.0, length=15, ngrids=41)
    >>> cm.eps = 78.3553
    >>> cm.atom_bottom = 'center'
    >>> cm.nelectron = mol.nelectron
    >>> cm.bias = 0.0e0
    >>> cm.surf = 0.0e0
    >>> solmf = pbe_for_scf(mf, cm)
    >>> solmf.kernel()

The following example introduces how to run the Poisson-Boltzmann solver under the Gouy-Chapman-Stern boundary values:

    >>> from pyscf import gto
    >>> from pyscf.dft import RKS
    >>> mol = gto.M(atom='''
                C       -1.718553971     -0.000000250     -0.626147715
                H       -2.739245971     -0.008907250     -0.227127715
                H       -1.200493971     -0.879491250     -0.227127715
                H       -1.215921971      0.888398750     -0.227127715
                S       -1.718553971     -0.000000250     -2.396147715
                H       -2.150082583      0.805150681     -2.710667448''',
            charge=0, basis='6-31g**')
    >>> mf = RKS(mol, xc='pbe')
    >>> mf.kernel()
    >>> from fcdft.wbl.rks import *
    >>> wblmf = WBLMolecule(mf, broad=0.01, nelectron=25.95)
    >>> wblmf.kernel()
    >>> dm = wblmf.make_rdm1()
    >>> from fcdft.solvent.pbe import *
    >>> cm = PBE(mol, cb=1.0, length=15, ngrids=41, stern_sam=3.0)
    >>> cm.eps = 78.3553
    >>> cm.eps_sam = 2.284
    >>> cm._dm = dm
    >>> solmf = pbe_for_scf(wblmf, cm)
    >>> solmf.kernel()


Geometry Optimization
----------------------
Our code supports analytic nuclear gradients of FC-DFT as well as the Poisson-Boltzmann solvation model. We have tested geometry optimization using GeomeTRIC, an external geometry optimizer implemented in PySCF:

    >>> from pyscf.geomopt.geometric_solver import optimize
    >>> moleq = optimize(solmf, maxstep=100)


Thermochemistry
----------------
We provide a code for numerical Hessian matrix constructed by analytic forces due to the non-Hermitian Hamiltonian resulted by the self-energy. Thermochemical properties can be calculated by utilizing ``pyscf.hessian.thermo`` module.
``Hessian`` offers three-point (default) and five-point finite difference method for calculating the Hessian matrix. The following code introduces how to obtain thermochemical properties using ``harmonic_analysis`` function:

    >>> from fcdft.hessian numhess import *
    >>> hessmf = Hessian(solmf)
    >>> hess = hessmf.kernel()
    >>> from pyscf.hessian.thermo import harmonic_analysis
    >>> freq_info = harmonic_analysis(moleq, hess)

Once the quantities are obtained, these can be saved into a molden format as implemented in ``fcdft.tools.molden``:

    >>> from fcdft.tools.molden import dump_freq
    >>> dump_freq(moleq, freq_info, 'freq.molden')