from functools import reduce
import numpy
from pyscf.lib import logger
from pyscf.data.nist import *

def frac_occ_(mf, nelec):
    """Direct deposition of fractional electrons.

    Args:
        mf (pyscf.dft.RKS): Restricted KS-DFT object.
        nelec (float): Total number of electrons.

    Raises:
        AttributeError: Only RKS supported.

    Returns:
        mf: Updated RKS object allowing fractional electrons.
    """
    if mf.istype('RKS'):
        mol = mf.mol
        nelec_old = mol.nelectron
        def get_occ(mo_energy, mo_coeff=None):
            # nocc = int((nelec+1) / 2)
            nocc = int(nelec / 2)
            frac_nelec = nelec - nelec_old
            mo_occ = numpy.zeros_like(mo_energy)
            mo_occ[:nocc] = 2.0e0
            if frac_nelec < 0:
                mo_occ[nocc-1] += frac_nelec
            else:
                mo_occ[nocc] += frac_nelec
            logger.info(mf, 'mo_occ = \n%s', mo_occ)
            return mo_occ
    else:
        raise AttributeError('Only spin restricted KS-DFT supported.')
    
    mol.nelectron = nelec
    mf.mol = mol
    mf.get_occ = get_occ
    return mf

frac_occ = frac_occ_

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(
        atom='''
C       -1.1367537947      0.1104289172      2.4844663896
C       -1.1385831318      0.1723328088      3.8772156394
C        0.0819843127      0.0788096973      1.7730802291
H       -2.0846565855      0.1966185690      4.4236084687
C        0.0806058727      0.2041086872      4.5921211233
C        1.2993389981      0.1104289172      2.4844663896
H        2.2526138470      0.0865980845      1.9483127672
C        1.2994126658      0.1723829840      3.8783367991
H        2.2453411518      0.1966879024      4.4251589385
H       -2.0869454458      0.0863720324      1.9432143952
C        0.0810980584      0.2676328718      6.0213144069
N        0.0819851974      0.3199013851      7.1972568519
S        0.0000000000      0.0000000000      0.0000000000
H        1.3390319419     -0.0095801980     -0.2157234144''',
        charge=1, basis='6-31g**', verbose=5, spin=1)
    from fcdft.wbl.uks import WBLMoleculeUKS
    wblmf = WBLMoleculeUKS(mol, xc='B3LYP', broad=0.01, smear=0.2, nelectron=69.00)
    wblmf.kernel()
    import ipdb
    ipdb.set_trace()

    