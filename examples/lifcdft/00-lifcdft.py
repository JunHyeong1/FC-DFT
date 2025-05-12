from pyscf import gto
from fcdft.lifcdft.lifcdft import LIFCDFT

'''
Linear Interpolation FC-DFT
"delta" variable should be a positive number.
For example, 69.99 electrons should be specified as 69 (charge=1, spin=1) + 0.99, not 70 (charge=0, spin=0) - 0.01.
converged SCF energy = -722.083089304217
'''

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
        charge=0, basis='6-31g**', verbose=5)
mf = LIFCDFT(mol, xc='pbe', delta=0.1)
mf.kernel()