from pyscf.gto.mole import Mole
from pyscf.lib import logger

# def split_env(mol, atm, bas, env, atom_id):
#     symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
#     sym1 = symbols[:atom_id]
#     from pyscf.gto.mole import make_bas_env
#     len_env = 0
#     for symb, basis_add in mol._basis.items():
#         if symb in sym1:
#             _, env0 = make_bas_env(basis_add, 0, 0)
#             len_env += len(env0)

#     env1, env2 = [numpy.zeros(PTR_ENV_START)], [numpy.zeros(PTR_ENV_START)]
#     env1.append(env[PTR_ENV_START:PTR_ENV_START+4*atom_id])
#     env2.append(env[PTR_ENV_START+4*atom_id:PTR_ENV_START+4*len(atm)])
#     env1, env2 = env[:len_env], env[len_env:]
#     idx = numpy.where(bas[:,0] == atom_id)[0][0]
#     bas1, bas2 = bas[:idx].copy(), bas[idx:].copy()
#     off = len(env1)
#     natm_off = atom_id
#     atm1, atm2 = atm[:atom_id].copy(), atm[atom_id:].copy()
#     atm2[:,PTR_COORD] -= off
#     atm2[:,PTR_ZETA ] -= off
#     bas2[:,ATOM_OF  ] -= natm_off
#     bas2[:,PTR_EXP  ] -= off
#     bas2[:,PTR_COEFF] -= off
#     return atm1, bas1, env1, atm2, bas2, env2

def conc_mol(mol1, mol2):
    from pyscf.gto import mole
    mol = mole.conc_mol(mol1, mol2)
    if mol1.basis == mol2.basis:
        mol.basis = mol1.basis
    return mol

def split_mol(mol, atom_id):
    '''Split the molecule into two Mole objects.
    Reverse of conc_mol in pyscf.gto.mole
    '''
    if not mol._built:
        logger.warn(mol, 'Warning: object %s not initialized. Initializing %s',
                    mol, mol)
        mol.build()

    # TODO: mol.stdout for logger
    mol1, mol2 = Mole(), Mole()
    mol1.basis = mol2.basis = mol.basis
    mol1.verbose = mol2.verbose = mol.verbose
    mol1.output = mol2.output = mol.output
    mol1.max_memory = mol2.max_memory = mol.max_memory
    mol1.spin = mol2.spin = 0
    mol1.symmetry = mol2.symmetry = False
    mol1.symmetry_subgroup = mol2.symmetry_subgroup = None
    mol1.cart = mol2.cart = mol.cart
    mol1._atom = mol._atom[:atom_id]
    mol2._atom = mol._atom[atom_id:]
    mol1.unit = mol2.unit = 'Bohr'
    mol1.build(), mol2.build()

    return mol1, mol2

if __name__=='__main__':
    from pyscf import gto
    mol1 = gto.M(
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
    mol2 = gto.M(
        atom='''
C       -0.8476079365    0.0026700202   -3.5281035617
C       -2.0218283577   -0.2604683627   -2.8073958219
C        0.3353012600    0.2642432684   -2.8231212050
H       -2.9520951875   -0.4549635668   -3.3338745954
H        1.2611570861    0.4600128780   -3.3568350109
C       -2.0073245682   -0.2604462093   -1.4139657047
C        0.3383704802    0.2619859179   -1.4285927349
H       -2.9349818456   -0.4420746761   -0.8791308006
H        1.2732326220    0.4430128273   -0.9060391242
C       -0.8293545128    0.0002996322   -0.6916543989
C       -0.8192397451   -0.0006201775    0.7937732451
C       -1.5616839183   -0.9482725602    1.5218647171
C       -0.0671411632    0.9462610490    1.5129186979
C       -1.5541603792   -0.9481485004    2.9180008413
C       -0.0558307025    0.9446630867    2.9090260355
C       -0.8001707945   -0.0022152441    3.6187748549
H       -2.1295491966   -1.7060626237    0.9896596517
H        0.4930453218    1.7050509252    0.9740007250
H       -2.1296311282   -1.6947234290    3.4583587491
H        0.5267024257    1.6907508988    3.4424644381
H       -0.7929373074   -0.0027907562    4.7049223527
S       -0.9353266992   -0.0162143726   -5.3112847893
H        0.3533301599    0.3018611373   -5.5386460312''',
charge=0, basis='6-31g**', verbose=5)
    mol = gto.conc_mol(mol1, mol2)
    _mol1, _mol2 = split_mol(mol, 14)