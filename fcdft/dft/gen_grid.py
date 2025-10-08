import numpy
from pyscf.dft import gen_grid
from pyscf.dft import radi
from pyscf.lib import logger
from fcdft.gto.mole import split_mol
import ipdb

# Grid for equilibrium DFT object.

def get_partition(mol, atom_slice, *args, **kwargs):
    mol1, mol2 = split_mol(mol, atom_slice)
    coords1, weights1 = gen_grid.get_partition(mol1, *args, **kwargs)
    coords2, weights2 = gen_grid.get_partition(mol2, *args, **kwargs)
    coords_all = [coords1, coords2]
    weights_all = [weights1, weights2]
    return coords_all, weights_all

class Grids(gen_grid.Grids):
    _keys = {'atom_slice'}

    def __init__(self, mol, atom_slice):
        gen_grid.Grids.__init__(self, mol)
        self.atom_slice = atom_slice

    def build(self, mol=None, with_non0tab=False, sort_grids=True, **kwargs):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        atom_grids_tab = self.gen_atomic_grids(
            mol, self.atom_grid, self.radi_method, self.level, self.prune, **kwargs)
        self.coords, self.weights = self.get_partition(
            mol, self.atom_slice, atom_grids_tab, self.radii_adjust, self.atomic_radii, self.becke_scheme)
        mol1, mol2 = split_mol(mol, self.atom_slice)
        mol = [mol1, mol2]
        atm_idx1 = numpy.empty(self.coords[0].shape[0], dtype=numpy.int32)
        atm_idx2 = numpy.empty(self.coords[1].shape[0], dtype=numpy.int32)
        atm_idx = [atm_idx1, atm_idx2]
        quad_weights1 = numpy.empty(self.coords[0].shape[0])
        quad_weights2 = numpy.empty(self.coords[1].shape[0])
        quadrature_weights = [quad_weights1, quad_weights2]
        for i in range(2):
            p0 = p1 = 0
            for ia in range(mol[i].natm):
                r, vol = atom_grids_tab[mol[i].atom_symbol(ia)]
                p0, p1 = p1, p1 + vol.size
                atm_idx[i][p0:p1] = ia
                quadrature_weights[i][p0:p1] = vol
        self.atm_idx = atm_idx
        self.quadrature_weights = quadrature_weights
        if sort_grids:
            for i in range(2):
                idx = gen_grid.arg_group_grids(mol[i], self.coords[i])
                self.coords[i] = self.coords[i][idx]
                self.weights[i] = self.weights[i][idx]
                self.atm_idx[i] = self.atm_idx[i][idx]
                self.quadrature_weights[i] = self.quadrature_weights[i][idx]
        # ??
        if self.alignment > 1:
            padding = gen_grid._padding_size(self.size, self.alignment)
            logger.debug(self, 'Padding %d grids', padding)
            if padding > 0:
                self.coords = numpy.vstack(
                    [self.coords, numpy.repeat([[1e-4]*3], padding, axis=0)])
                self.weights = numpy.hstack([self.weights, numpy.zeros(padding)])
                self.atm_idx = numpy.hstack([self.atm_idx, numpy.full(padding, -1, dtype=numpy.int32)])
                self.quadrature_weights = numpy.hstack([self.quadrature_weights, numpy.zeros(padding)])
        if with_non0tab:
            non0tab = []
            for i in range(2):
                non0tab.append(self.make_mask(mol[i], self.coords[i]))
            self.non0tab = non0tab
            self.screen_index = self.non0tab
        else:
            self.screen_index = self.non0tab = None
        logger.info(self, 'tot grids = %d', len(self.weights[0]) + len(self.weights[1]))
        return self
    
    def kernel(self, **kwargs):
        self.dump_flags()
        return self.build(**kwargs)

    def get_partition(self, mol, atom_slice, atom_grids_tab=None,
                      radii_adjust=None, atomic_radii=radi.BRAGG_RADII,
                      becke_scheme=gen_grid.original_becke, concat=True):
        if atom_grids_tab is None:
            atom_grids_tab = self.gen_atomic_grids(mol)
        return get_partition(mol, atom_slice, atom_grids_tab, radii_adjust, atomic_radii,
                             becke_scheme, concat=concat)
