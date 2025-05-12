from pyscf import lib
from pyscf.dft import numint
import fcdft
import os
import numpy
import ctypes

libdft = numint.libdft
libfcdft = lib.load_library(os.path.join(fcdft.__path__[0], 'lib', 'libfcdft'))
OCCDROP = numint.OCCDROP
SWITCH_SIZE = numint.SWITCH_SIZE
MGGA_DENSITY_LAPL = numint.MGGA_DENSITY_LAPL

def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0,
             with_lapl=True, verbose=None):
    """Hooker to allow double * complex double _contract_rho"""
    xctype = xctype.upper()
    ngrids, nao = ao.shape[-2:]

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    if xctype == 'LDA' or xctype == 'HF':
        c0 = _dot_ao_dm(mol, ao, dm, non0tab, shls_slice, ao_loc)
        #:rho = numpy.einsum('pi,pi->p', ao, c0)
        rho = _contract_rho(ao, c0)
    elif xctype in ('GGA', 'NLC'):
        rho = numpy.empty((4,ngrids))
        c0 = _dot_ao_dm(mol, ao[0], dm, non0tab, shls_slice, ao_loc)
        #:rho[0] = numpy.einsum('pi,pi->p', c0, ao[0])
        rho[0] = _contract_rho(ao[0], c0)
        for i in range(1, 4):
            #:rho[i] = numpy.einsum('pi,pi->p', c0, ao[i])
            rho[i] = _contract_rho(ao[i], c0)
        if hermi:
            rho[1:4] *= 2  # *2 for + einsum('pi,ij,pj->p', ao[i], dm, ao[0])
        else:
            c1 = _dot_ao_dm(mol, ao[0], dm.conj().T, non0tab, shls_slice, ao_loc)
            for i in range(1, 4):
                rho[i] += _contract_rho(c1, ao[i])
    else: # meta-GGA
        if with_lapl:
            # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
            rho = numpy.empty((6,ngrids))
            tau_idx = 5
        else:
            rho = numpy.empty((5,ngrids))
            tau_idx = 4
        c0 = _dot_ao_dm(mol, ao[0], dm, non0tab, shls_slice, ao_loc)
        #:rho[0] = numpy.einsum('pi,pi->p', ao[0], c0)
        rho[0] = _contract_rho(ao[0], c0)

        rho[tau_idx] = 0
        for i in range(1, 4):
            c1 = _dot_ao_dm(mol, ao[i], dm, non0tab, shls_slice, ao_loc)
            #:rho[tau_idx] += numpy.einsum('pi,pi->p', c1, ao[i])
            rho[tau_idx] += _contract_rho(ao[i], c1)

            #:rho[i] = numpy.einsum('pi,pi->p', c0, ao[i])
            rho[i] = _contract_rho(ao[i], c0)
            if hermi:
                rho[i] *= 2
            else:
                rho[i] += _contract_rho(c1, ao[0])

        if with_lapl:
            if ao.shape[0] > 4:
                XX, YY, ZZ = 4, 7, 9
                ao2 = ao[XX] + ao[YY] + ao[ZZ]
                # \nabla^2 rho
                #:rho[4] = numpy.einsum('pi,pi->p', c0, ao2)
                rho[4] = _contract_rho(ao2, c0)
                rho[4] += rho[5]
                if hermi:
                    rho[4] *= 2
                else:
                    c2 = _dot_ao_dm(mol, ao2, dm, non0tab, shls_slice, ao_loc)
                    rho[4] += _contract_rho(ao[0], c2)
                    rho[4] += rho[5]
            elif MGGA_DENSITY_LAPL:
                raise ValueError('Not enough derivatives in ao')
        # tau = 1/2 (\nabla f)^2
        rho[tau_idx] *= .5
    return rho

def _contract_rho(bra, ket):
    '''Real part of rho for rho=einsum('pi,pi->p', bra.conj(), ket)'''
    bra = bra.T
    ket = ket.T
    nao, ngrids = bra.shape
    rho = numpy.empty(ngrids)

    if not (bra.flags.c_contiguous and ket.flags.c_contiguous):
        rho  = numpy.einsum('ip,ip->p', bra.real, ket.real)
        rho += numpy.einsum('ip,ip->p', bra.imag, ket.imag)
    elif bra.dtype == numpy.double and ket.dtype == numpy.double:
        libdft.VXC_dcontract_rho(rho.ctypes.data_as(ctypes.c_void_p),
                                 bra.ctypes.data_as(ctypes.c_void_p),
                                 ket.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_int(nao), ctypes.c_int(ngrids))
    elif bra.dtype == numpy.complex128 and ket.dtype == numpy.complex128:
        libdft.VXC_zcontract_rho(rho.ctypes.data_as(ctypes.c_void_p),
                                 bra.ctypes.data_as(ctypes.c_void_p),
                                 ket.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_int(nao), ctypes.c_int(ngrids))
    elif bra.dtype == numpy.double and ket.dtype == numpy.complex128:
        libfcdft.VXC_dzcontract_rho(rho.ctypes.data_as(ctypes.c_void_p),
                                    bra.ctypes.data_as(ctypes.c_void_p),
                                    ket.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(nao), ctypes.c_int(ngrids))
    else:
        rho  = numpy.einsum('ip,ip->p', bra.real, ket.real)
        rho += numpy.einsum('ip,ip->p', bra.imag, ket.imag)
    return rho

def _dot_ao_dm(mol, ao, dm, non0tab, shls_slice, ao_loc, out=None):
    '''return numpy.dot(ao, dm)'''
    if not (ao.dtype == numpy.double and dm.dtype == numpy.complex128):
        return numint._dot_ao_dm(mol, ao, dm, non0tab, shls_slice, ao_loc, out=out)
    ngrids, nao = ao.shape
    if (nao < SWITCH_SIZE or
        non0tab is None or shls_slice is None or ao_loc is None):
        return lib.dot(dm.T, ao.T).T

    if not ao.flags.f_contiguous:
        ao = lib.transpose(ao)
    fn = libdft.VXCdot_ao_dm

    vmre = numpy.ndarray((ngrids,dm.shape[1]), dtype=ao.dtype, order='F', buffer=out)
    vmim = numpy.ndarray((ngrids,dm.shape[1]), dtype=ao.dtype, order='F', buffer=out)
    dmre = numpy.asarray(dm.real, order='C')
    dmim = numpy.asarray(dm.imag, order='C')
    fn(vmre.ctypes.data_as(ctypes.c_void_p),
       ao.ctypes.data_as(ctypes.c_void_p),
       dmre.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nao), ctypes.c_int(dmre.shape[1]),
       ctypes.c_int(ngrids), ctypes.c_int(mol.nbas),
       non0tab.ctypes.data_as(ctypes.c_void_p),
       (ctypes.c_int*2)(*shls_slice),
       ao_loc.ctypes.data_as(ctypes.c_void_p))
    fn(vmim.ctypes.data_as(ctypes.c_void_p),
       ao.ctypes.data_as(ctypes.c_void_p),
       dmim.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nao), ctypes.c_int(dmim.shape[1]),
       ctypes.c_int(ngrids), ctypes.c_int(mol.nbas),
       non0tab.ctypes.data_as(ctypes.c_void_p),
       (ctypes.c_int*2)(*shls_slice),
       ao_loc.ctypes.data_as(ctypes.c_void_p))
    vm = vmre + vmim*1.0j
    return vm

class NumInt(numint.NumInt):
    """Bypassing warning messages"""
    def _gen_rho_evaluator(self, mol, dms, hermi=0, with_lapl=True, grids=None):
        if dms[0].dtype == numpy.double:
            return super()._gen_rho_evaluator(mol, dms, hermi, with_lapl, grids)
        else:
            if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
                dms = dms[numpy.newaxis]
            nao = dms[0].shape[0]
            ndms = len(dms)
            def make_rho(idm, ao, sindex, xctype):
                return eval_rho(mol, ao, dms[idm], sindex, xctype, hermi, with_lapl)
            return make_rho, ndms, nao