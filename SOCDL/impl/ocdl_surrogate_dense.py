#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implementation of second order surrogate-splitting based
online convolutional dictionary learning."""
import copy
import logging
from future.utils import with_metaclass
import numpy as np
from scipy import linalg
import sporco.cnvrep as cr
import sporco.util as su
import sporco.linalg as sl
from sporco.util import u
from sporco import common, cdict
from sporco.dictlrn import dictlrn
from sporco.admm import cbpdn
from sporco.fista import fista
try:
    import sporco_cuda.cbpdn as cucbpdn
except ImportError:
    cucbpdn = None

from ..utils import Pcn, einsum

logger = logging.getLogger(__name__)


class SpatialFISTA(fista.FISTA):
    """FISTA update for spatial scheme."""

    class Options(fista.FISTA.Options):
        defaults = copy.deepcopy(fista.FISTA.Options.defaults)
        defaults.update({
            'ZeroMean': True
        })

        def __init__(self, opt=None):
            if opt is None:
                opt = {}
            super().__init__(opt)

    itstat_fields_objfn = ('Cnstr', )
    hdrtxt_objfn = ('Cnstr', )
    hdrval_objfun = {'Cnstr': 'Cnstr'}

    def __init__(self, At, Bt, dsz=None, opt=None):
        """We use the following shape formulation:
        ----------------------
        D  | (Hc, Wc, C, 1, M)
        At | (Hc*Wc*M, Hc*Wc*M)
        Bt | (Hc*Wc*M, C)
        ----------------------
        """
        if opt is None:
            opt = SpatialFISTA.Options()
        assert isinstance(opt, SpatialFISTA.Options)
        if opt['X0'] is not None:
            self.dsz = opt['X0'].shape
        else:
            self.dsz = dsz
        super().__init__(np.prod(dsz), dsz, At.dtype, opt)
        self.At = At
        self.Bt = Bt
        self.Y = self.X.copy()

    def Pcn(self, D):
        return Pcn(D, self.opt['ZeroMean'])

    def eval_grad(self):
        r"""Evaluate the gradient:

            .. ::math:
                \nabla_D f = At * D - Bt
        """
        #  if 1:
        #      grad = einsum('ijklmno,klpqn->ijpqo', (self.At, self.Y))
        #  else:
        #      # [Hc, Wc, Hc, Wc, 1, M, M] -> [Hc*Wc*M, Hc*Wc*M]
        #      A = np.moveaxis(self.At, -1, 2)
        #      A = A.reshape(np.prod(A.shape[:3]), -1)
        #      # [Hc, Wc, C, 1, M] -> [Hc*Wc*M, C]
        #      D = np.moveaxis(self.Y, -1, 2)
        #      orig_shp = D.shape
        #      D = D.reshape(np.prod(D.shape[:3]), -1)
        #      grad = A.dot(D)
        #      grad = grad.reshape(orig_shp)
        #      grad = np.moveaxis(grad, 2, -1)
        D = np.moveaxis(self.Y, -1, 2)
        orig_shp = D.shape
        D = D.reshape(-1, D.shape[-2])  # the last dim is C
        grad = self.At.dot(D)
        grad -= self.Bt
        grad = grad.reshape(orig_shp)
        grad = np.moveaxis(grad, 2, -1)
        return grad

    def eval_proxop(self, V):
        return self.Pcn(V)

    def rsdl(self):
        """Fixed point residual."""
        return linalg.norm(self.X - self.Yprv)

    def eval_objfn(self):
        """Eval constraint only."""
        cnstr = linalg.norm(self.X - self.Pcn(self.X))
        return (cnstr, )


class OnlineDictLearnDenseSurrogate(
        with_metaclass(dictlrn._DictLearn_Meta, common.BasicIterativeSolver)
):
    """Implementation of online convolutional dictionary learning method
    with dense surrogate splitting."""

    class Options(cdict.ConstrainedDict):
        defaults = {
            'Verbose': True, 'StatusHeader': True, 'IterTimer': 'solve',
            'MaxMainIter': 1000, 'Callback': None, 'AccurateDFid': False,
            'DataType': None,
            'CBPDN': copy.deepcopy(cbpdn.ConvBPDN.Options.defaults),
            'CCMOD': copy.deepcopy(SpatialFISTA.Options.defaults),
            'OCDL': {
                'p': 1.,  # forgetting exponent
                'DiminishingTol': False,  # diminishing tolerance for FISTA
                'CUCBPDN': False,  # whether to use CUDA version of CBPDN
                'PARCBPDN': False,
                'nproc': None,
            }
        }
        defaults['CBPDN'].update({
            'Verbose': False, 'MaxMainIter': 100, 'AuxVarObj': False,
            'RelStopTol': 1e-7, 'DataType': None, 'FastSolve': False,
        })
        defaults['CBPDN']['AutoRho'].update({'Enabled': False})
        defaults['CCMOD']['BackTrack'].update({'Enabled': False})

        def __init__(self, opt=None):
            super().__init__({
                'CBPDN': cbpdn.ConvBPDN.Options(self.defaults['CBPDN']),
                'CCMOD': SpatialFISTA.Options(self.defaults['CCMOD']),
            })
            if opt is None:
                opt = {}
            self.update(opt)

    def __new__(cls, *args, **kwargs):
        instance = super(OnlineDictLearnDenseSurrogate, cls).__new__(cls)
        instance.timer = su.Timer(['init', 'solve', 'solve_wo_eval', 'hessian',
                                   'xstep', 'dstep'])
        instance.timer.start('init')
        return instance

    def __init__(self, D0, S0, lmbda=None, opt=None, dimK=None, dimN=2):
        if opt is None:
            opt = OnlineDictLearnDenseSurrogate.Options()
        assert isinstance(opt, OnlineDictLearnDenseSurrogate.Options)
        self.opt = opt

        self.set_dtype(opt, S0.dtype)

        self.cri = cr.CSC_ConvRepIndexing(D0, S0, dimK=dimK, dimN=dimN)

        self.isc = self.config_itstats()
        self.itstat = []
        self.j = 0

        self.set_attr('lmbda', lmbda, dtype=self.dtype)

        D0 = Pcn(D0, opt['CCMOD', 'ZeroMean'])

        self.D = np.asarray(D0.reshape(self.cri.shpD), dtype=self.dtype)
        self.S0 = np.asarray(S0.reshape(self.cri.shpS), dtype=self.dtype)
        self.At = self.dtype.type(0.)
        self.Bt = self.dtype.type(0.)

        self.lmbda = self.dtype.type(lmbda)
        self.Lmbda = self.dtype.type(0.)
        self.p = self.dtype.type(self.opt['OCDL', 'p'])

        if self.opt['Verbose'] and self.opt['StatusHeader']:
            self.isc.printheader()

    def solve(self, S):
        self.cri = cr.CSC_ConvRepIndexing(self.getdict(), S,
                                          dimK=self.cri.dimK,
                                          dimN=self.cri.dimN)

        self.timer.start(['solve', 'solve_wo_eval'])

        # Initialize with CBPDN
        self.timer.start('xstep')
        copt = copy.deepcopy(self.opt['CBPDN'])
        if self.opt['OCDL', 'CUCBPDN']:
            X = cucbpdn.cbpdn(self.getdict(), S.squeeze(), self.lmbda, opt=copt)
            X = np.asarray(X.reshape(self.cri.shpX), dtype=self.dtype)
        elif self.opt['OCDL', 'PARCBPDN']:
            popt = parcbpdn.ParConvBPDN.Options(dict(self.opt['CBPDN']))
            xstep = parcbpdn.ParConvBPDN(self.getdict(), S, self.lmbda,
                                         opt=popt,
                                         nproc=self.opt['OCDL', 'nproc'])
            X = xstep.solve()
            X = np.asarray(X.reshape(self.cri.shpX), dtype=self.dtype)
        else:
            xstep = cbpdn.ConvBPDN(self.getdict(), S, self.lmbda, opt=copt)
            xstep.solve()
            X = np.asarray(xstep.getcoef().reshape(self.cri.shpX),
                           dtype=self.dtype)
        self.timer.stop('xstep')

        # X = np.asarray(xstep.getcoef().reshape(self.cri.shpX), dtype=self.dtype)
        S = np.asarray(S.reshape(self.cri.shpS), dtype=self.dtype)

        # update At and Bt
        # (H, W, 1, K, M) -> (H, W, Hc, Wc, 1, K, M)
        self.timer.start('hessian')
        Xe = self.extend_code(X)
        self.update_At(Xe)
        self.update_Bt(Xe, S)
        self.timer.stop('hessian')
        self.Lmbda = self.dtype.type(self.alpha*self.Lmbda+1)

        # update dictionary with FISTA
        fopt = copy.deepcopy(self.opt['CCMOD'])
        fopt['X0'] = self.D
        if self.opt['OCDL', 'DiminishingTol']:
            fopt['RelStopTol'] = \
                self.dtype.type(self.opt['CCMOD', 'RelStopTol']/(1.+self.j))
        self.timer.start('dstep')
        dstep = SpatialFISTA(self.At, self.Bt, opt=fopt)
        dstep.solve()
        self.timer.stop('dstep')

        # set dictionary
        self.setdict(dstep.getmin())

        self.timer.stop('solve_wo_eval')
        evl = self.evaluate(S, X)
        self.timer.start('solve_wo_eval')

        t = self.timer.elapsed(self.opt['IterTimer'])
        if self.opt['OCDL', 'CUCBPDN']:
            # this requires a slight modification of dictlrn
            itst = self.isc.iterstats(self.j, t, None, dstep.itstat[-1], evl)
        else:
            itst = self.isc.iterstats(self.j, t, xstep.itstat[-1],
                                      dstep.itstat[-1], evl)
        self.itstat.append(itst)

        if self.opt['Verbose']:
            self.isc.printiterstats(itst)

        self.j += 1

        self.timer.stop(['solve', 'solve_wo_eval'])

        if 0:
            import matplotlib.pyplot as plt
            plt.imshow(su.tiledict(self.getdict().squeeze()))
            plt.show()

        return self.getdict()

    def config_itstats(self):
        """Setup config fields."""
        # NOTE: BackTrack is not implemented so always False.
        isfld = ['Iter', 'ObjFun', 'DFid', 'RegL1',
                 'XPrRsdl', 'XDlRsdl', 'XRho', 'X_It',
                 'D_L', 'D_Rsdl', 'D_It', 'Time']
        isxmap = {'XPrRsdl': 'PrimalRsdl', 'XDlRsdl': 'DualRsdl',
                  'XRho': 'Rho', 'X_It': 'Iter'}
        isdmap = {'D_L': 'L', 'D_Rsdl': 'Rsdl', 'D_It': 'Iter'}
        hdrtxt = ['Itn', 'Fnc', 'DFid', u('ℓ1'),
                  'Itn_X', 'r_X', 's_X', u('ρ_X'),
                  'Itn_D', 'r_D', 'L_D', 'Time']
        hdrmap = {'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
                  u('ℓ1'): 'RegL1', 'r_X': 'XPrRsdl', 's_X': 'XDlRsdl',
                  u('ρ_X'): 'XRho', 'Itn_X': 'X_It',
                  'r_D': 'D_Rsdl', 'L_D': 'D_L', 'Itn_D': 'D_It', 'Time': 'Time'}
        _evlmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1'}
        if self.opt['AccurateDFid']:
            evlmap = _evlmap
        else:
            evlmap = {}
            isxmap.update(_evlmap)
        return dictlrn.IterStatsConfig(
            isfld=isfld, isxmap=isxmap, isdmap=isdmap, evlmap=evlmap,
            hdrtxt=hdrtxt, hdrmap=hdrmap,
            fmtmap={'Itn_X': '%4d', 'Itn_D': '%4d'}
        )

    def getdict(self):
        """getdict() returns a squeezed version of internal dictionary."""
        return self.D.squeeze()

    def setdict(self, D=None):
        """Set dictionary properly."""
        self.D = np.asarray(D.reshape(self.cri.shpD), dtype=self.dtype)

    def update_At(self, Xe):
        r"""Update At. At is computed as following:

        .. math::
            At = \sum X^(\tau)^T X^(\tau)

        """
        Lmbda_new = self.dtype.type(self.alpha*self.Lmbda+1)
        #  if 0:
        #      Hc, Wc = self.D.shape[:2]
        #      # [H, W, Hc, Wc, 1, K, M] -> [H, W, K, Hc, Wc, 1, M]
        #      new = np.moveaxis(Xe, 5, 2)
        #      shp = [np.prod(new.shape[:3]), np.prod(new.shape[3:])]
        #      # -> [H*W*K, Hc*Wc*M]
        #      new = new.reshape(shp)
        #      # -> [Hc*Wc*M, Hc*Wc*M]
        #      new = new.T.dot(new)
        #      new = new.reshape(Hc, Wc, self.cri.M, Hc, Wc, 1, self.cri.M)
        #      new = np.moveaxis(new, 2, -1)
        #  else:
        #      # (H, W, Hc, Wc, 1, K, M)*2 -> (Hc, Wc, Hc, Wc, 1, M, M)
        #      new = einsum('ijklmno,ijpqrns->klpqmso', (Xe, Xe))
        X = np.moveaxis(Xe, 5, 2)
        shp = [np.prod(X.shape[:3]), np.prod(X.shape[3:])]
        X = X.reshape(shp)
        new = X.T.dot(X)
        self.At = np.asarray(
            (self.At*self.alpha*self.Lmbda+new) * (1./Lmbda_new),
            dtype=self.dtype
        )

    def update_Bt(self, Xe, S):
        r"""Update Bt. Bt is computed as following:

        .. math::
            Bt = \sum X^(\tau)^T S

        """
        # (H, W, Hc, Wc, 1, K, M), (H, W, C, K, 1) -> (Hc, Wc, C, 1, M)
        Lmbda_new = self.dtype.type(self.alpha*self.Lmbda+1)
        #  if 0:
        #      Hc, Wc, = self.D.shape[:2]
        #      Xe_mat = np.moveaxis(Xe, 5, 2)
        #      S_mat = np.moveaxis(S, 3, 2)
        #      # [H*W*K, Hc, Wc, 1, M]
        #      Xe_mat = Xe_mat.reshape(np.prod(Xe_mat.shape[:3]), *Xe_mat.shape[3:])
        #      # [H*W*K, C]
        #      S_mat = S_mat.reshape(np.prod(S_mat.shape[:3]), -1)
        #      new = np.zeros((Hc, Wc, self.cri.C, 1, self.cri.M), dtype=self.dtype)
        #      for c in range(self.cri.C):
        #          out = np.einsum('i,i...', S_mat[:, c], Xe_mat)
        #          new[:, :, c, ...] = out
        #  else:
        #      new = einsum('ijklmno,ijpnq->klpqo', (Xe, S))
        X = np.moveaxis(Xe, 5, 2)
        X = X.reshape(np.prod(X.shape[:3]), -1)
        S = np.moveaxis(S, 3, 2)
        S = S.reshape(np.prod(S.shape[:3]), -1)
        new = X.T.dot(S)
        self.Bt = np.asarray(
            (self.Bt*self.alpha*self.Lmbda+new) * (1./Lmbda_new),
            dtype=self.dtype
        )

    @property
    def alpha(self):
        """Forgetting factor."""
        if self.j == 0:
            return self.dtype.type(0.)
        # j starts from 0
        alpha = self.dtype.type(pow(1.-1./(self.j+1.), self.p))
        return alpha

    def evaluate(self, S, X):
        """Optionally evaluate functional values."""
        if self.opt['AccurateDFid']:
            Df = sl.rfftn(self.D, self.cri.Nv, self.cri.axisN)
            Xf = sl.rfftn(X, self.cri.Nv, self.cri.axisN)
            Sf = sl.rfftn(S, self.cri.Nv, self.cri.axisN)
            Ef = sl.inner(Df, Xf, axis=self.cri.axisM) - Sf
            dfd = sl.rfl2norm2(Ef, S.shape, axis=self.cri.axisN) / 2.
            rl1 = np.sum(np.abs(X))
            evl = dict(DFid=dfd, RegL1=rl1, ObjFun=dfd+self.lmbda*rl1)
        else:
            evl = None
        return evl

    def extend_code(self, X):
        """Extend 5-dim X to 7-dim Xe."""
        H, W = X.shape[:2]
        kernel_h, kernel_w = self.D.shape[:2]
        code = np.zeros((H, W, kernel_h, kernel_w, 1, self.cri.K, self.cri.M), dtype=self.dtype)
        pad = [(kernel_h-1, 0), (kernel_w-1, 0)] + [(0, 0) for _ in range(X.ndim-2)]
        Xp = np.pad(X, pad, 'wrap')
        for h in range(kernel_h):
            for w in range(kernel_w):
                slices = Xp[h:h+H, w:w+W, ...]
                code[:, :, kernel_h-1-h, kernel_w-1-w, ...] = slices
        return code

    def getitstat(self):
        """Get iteration stats as named tuple of arrays instead of array of
        named tuples.
        """

        return su.transpose_ntpl_list(self.itstat)
