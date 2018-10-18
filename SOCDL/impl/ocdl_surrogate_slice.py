# -*- coding: utf-8 -*-
"""Online version of slice based CDL with FISTA solver."""
import copy
import collections
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
from sporco.admm import parcbpdn
from sporco.fista import fista
try:
    import sporco_cuda.cbpdn as cucbpdn
except ImportError:
    cucbpdn = None

from ..utils import Pcn, einsum

logger = logging.getLogger(__name__)


def reconstruct_additive_component(ams):
    """Get the reconstructed component from ams."""
    Xn = ams.cbpdn.Y[ams.index_addmsk()]
    Dnf = ams.cbpdn.Df[ams.index_addmsk()]
    Xnf = sl.rfftn(Xn, None, ams.cri.axisN)
    Sf = np.sum(Dnf * Xnf, axis=ams.cri.axisM)
    return sl.irfftn(Sf, ams.cri.Nv, ams.cri.axisN).squeeze()


class IterStatsConfig(object):

    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""

    def __init__(self, isfld, isxmap, isdmap, evlmap, hdrtxt, hdrmap,
                 fmtmap=None):
        self.IterationStats = collections.namedtuple('IterationStats', isfld)
        self.isxmap = isxmap
        self.isdmap = isdmap
        self.evlmap = evlmap
        self.hdrtxt = hdrtxt
        self.hdrmap = hdrmap

        # Call utility function to construct status display formatting
        self.hdrstr, self.fmtstr, self.nsep = common.solve_status_str(
            hdrtxt, fmtmap=fmtmap, fwdth0=type(self).fwiter,
            fprec=type(self).fpothr)

    def iterstats(self, j, t, dtx, dtd, evl):
        """Construct IterationStats namedtuple from X step and D step
        statistics.

        Parameters
        ----------
        j: int
            Iteration number
        t: float
            Iteration time
        dtx: dict
            Dict holding statistics from X step
        dtd: dict
            Dict holding statistics from D step
        evl: dict
            Dict holding statistics from extra evaluations
        """
        vlst = []
        for fnm in self.IterationStats._fields:
            if fnm in self.isxmap:
                vlst.append(dtx[self.isxmap[fnm]])
            elif fnm in self.isdmap:
                vlst.append(dtd[self.isdmap[fnm]])
            elif fnm in self.evlmap:
                vlst.append(evl[fnm])
            elif fnm == 'Iter':
                vlst.append(j)
            elif fnm == 'Time':
                vlst.append(t)
            else:
                vlst.append(None)

        return self.IterationStats._make(vlst)

    def printheader(self):
        self.print_func(self.hdrstr)
        self.printseparator()

    def printseparator(self):
        self.print_func("-" * self.nsep)

    def printiterstats(self, itst):
        """Print iteration statistics.

        Parameters
        ----------
        itst : namedtuple
            IterationStats namedtuple as returned by :meth:`iterstats`
        """

        itdsp = tuple([getattr(itst, self.hdrmap[col]) for col in self.hdrtxt])
        self.print_func(self.fmtstr % itdsp)

    def print_func(self, s):
        """Print function."""
        print(s)


class StripeSliceFISTA(fista.FISTA):
    r"""FISTA algorithm to solve for the dictionary, where the derivative is
    given by

    .. math::
        \nabla_D f = \Omega At - Bt,

    where :math:`\Omega` is the stripe dictionary.
    """

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
        if opt is None:
            opt = StripeSliceFISTA.Options()
        if opt['X0'] is not None:
            self.dsz = opt['X0'].shape
        else:
            self.dsz = dsz
        super().__init__(np.prod(dsz), dsz, At.dtype, opt)
        self.At = At
        self.Bt = Bt
        self.Y = self.X.copy()
        self.osz = list(copy.deepcopy(self.dsz))
        self.osz[2] = 2 * self.osz[0] - 1
        self.osz[3] = 2 * self.osz[1] - 1
        self.Omega = np.zeros(self.osz, dtype=self.dtype)

    def set_Omega(self, D=None):
        r"""Set the stripe dictionary :math:`\Omega` from D."""
        if D is None:
            D = self.Y
        self.Omega.fill(0.)
        for ih, h in enumerate(range(-self.osz[0]+1, self.osz[0])):
            for iw, w in enumerate(range(-self.osz[1]+1, self.osz[1])):
                begh = -min(h, 0)
                endh = self.osz[0] - max(h, 0)
                begw = -min(w, 0)
                endw = self.osz[1] - max(w, 0)
                stripe_dict = D[begh:endh, begw:endw, 0, 0, ...]
                begh_c = max(h, 0)
                endh_c = min(self.osz[0] + h, self.osz[0])
                begw_c = max(w, 0)
                endw_c = min(self.osz[1] + w, self.osz[1])
                self.Omega[begh_c:endh_c, begw_c:endw_c, ih, iw, ...] = \
                    stripe_dict

    def Pcn(self, D):
        """Proximal function."""
        return Pcn(D, self.opt['ZeroMean'])

    def eval_grad(self):
        r"""Evaluate the gradient:

            .. ::math:
                \nabla_D f = \Omega At - Bt
        """
        self.set_Omega()
        grad = einsum('ijklmno,klpqros->ijpqmns', (self.Omega, self.At))
        grad -= self.Bt
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


class OnlineDictLearnSliceSurrogate(
    with_metaclass(dictlrn._DictLearn_Meta, common.BasicIterativeSolver)
):
    r"""Stochastic Approximation based online convolutional dictionary
    learning.
    """

    class Options(cdict.ConstrainedDict):
        defaults = {
            'Verbose': True, 'StatusHeader': True, 'IterTimer': 'solve',
            'MaxMainIter': 1000, 'Callback': None,
            'AccurateDFid': False, 'DataType': None,
            'CBPDN': copy.deepcopy(cbpdn.ConvBPDN.Options.defaults),
            'CCMOD': copy.deepcopy(StripeSliceFISTA.Options.defaults),
            'OCDL': {
                'p': 1.,  # forgetting exponent
                'DiminishingTol': False,  # diminishing tolerance for FISTA
                'CUCBPDN': False,  # use gpu version of cbpdn
                'PARCBPDN': False,  # use parallel version of cbpdn
                'nproc': None,  # Number of process for parcbpdn
            }
        }
        defaults['CBPDN'].update({
            'Verbose': False, 'MaxMainIter': 50, 'AuxVarObj': False,
            'RelStopTol': 1e-7, 'DataType': None, 'FastSolve': True,
        })
        defaults['CBPDN']['AutoRho'].update({'Enabled': False})
        defaults['CCMOD']['BackTrack'].update({'Enabled': False})

        def __init__(self, opt=None):
            super().__init__({
                'CBPDN': cbpdn.ConvBPDN.Options(self.defaults['CBPDN']),
                'CCMOD': StripeSliceFISTA.Options(self.defaults['CCMOD']),
            })
            if opt is None:
                opt = {}
            self.update(opt)

    def __new__(cls, *args, **kwargs):
        instance = super(OnlineDictLearnSliceSurrogate, cls).__new__(cls)
        instance.timer = su.Timer(['init', 'solve', 'solve_wo_eval', 'hessian',
                                   'xstep', 'dstep'])
        instance.timer.start('init')
        return instance

    def __init__(self, D0, S0, lmbda=None, opt=None, dimK=1, dimN=2):
        """Internally we use a 7-dim representation over blobs. This increases
        the spatial dimension of 2 to 4 to allow for extra dimensions for
        slices.

        -------------------------------------------------------------------
        blob     | spatial                                ,chn  ,sig  ,fil
        -------------------------------------------------------------------
        S        |  (H      ,  W      ,  1      ,  1      ,  C  ,  K  ,  1)
        D        |  (Hc     ,  Wc     ,  1      ,  1      ,  C  ,  1  ,  M)
        X        |  (H      ,  W      ,  1      ,  1      ,  1  ,  K  ,  M)
        Omega    |  (Hc     ,  Wc     ,  2Hc-1  ,  2Wc-1  ,  C  ,  1  ,  M)
        At       |  (2Hc-1  ,  2Wc-1  ,  1      ,  1      ,  1  ,  M  ,  M)
        Bt       |  (Hc     ,  Wc     ,  1      ,  1      ,  C  ,  1  ,  M)
        patches  |  (H      ,  W      ,  Hc     ,  Wc     ,  C  ,  K  ,  1)
        gamma    |  (H      ,  W      ,  2Hc-1  ,  2Wc-1  ,  1  ,  K  ,  M)
        -------------------------------------------------------------------

        Here the `signal` dimension of At is occupied by M, which comes from
        stripe dictionary Omega.
        """
        if opt is None:
            opt = OnlineDictLearnSliceSurrogate.Options()
        assert isinstance(opt, OnlineDictLearnSliceSurrogate.Options)
        self.opt = opt

        self.set_dtype(opt, S0.dtype)

        # insert extra dims
        D0 = D0[:, :, np.newaxis, np.newaxis, ...]
        S0 = S0[:, :, np.newaxis, np.newaxis, ...]

        assert dimN == 2
        self.cri = cr.CSC_ConvRepIndexing(D0, S0, dimK=None, dimN=4)
        self.osz = list(copy.deepcopy(self.cri.shpD))
        self.osz[2], self.osz[3] = 2*self.osz[0]-1, 2*self.osz[1]-1

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

    def solve(self, S, W=None):
        """Solve for given signal S, optionally with mask W."""
        self.cri = cr.CSC_ConvRepIndexing(
            self.D.squeeze()[:, :, None, None, ...],
            S[:, :, None, None, ...],
            dimK=None, dimN=4
        )

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
            if W is None:
                xstep = cbpdn.ConvBPDN(self.getdict(), S, self.lmbda, opt=copt)
                xstep.solve()
                X = np.asarray(xstep.getcoef().reshape(self.cri.shpX),
                               dtype=self.dtype)
            else:
                xstep = cbpdn.AddMaskSim(cbpdn.ConvBPDN, self.getdict(), S, W,
                                         self.lmbda, opt=copt)
                X = xstep.solve()
                X = np.asarray(X.reshape(self.cri.shpX), dtype=self.dtype)
                # The additive component is removed from masked signal
                add_cpnt = reconstruct_additive_component(xstep)
                S -= add_cpnt.reshape(S.shape)

        self.timer.stop('xstep')

        # update At and Bt
        self.timer.start('hessian')
        patches = self.im2slices(S)
        self.update_At(X)
        self.update_Bt(X, patches)
        self.timer.stop('hessian')
        self.Lmbda = self.dtype.type(self.alpha*self.Lmbda+1)

        # update dictionary with FISTA
        fopt = copy.deepcopy(self.opt['CCMOD'])
        fopt['X0'] = self.D
        if self.opt['OCDL', 'DiminishingTol']:
            fopt['RelStopTol'] = \
                self.dtype.type(self.opt['CCMOD', 'RelStopTol']/(1.+self.j))
        self.timer.start('dstep')
        dstep = StripeSliceFISTA(self.At, self.Bt, opt=fopt)
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

    def update_At(self, X):
        r"""Update At. At is computed as following:

        .. math::
            At = \sum \gamma_i x_i^T.

        """
        gamma = self.stripe_slice(X)
        Lmbda_new = self.dtype.type(self.alpha*self.Lmbda+1)
        if 0:
            new = einsum('ijklmno,ijpqrns->klpqmos', (gamma, X))
        else:
            gamma = np.moveaxis(gamma, 5, 2)
            gamma = gamma.reshape(np.prod(gamma.shape[:3]), -1)
            X = X.reshape(-1, X.shape[-1])
            new = gamma.T.dot(X)
            Hc, Wc = self.cri.shpD[:2]
            shp = [2*Hc-1, 2*Wc-1, 1, 1, 1, self.cri.M, self.cri.M]
            new = new.reshape(shp)
        self.At = np.asarray(
            (self.At*self.alpha*self.Lmbda+new) * (1./Lmbda_new),
            dtype=self.dtype
        )

    def update_Bt(self, X, patches):
        r"""Update Bt. Bt is computed as following:

        .. math::
            Bt = \sum s_i x_i^T.

        """
        Lmbda_new = self.dtype.type(self.alpha*self.Lmbda+1)
        if 0:
            new = einsum('ijklmno,ijpqrns->klpqmos', (patches, X))
        else:
            patches = np.moveaxis(patches, 5, 2)
            patches = patches.reshape(np.prod(patches.shape[:3]), -1)
            X = X.reshape(-1, X.shape[-1])
            new = patches.T.dot(X)
            Hc, Wc = self.cri.shpD[:2]
            shp = [Hc, Wc, 1, 1, self.cri.Cd, 1, self.cri.M]
            new = new.reshape(shp)
        self.Bt = np.asarray(
            (self.Bt*self.alpha*self.Lmbda+new) * (1./Lmbda_new),
            dtype=self.dtype
        )

    def stripe_slice(self, X):
        r"""Construct stripe slice (:math:`\gamma`) from sparse code X."""
        H, W = X.shape[:2]
        Hc, Wc = self.cri.shpD[:2]
        sz = list(copy.deepcopy(X.shape))
        sz[2], sz[3] = 2*Hc-1, 2*Wc-1
        slices = np.zeros(sz, dtype=self.dtype)
        pad = [(Hc-1, Hc-1), (Wc-1, Wc-1)] + [(0, 0) for _ in range(X.ndim-2)]
        Xp = np.pad(X, pad, 'wrap')
        for h in range(sz[2]):
            for w in range(sz[3]):
                gamma = Xp[h:h+H, w:w+W, 0, 0, ...]
                slices[:, :, h, w, ...] = gamma
        return slices

    def im2slices(self, S):
        """Convert signals to patches."""
        Hc, Wc = self.cri.shpD[:2]
        H, W = S.shape[:2]
        if self.cri.C > 1:
            shp = [H, W, Hc, Wc, self.cri.C, self.cri.K]
        else:
            shp = [H, W, Hc, Wc, self.cri.K]
        slices = np.zeros(shp, dtype=self.dtype)
        pad = [(0, Hc-1), (0, Wc-1)] + [(0, 0) for _ in range(S.ndim-2)]
        Sp = np.pad(S, pad, 'wrap')
        for h in range(Hc):
            for w in range(Wc):
                slices[:, :, h, w, ...] = Sp[h:h+H, w:w+W, ...]
        if self.cri.C == 1:
            slices = np.expand_dims(slices, 4)
        return np.expand_dims(slices, -1)

    @property
    def alpha(self):
        """Forgetting factor."""
        # j starts from 0
        alpha = self.dtype.type(pow(1.-1./(self.j+1.), self.p))
        return alpha

    def evaluate(self, S, X):
        """Optionally evaluate functional values."""
        if self.opt['AccurateDFid']:
            cri_s = cr.CSC_ConvRepIndexing(self.getdict(), S.squeeze(),
                                           dimK=None, dimN=2)
            Df = sl.rfftn(self.D.reshape(cri_s.shpD), cri_s.Nv, cri_s.axisN)
            Xf = sl.rfftn(X.reshape(cri_s.shpX), cri_s.Nv, cri_s.axisN)
            Sf = sl.rfftn(S.reshape(cri_s.shpS), cri_s.Nv, cri_s.axisN)
            Ef = sl.inner(Df, Xf, axis=cri_s.axisM) - Sf
            dfd = sl.rfl2norm2(Ef, S.shape, axis=cri_s.axisN) / 2.
            rl1 = np.sum(np.abs(X))
            evl = dict(DFid=dfd, RegL1=rl1, ObjFun=dfd+self.lmbda*rl1)
        else:
            evl = None
        return evl

    def getitstat(self):
        """Get iteration stats as named tuple of arrays instead of array of
        named tuples.
        """

        return su.transpose_ntpl_list(self.itstat)
