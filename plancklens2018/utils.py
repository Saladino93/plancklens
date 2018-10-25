from __future__ import print_function
from __future__ import division

import healpy as hp
from healpy.projector import CartesianProj
import time
import numpy as np
import sys
import hashlib

def alm_copy(alm, lmax=None):
    """ Copies the healpy alm array, with the option to reduce its lmax """
    alm_lmax = int(np.floor(np.sqrt(2 * len(alm)) - 1))
    assert lmax <= alm_lmax, (lmax, alm_lmax)
    if (alm_lmax == lmax) or (lmax is None):
        ret = np.copy(alm)
    else:
        ret = np.zeros((lmax + 1) * (lmax + 2) // 2, dtype=np.complex)
        for m in range(0, lmax + 1):
            ret[((m * (2 * lmax + 1 - m) // 2) + m):(m * (2 * lmax + 1 - m) // 2 + lmax + 1)] \
                = alm[(m * (2 * alm_lmax + 1 - m) // 2 + m):(m * (2 * alm_lmax + 1 - m) // 2 + lmax + 1)]
    return ret

def projectmap(hpmap, lcell_amin, Npts, lon=0., lat=-45. ):
    """ Projects portion of healpix map onto square map. Returns projected map and projector for future calls """
    assert 0. <= lon <= 360. and -90. <= lat <= 90.,(lon,lat)
    _lon = lon if lon <= 180 else lon - 360
    lonra = (-lcell_amin * Npts / 60. / 2., lcell_amin / 60 * Npts / 2.)
    latra = (-lcell_amin * Npts / 60  / 2., lcell_amin / 60 * Npts / 2.)
    P = CartesianProj(rot = [_lon,lat,0.],lonra=lonra, latra=latra, xsize=Npts, ysize=Npts)
    P.set_flip('astro')
    return P.projmap(hpmap, lambda x, y, z: hp.vec2pix(hp.npix2nside(len(hpmap)), x, y, z)), P

def enumerate_progress(list, label=''):
    """ Progress bar """
    t0 = time.time()
    ni = len(list)
    for i, v in enumerate(list):
        yield i, v
        ppct = int(100. * (i - 1) / ni)
        cpct = int(100. * (i + 0) / ni)
        if cpct > ppct:
            dt = time.time() - t0
            dh = np.floor(dt / 3600.)
            dm = np.floor(np.mod(dt, 3600.) / 60.)
            ds = np.floor(np.mod(dt, 60))
            sys.stdout.write("\r [" + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " +
                             label + " " + int(10. * cpct / 100) * "-" + "> " + ("%02d" % cpct) + r"%")
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

def clhash(cl, dtype=np.float16):
    """ Hash for generic cl. By default we avoid double precision checks since this might be machine dependent """
    return hashlib.sha1(np.copy(cl.astype(dtype), order='C')).hexdigest()

def mchash(cl):
    """ Hash for integer (e.g. sim indices) array where order does not matter """
    return hashlib.sha1(np.copy(np.sort(cl), order='C')).hexdigest()

def cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

def get_cl(cl):
    if isinstance(cl, str):
        data =  np.loadtxt(cl)
        if data.ndim > 1:
            data = data.transpose()
            ell = np.int_(data[0])
            assert np.all(ell == np.arange(ell[0], ell[-1] + 1, dtype=int)), 'I dont understand this file: ' + cl
            ret = np.zeros(ell[-1] + 1)
            ret[ell] =  data[-1]
            return ret
    else:
        return cl

def hash_check(hash1, hash2, ignore=['lib_dir', 'prefix'], keychain=[]):
    keys1 = hash1.keys()
    keys2 = hash2.keys()

    for key in ignore:
        if key in keys1: keys1.remove(key)
        if key in keys2: keys2.remove(key)

    for key in set(keys1).union(set(keys2)):
        v1 = hash1[key]
        v2 = hash2[key]

        def hashfail(msg=None):
            print("ERROR: HASHCHECK FAIL AT KEY = " + ':'.join(keychain + [key]))
            if msg is not None:
                print("   " + msg)
            print("   ", "V1 = ", v1)
            print("   ", "V2 = ", v2)
            assert 0

        if type(v1) != type(v2):
            hashfail('UNEQUAL TYPES')
        elif type(v2) == dict:
            hash_check( v1, v2, ignore=ignore, keychain=keychain + [key] )
        elif type(v1) == np.ndarray:
            if not np.allclose(v1, v2):
                hashfail('UNEQUAL ARRAY')
        else:
            if not( v1 == v2 ):
                hashfail('UNEQUAL VALUES')
class stats:
    """
    Simple minded routines for means and averages of sims .
    Calculates means as 1/N sum()
    and Cov as 1/(N-1)sum(x - mean)(x - mean)^t
    """

    def __init__(self, size, xcoord=None, docov=True):
        self.N = 0  # number of samples
        self.size = size  # dim of data vector
        self.sum = np.zeros(self.size)  # sum_i x_i
        if docov: self.mom = np.zeros((self.size, self.size))  # sum_i x_ix_i^t
        self.xcoord = xcoord
        self.docov = docov

    def add(self, v):
        assert (v.shape == (self.size,)), "input not understood"
        self.sum += v
        if self.docov:
            self.mom += np.outer(v, v)
        self.N += 1

    def mean(self):
        assert (self.N > 0)
        return self.sum / float(self.N)

    def avg(self):
        return self.mean()

    def cov(self):
        assert self.docov
        assert (self.N > 0)
        if self.N == 1: return np.zeros((self.size, self.size))
        mean = self.mean()
        return self.mom / (self.N - 1.) - self.N / (self.N - 1.) * np.outer(mean, mean)

    def sigmas(self):
        return np.sqrt(np.diagonal(self.cov()))

    def corrcoeffs(self):
        sigmas = self.sigmas()
        return self.cov() / np.outer(sigmas, sigmas)

    def sigmas_on_mean(self):
        assert (self.N > 0)
        return self.sigmas() / np.sqrt(self.N)

    def inverse(self, bias_p=None):  # inverse cov, using unbiasing a factor following G. statistics
        assert (self.N > self.size), "Non invertible cov.matrix"
        if bias_p is None: bias_p = (self.N - self.size - 2.) / (self.N - 1)
        return bias_p * np.linalg.inv(self.cov())

    def get_chisq(self, data):  # Returns (data -mean)Sig^{-1}(data-mean)
        assert (data.size == self.size), "incompatible input"
        dx = data - self.mean()
        return np.sum(np.outer(dx, dx) * self.inverse())

    def get_chisq_pte(self, data):  # probability to exceed, or survival function
        from scipy.stats import chi2
        return chi2.sf(self.get_chisq(data), self.N - 1)  # 'survival function' of chisq distribution with N -1 dof

    def rebin_that_nooverlap(self, orig_coord, lmins, lmaxs, weights=None):
        # Returns a new stat instance rebinning with non-overlapping weights
        # >= a gauche, <= a droite.
        assert (orig_coord.size == self.size), "Incompatible input"
        assert (lmins.size == lmaxs.size), "Incompatible input"
        assert (np.all(np.diff(np.array(lmins)) > 0.)), "This only for non overlapping bins."
        assert (np.all(np.diff(np.array(lmaxs)) > 0.)), "This only for non overlapping bins."
        assert (np.all(lmaxs - lmins) > 0.), "This only for non overlapping bins."

        if weights is None: weights = np.ones(self.size)
        assert (weights.size == self.size), "incompatible input"
        newsize = len(lmaxs)
        assert (self.size > newsize), "Incompatible dimensions"
        Tmat = np.zeros((newsize, self.size))
        newsum = np.zeros(newsize)
        for k, lmin, lmax in zip(np.arange(newsize), lmins, lmaxs):
            idc = np.where((orig_coord >= lmin) & (orig_coord <= lmax))
            if len(idc) > 0:
                norm = np.sum(weights[idc])
                Tmat[k, idc] = weights[idc] / norm
                newsum[k] = np.sum(weights[idc] * self.sum[idc]) / norm

        newmom = np.dot(np.dot(Tmat, self.mom), Tmat.transpose())  # New mom. matrix is T M T^T
        newstats = stats(newsize, xcoord=0.5 * (lmins[0:len(lmins) - 1] + lmaxs[1:]))
        # Resets the stats things
        newstats.mom = newmom
        newstats.sum = newsum
        newstats.N = self.N
        return newstats

def camb_clfile(fname, lmax=None):
    """
    CAMB spectra (lenspotentialCls, lensedCls or tensCls types) returned as a dict of numpy arrays.
    """
    cols = np.loadtxt(fname).transpose()
    ell = np.int_(cols[0])
    if lmax is None: lmax = ell[-1]
    assert ell[-1] >= lmax, (ell[-1], lmax)
    cls = {k : np.zeros(lmax + 1, dtype=float) for k in ['tt', 'ee', 'bb', 'te']}
    w = ell * (ell + 1) / (2. * np.pi)  # weights in output file
    idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell), dtype=int)
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        cls[k][ell[idc]] = cols[i + 1][idc] / w[idc]
    if len(cols) > 5:
        wpp = lambda ell : ell ** 2 * (ell + 1) ** 2 / (2. * np.pi)
        wptpe = lambda ell : np.sqrt(ell.astype(float) ** 3 * (ell + 1.) ** 3) / (2. * np.pi)
        for i, k in enumerate(['pp', 'pt', 'pe']):
            cls[k] = np.zeros(lmax + 1, dtype=float)
        cls['pp'][ell[idc]] = cols[5][idc] / wpp(ell[idc])
        cls['pt'][ell[idc]] = cols[6][idc] / wptpe(ell[idc])
        cls['pe'][ell[idc]] = cols[7][idc] / wptpe(ell[idc])
    return cls