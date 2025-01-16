from __future__ import print_function

import os
import pickle as pk
import healpy as hp
import numpy as np
import plancklens.sims.phas

from plancklens.utils import clhash, hash_check, alm_copy
from plancklens.helpers import mpi
from plancklens.sims import phas, cmbs

class cmb_maps(object):
    r"""CMB simulation library combining a lensed CMB library and a transfer function.

        Args:
            sims_cmb_len: lensed CMB library (e.g. *plancklens.sims.planck2018_sims.cmb_len_ffp10*)
            cl_transf: CMB temperature transfer function
            nside: healpy resolution of the maps. Defaults to 2048.
            lib_dir(optional): hash checks will be cached, as well as possibly other things for subclasses.
            cl_transf_P: CMB pol transfer function (if different from cl_transf)

    """
    def __init__(self, sims_cmb_len, cl_transf, nside=2048, cl_transf_P=None, lib_dir=None, fixed_index = None):
        if cl_transf_P is None:
            cl_transf_P = np.copy(cl_transf)

        self.sims_cmb_len = sims_cmb_len
        self.cl_transf_T = cl_transf
        self.cl_transf_P = cl_transf_P
        self.nside = nside
        self.fixed_index = fixed_index

        if lib_dir is not None:
            fn_hash = os.path.join(lib_dir, 'sim_hash.pk')
            if mpi.rank == 0 and not os.path.exists(fn_hash):
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
            mpi.barrier()
            hash_check(self.hashdict(), pk.load(open(fn_hash, 'rb')), fn=fn_hash)

    def hashdict(self):
        ret = {'sims_cmb_len':self.sims_cmb_len.hashdict(),'nside':self.nside,'cl_transf':clhash(self.cl_transf_T)}
        if not (np.all(self.cl_transf_P == self.cl_transf_T)):
            ret['cl_transf_P'] = clhash(self.cl_transf_P)
        return ret

    def get_sim_tmap(self,idx):
        """Returns temperature healpy map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """        
        print("idx lensed simulation: ", idx, "vs", self.fixed_index)

        tmap = self.sims_cmb_len.get_sim_tlm(idx, self.fixed_index)
        tmap = hp.almxfl(tmap, self.cl_transf_T)
        tmap = hp.alm2map(tmap,self.nside)
        return (tmap + self.get_sim_tnoise(idx)).astype(np.float32)

    def get_sim_pmap(self,idx):
        """Returns polarization healpy maps for a simulation

            Args:
                idx: simulation index

            Returns:
                Q and U healpy maps

        """
        elm = self.sims_cmb_len.get_sim_elm(idx)
        hp.almxfl(elm,self.cl_transf_P,inplace=True)
        blm = self.sims_cmb_len.get_sim_blm(idx)
        hp.almxfl(blm, self.cl_transf_P, inplace=True)
        Q,U = hp.alm2map_spin([elm,blm], self.nside, 2,hp.Alm.getlmax(elm.size))
        del elm,blm
        return Q + self.get_sim_qnoise(idx),U + self.get_sim_unoise(idx)

    def get_sim_tnoise(self,idx):
        assert 0,'subclass this'

    def get_sim_qnoise(self, idx):
        assert 0, 'subclass this'

    def get_sim_unoise(self, idx):
        assert 0, 'subclass this'

class cmb_maps_noisefree(cmb_maps):
    def __init__(self,sims_cmb_len,cl_transf,nside=2048, cl_transf_P=None):
        super(cmb_maps_noisefree, self).__init__(sims_cmb_len, cl_transf, nside=nside, cl_transf_P=cl_transf_P)

    def get_sim_tnoise(self,idx):
        return np.zeros(hp.nside2npix(self.nside))

    def get_sim_qnoise(self, idx):
        return np.zeros(hp.nside2npix(self.nside))

    def get_sim_unoise(self, idx):
        return np.zeros(hp.nside2npix(self.nside))

class cmb_maps_nlev(cmb_maps):
    r"""CMB simulation library combining a lensed CMB library, transfer function and idealized homogeneous noise.

        Args:
            sims_cmb_len: lensed CMB library (e.g. *plancklens.sims.planck2018_sims.cmb_len_ffp10*)
            cl_transf: CMB transfer function, identical in temperature and polarization
            nlev_t: temperature noise level in :math:`\mu K`-arcmin
            nlev_p: polarization noise level in :math:`\mu K`-arcmin
            nside: healpy resolution of the maps
            lib_dir(optional): noise maps random phases will be cached there. Only relevant if *pix_lib_phas is not set*
            pix_lib_phas(optional): random phases library for the noise maps (from *plancklens.sims.phas.py*).
                                    If not set, *lib_dir* arg must be set.


    """
    def __init__(self,sims_cmb_len, cl_transf, nlev_t, nlev_p, nside, lib_dir=None, pix_lib_phas=None, 
                 fixed_index = None, zero_noise = False, modulation = 0, cls_noise = None, noise_phas:plancklens.sims.phas.lib_phas = None, rmat = None, noise_index : int = 0,
                 variance_map = None):
        if pix_lib_phas is None:
            assert lib_dir is not None
            pix_lib_phas = phas.pix_lib_phas(lib_dir, 3, (hp.nside2npix(nside),))
        assert pix_lib_phas.shape == (hp.nside2npix(nside),), (pix_lib_phas.shape, (hp.nside2npix(nside),))
        self.pix_lib_phas = pix_lib_phas
        self.nlev_t = nlev_t
        self.nlev_p = nlev_p
        self.cls_noise = cls_noise
        self.zero_noise = zero_noise
        self.modulation = modulation
        self.noise_phas = noise_phas
        self.rmat = rmat
        self.noise_index = noise_index
        self.variance_map = variance_map

        super(cmb_maps_nlev, self).__init__(sims_cmb_len, cl_transf, nside=nside, lib_dir=lib_dir, fixed_index = fixed_index)


    def hashdict(self):
        ret = {'sims_cmb_len':self.sims_cmb_len.hashdict(),
                'nside':self.nside,'cl_transf':clhash(self.cl_transf_T),
                'nlev_t':self.nlev_t,'nlev_p':self.nlev_p, 'pixphas':self.pix_lib_phas.hashdict()}
        if not (np.all(self.cl_transf_P == self.cl_transf_T)):
            ret['cl_transf_P'] = clhash(self.cl_transf_P)
        return ret

    def get_sim_tnoise(self,idx):
        """Returns noise temperature map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """
        if (self.cls_noise is None) and (self.rmat is None) and (self.variance_map is None):
            vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
            print("Modulating noise is", self.modulation)
            #np.save("noise_map.npy", self.nlev_t / vamin * self.pix_lib_phas.get_sim(idx, idf=0)* (not self.zero_noise) * np.sqrt((1 + self.modulation)))
            #return self.nlev_t / vamin * self.pix_lib_phas.get_sim(idx, idf=0) * (not self.zero_noise) * (1 + self.modulation)
            return self.nlev_t / vamin * (not self.zero_noise) * (1 + self.modulation)
        elif (self.variance_map is not None):
            print("Getting noise map from variance map.")
            return self.get_sim_tnoise_from_variance_map(idx)
        elif self.rmat is not None:
            noise_alms = self._get_sim_alm(idx, self.noise_index)
            noise_map = hp.alm2map(noise_alms, self.nside)
            np.save(f"noise_map_{self.noise_index}.npy", noise_map)
            return noise_map * (not self.zero_noise)        
        else:
            assert 'tt' in self.cls_noise
            noise_alms = hp.almxfl(self.noise_phas.get_sim(idx, 0), np.sqrt(self.cls_noise['tt']))
            noise_map = hp.alm2map(noise_alms, self.nside)
            np.save("noise_map.npy", noise_map)
            return noise_map * (not self.zero_noise)        
        

    def get_sim_tnoise_from_variance_map(self, idx):
        """Returns noise temperature map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        tmap = self.variance_map
        assert hp.npix2nside(len(tmap)) == self.nside
        return self.nlev_t / vamin /np.std(tmap) * tmap * self.pix_lib_phas.get_sim(idx, idf=0)

    def _get_sim_alm(self, idx, idf):
        ret = hp.almxfl(self.noise_phas.get_sim(idx, idf=0), self.rmat[:, idf, 0])
        for _i in range(1,2):
            ret += hp.almxfl(self.noise_phas.get_sim(idx, idf=_i), self.rmat[:, idf, _i])
        return ret


    def get_sim_qnoise(self, idx):
        """Returns noise Q-polarization map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        return self.nlev_p / vamin * self.pix_lib_phas.get_sim(idx, idf=1)

    def get_sim_unoise(self, idx):
        """Returns noise U-polarization map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        return self.nlev_p / vamin * self.pix_lib_phas.get_sim(idx, idf=2)



class cmb_maps_harmonicspace(object):
    r"""CMB simulation library combining a lensed CMB library and a transfer function

        Note:
            In this version, maps are directly produced in harmonic space with possibly non-white but stat. isotropic noise

        Args:
            sims_cmb_len: lensed CMB library (e.g. *plancklens.sims.planck2018_sims.cmb_len_ffp10*)
            cls_transf: dict with transfer function for 't' 'e' and 'b' CMB fields
            cls_noise: dict with noise spectra for 't' 'e' and 'b'
            noise_phas: *plancklens.sims.phas.lib-phas* with at least 3 fields for the random phase library for the noise generation
            lib_dir(optional): hash checks will be cached, as well as possibly other things for subclasses.
            nside: If provided, maps are returned in pixel space instead of harmonic space

        Note:
            lmax's of len cmbs and noise phases must match


    """
    def __init__(self, sims_cmb_len, cls_transf:dict, cls_noise:dict, noise_phas:plancklens.sims.phas.lib_phas, lib_dir=None, nside=None):
        assert noise_phas.nfields >= 3, noise_phas.nfields
        self.sims_cmb_len = sims_cmb_len
        self.cls_transf = cls_transf
        self.cls_noise = cls_noise
        self.phas = noise_phas
        self.nside = nside

        if hasattr(sims_cmb_len, 'lmax'):
            assert self.sims_cmb_len.lmax == self.phas.lmax, f"Lmax of lensed CMB and of noise phases should match, here {self.sims_cmb_len.lmax} and {self.phas.lmax}"

        if lib_dir is not None:
            fn_hash = os.path.join(lib_dir, 'sim_hash.pk')
            if mpi.rank == 0 and not os.path.exists(fn_hash):
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
            mpi.barrier()
            hash_check(self.hashdict(), pk.load(open(fn_hash, 'rb')))

    def hashdict(self):
        ret = {'sims_cmb_len':self.sims_cmb_len.hashdict(), 'phas':self.phas.hashdict()}
        for k in self.cls_noise:
            ret['noise' + k] = clhash(self.cls_noise[k])
        for k in self.cls_transf:
            ret['transf' + k] = clhash(self.cls_transf[k])
        return ret

    def get_sim_tmap(self,idx):
        """Returns temperature healpy map for a simulation

            Args:
                idx: simulation index

            Returns:
                Temperature alm's 
                or Temperature healpy map if nside is given

        """
        assert 't' in self.cls_transf
        tlm = self.sims_cmb_len.get_sim_tlm(idx)
        hp.almxfl(tlm,self.cls_transf['t'],inplace=True)
        tlm +=  self.get_sim_tnoise(idx)
        if self.nside:
            return hp.alm2map(tlm, self.nside)
        return tlm 
    

    def fg_phases(self, mappa: np.ndarray, seed: int = 0):
        np.random.seed(seed)
        f = lambda z: np.exp(1j*np.random.uniform(0., 2.*np.pi, size = z.shape))
        return f(mappa)
    
    def randomize(self, xlm, idx = 0, shift = 100):
        if shift == 0:
            return xlm
        else:
            print("Randomizing maaap!!!", idx, shift)
            return xlm*self.fg_phases(xlm, idx+shift)
        

    def get_custom_sim_eblm(self, plm, clm, idx, shift):
        elm = self.sims_cmb_len.unlcmbs.get_sim_elm(self.sims_cmb_len.offset_index(idx, self.sims_cmb_len.offset_cmb[0], self.sims_cmb_len.offset_cmb[1]))
        blm = None if 'b' not in self.sims_cmb_len.fields else self.sims_cmb_len.unlcmbs.get_sim_blm(self.sims_cmb_len.offset_index(idx, self.sims_cmb_len.offset_cmb[0], self.sims_cmb_len.offset_cmb[1]))

        elm = self.randomize(elm, idx, shift)
        blm = self.randomize(blm, idx, shift)

        if not self.sims_cmb_len.zerolensing:
            dlm, dclm = plm.copy(), clm.copy()

            lmax_dlm = hp.Alm.getlmax(dlm.size, -1)
            mmax_dlm = lmax_dlm
            # potentials to deflection
            """p2d = np.sqrt(np.arange(lmax_dlm + 1, dtype=float) * np.arange(1, lmax_dlm + 2, dtype=float))
            p2d[:self.sims_cmb_len.lmin_dlm] = 0
            hp.almxfl(dlm, p2d, mmax_dlm, inplace=True)
            hp.almxfl(dclm, p2d, mmax_dlm, inplace=True)"""

            lmax_map = hp.Alm.getlmax(elm.size)
            Qlen, Ulen = self.sims_cmb_len.lens_module.alm2lenmap_spin([elm, blm], [dlm, dclm], 2, geometry = ('healpix', {'nside': self.sims_cmb_len.nside_lens}), epsilon = self.sims_cmb_len.epsilon, verbose = 0)
            elm, blm = hp.map2alm_spin([Qlen, Ulen], 2, lmax=lmax_map)
            del Qlen, Ulen

        elm = alm_copy(elm, self.sims_cmb_len.lmax)
        blm = alm_copy(blm, self.sims_cmb_len.lmax)

        return elm, blm
    
    def get_sim_pmap_custom(self, plm, clm, idx, shift, lmax):
        """Returns polarization healpy maps for a simulation

            Args:
                idx: simulation index

            Returns:
                Elm and Blm
                 or Q and U healpy maps if nside is given

        """
        assert 'e' in self.cls_transf
        assert 'b' in self.cls_transf

        elm, blm = self.get_custom_sim_eblm(plm, clm, idx, shift)
        hp.almxfl(elm, self.cls_transf['e'], inplace=True)
        hp.almxfl(blm, self.cls_transf['b'], inplace=True)
        elm += self.get_sim_enoise(idx)
        blm += self.get_sim_bnoise(idx)

        elm = alm_copy(elm, lmax)
        blm = alm_copy(blm, lmax)

        return np.array([elm, blm]) 

    def get_sim_pmap(self,idx):
        """Returns polarization healpy maps for a simulation

            Args:
                idx: simulation index

            Returns:
                Elm and Blm
                 or Q and U healpy maps if nside is given

        """
        assert 'e' in self.cls_transf
        assert 'b' in self.cls_transf

        elm = self.sims_cmb_len.get_sim_elm(idx)
        blm = self.sims_cmb_len.get_sim_blm(idx)
        hp.almxfl(elm, self.cls_transf['e'], inplace=True)
        hp.almxfl(blm, self.cls_transf['b'], inplace=True)
        elm += self.get_sim_enoise(idx)
        blm += self.get_sim_bnoise(idx)
        if self.nside is not None:
            return hp.alm2map_spin([elm,blm], self.nside, 2, hp.Alm.getlmax(elm.size))
        return elm, blm 

    def get_sim_tnoise(self,idx):
        assert 't' in self.cls_noise
        return hp.almxfl(self.phas.get_sim(idx, 0), np.sqrt(self.cls_noise['t']))

    def get_sim_enoise(self, idx):
        assert 'e' in self.cls_noise
        return hp.almxfl(self.phas.get_sim(idx, 1), np.sqrt(self.cls_noise['e']))

    def get_sim_bnoise(self, idx):
        assert 'b' in self.cls_noise
        return hp.almxfl(self.phas.get_sim(idx, 2), np.sqrt(self.cls_noise['b']))
