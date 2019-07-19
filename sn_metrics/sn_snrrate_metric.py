import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from sn_stackers.coadd_stacker import CoaddStacker
import healpy as hp
import numpy.lib.recfunctions as rf
import multiprocessing
import yaml
from scipy import interpolate
import os
import pandas as pd
import random


class SNSNRRateMetric(BaseMetric):
    """
    Measure SN-Signal-to-Noise Ratio as a function of time.
    Extract the detection rate from these measurements

    Parameters
    --------------
    lim_sn : str
       SN reference data (LC)
    names_ref : str
       names of the simulator used to generate reference files
    metricName : str, opt
      metric name
      Default : SNSNRMetric
    mjdCol : str, opt
      mjd column name
      Default : observationStartMJD,
    RaCol : str,opt
      Right Ascension column name
      Default : fieldRA
    DecCol : str,opt
      Declinaison column name
      Default : fieldDec
    filterCol : str,opt
       filter column name
       Default: filter
    m5Col : str, opt
       five-sigma depth column name
       Default : fiveSigmaDepth
    exptimeCol : str,opt
       exposure time column name
       Default : visitExposureTime
    nightCol : str,opt
       night column name
       Default : night
    obsidCol : str,opt
      observation id column name
      Default : observationId
    nexpCol : str,opt
      number of exposure column name
      Default : numExposures
     vistimeCol : str,opt
        visit time column name
        Default : visitTime
    coadd : bool,opt
       coaddition per night (and per band)
       Default : True
    season : list,opt
       list of seasons to process (float)
       Default : -1. (all seasons)
    shift : float,opt
       T0 of the SN to consider = MJD-shift
       Default: 10.
    z : float,opt
       redshift for the study
       Default : 0.01


    """

    def __init__(self, lim_sn, names_ref,
                 metricName='SNSNRMetric',
                 mjdCol='observationStartMJD', RaCol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', season=-1, coadd=True, z=0.01, bands='griz', snr_ref={}, **kwargs):

        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.RaCol = RaCol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.seasonCol = 'season'
        self.nightCol = nightCol
        self.obsidCol = obsidCol
        self.nexpCol = nexpCol
        self.vistimeCol = vistimeCol

        cols = [self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol]
        if coadd:
            cols += ['coadd']
        super(SNSNRRateMetric, self).__init__(
            col=cols, metricDtype='object', metricName=metricName, **kwargs)

        self.filterNames = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        # self.config = config
        self.blue_cutoff = 300.
        self.red_cutoff = 800.
        self.min_rf_phase = -20.
        self.max_rf_phase = 60.
        self.z = z
        self.season = season
        self.bands = bands

        self.lim_sn = lim_sn
        self.names_ref = names_ref
        self.snr_ref = snr_ref

    def run(self, dataSlice,  slicePoint=None):
        """
        goodFilters = np.in1d(dataSlice[self.filterCol], self.filterNames)
        dataSlice = dataSlice[goodFilters]
        if dataSlice.size == 0:
            return None
        dataSlice.sort(order=self.mjdCol)

        time = dataSlice[self.mjdCol]-dataSlice[self.mjdCol].min()
        """

        seasons = self.season
        if self.season == -1:
            seasons = np.unique(dataSlice[self.seasonCol])

        # get infos on seasons

        self.info_season = self.seasonInfo(dataSlice, seasons)
        fieldRA = np.mean(dataSlice[self.RaCol])
        fieldDec = np.mean(dataSlice[self.DecCol])
        if self.info_season is None:
            res = np.array(seasons, dtype=[('season', 'f8')])
            res = rf.append_fields(
                res, 'fieldRA', [fieldRA]*len(res), usemask=False)
            res = rf.append_fields(
                res, 'fieldDec', [fieldDec]*len(res), usemask=False)
            res = rf.append_fields(res, 'frac_obs_SNCosmo', [
                                   0.]*len(res), usemask=False)
            res = rf.append_fields(
                res, 'band', [self.bands]*len(res), usemask=False)

            return res

        seasons = self.info_season['season']
        snr_tot = None
        for band in self.bands:
            idx = dataSlice[self.filterCol] == band

            if len(dataSlice[idx]) > 0:
                snr_rate = self.snrRateSeason(
                    dataSlice[idx], seasons)  # SNR for observations
                if snr_rate is not None:
                    if snr_tot is None:
                        snr_tot = snr_rate
                    else:
                        snr_tot = np.concatenate((snr_tot, snr_rate))

        res = None
        if snr_tot is not None:
            res = self.groupInfo(snr_tot)

        print(res)
        return res

    def snrRateSeason(self, dataSlice, seasons, j=-1, output_q=None):
        """
        Estimate SNR for a dataSlice

        Parameters
        ---------
        dataSlice : array
          Array of observations
        seasons : list
          list of seasons to consider
        j : int, opt
          index for multiprocessing
          Default : -1
        output_q : multiprocessing queue, opt
          queue for multiprocessing
          Default : none

        Returns
        -----
        array with the following fields (all are of f8 type, except band which is of U1)
        SNR_name_ref:  Signal-To-Noise Ratio estimator
        season : season
        cadence: cadence of the season
        season_length: length of the season
        MJD_min: min MJD of the season
        DayMax: SN max luminosity MJD (aka T0)
        MJD:
        m5_eff: mean m5 of obs passing the min_phase, max_phase cut
        fieldRA: mean field RA
        fieldDec: mean field Dec
        band:  band
        m5: mean m5 (over the season)
        Nvisits: median number of visits (per observation) (over the season)
        ExposureTime: median exposure time (per observation) (over the season)
        """

        # check whether we do have data for the considered season for this dataSlice
        sel = None
        for season in seasons:
            idx = dataSlice[self.seasonCol] == season
            if sel is None:
                sel = dataSlice[idx]
            else:
                sel = np.concatenate((sel, dataSlice[idx]))

        if len(sel) == 0:
            return None

        # Get few infos: RA, Dec, Nvisits, m5, exptime
        fieldRA = np.mean(sel[self.RaCol])
        fieldDec = np.mean(sel[self.DecCol])
        Nvisits = np.median(sel[self.nexpCol]/2.)  # one visit = 2 exposures
        m5 = np.mean(sel[self.m5Col])
        exptime = np.median(sel[self.exptimeCol])
        sel.sort(order=self.mjdCol)
        mjds = sel[self.mjdCol]
        band = np.unique(sel[self.filterCol])[0]

        # Define MJDs to consider for metric estimation
        # basically: step of one day between MJDmin and MJDmax
        dates = None
        for val in self.info_season:
            if dates is None:
                dates = np.arange(val['MJD_min'], val['MJD_max'], 1.)
            else:
                dates = np.concatenate(
                    (dates, np.arange(val['MJD_min'], val['MJD_max'], 1.)))

        # SN  DayMax: dates-shift where shift is chosen in the input yaml file
        T0_lc = dates

        # for these DayMax, estimate the phases of LC points corresponding to the current dataSlice MJDs
        # diff_time = dates[:, np.newaxis]-mjds
        time_for_lc = mjds-T0_lc[:, None]

        phase = time_for_lc/(1.+self.z)  # phases of LC points
        # flag: select LC points only in between min_rf_phase and max_phase
        # phase_max = self.shift/(1.+self.z)
        flag = (phase >= self.min_rf_phase) & (phase <= self.max_rf_phase)

        # tile m5, MJDs, and seasons to estimate all fluxes and SNR at once
        m5_vals = np.tile(sel[self.m5Col], (len(time_for_lc), 1))
        mjd_vals = np.tile(sel[self.mjdCol], (len(time_for_lc), 1))
        season_vals = np.tile(sel[self.seasonCol], (len(time_for_lc), 1))

        # estimate fluxes and snr in SNR function
        fluxes_tot, snr = self.SNR(
            time_for_lc, band, m5_vals, flag, season_vals, T0_lc)

        # now save the results in a record array
        snr_nomask = np.ma.copy(snr)
        _, idx = np.unique(snr['season'], return_inverse=True)

        infos = self.info_season[idx]
        vars_info = ['cadence', 'season_length', 'MJD_min']
        snr = rf.append_fields(
            snr, vars_info, [infos[name] for name in vars_info])
        snr = rf.append_fields(snr, 'daymax', T0_lc)
        snr = rf.append_fields(snr, 'MJD', dates)
        snr = rf.append_fields(snr, 'm5_eff', np.mean(
            np.ma.array(m5_vals, mask=~flag), axis=1))
        global_info = [(fieldRA, fieldDec, band, m5,
                        Nvisits, exptime)]*len(snr)
        names = ['fieldRA', 'fieldDec', 'band',
                 'm5', 'Nvisits', 'ExposureTime']
        global_info = np.rec.fromrecords(global_info, names=names)
        snr = rf.append_fields(
            snr, names, [global_info[name] for name in names])

        if output_q is not None:
            output_q.put({j: snr})
        else:
            return snr

    def seasonInfo(self, dataSlice, seasons):
        """
        Get info on seasons for each dataSlice
        Parameters
        -----
        dataSlice : array
          array of observations
        Returns
        -----
        recordarray with the following fields:
        season, cadence, season_length, MJDmin, MJDmax
        """

        rv = []
        for season in seasons:
            idx = (dataSlice[self.seasonCol] == season)
            slice_sel = dataSlice[idx]
            if len(slice_sel) < 5:
                continue
            slice_sel.sort(order=self.mjdCol)
            mjds_season = slice_sel[self.mjdCol]
            cadence = np.mean(mjds_season[1:]-mjds_season[:-1])
            mjd_min = np.min(mjds_season)
            mjd_max = np.max(mjds_season)
            season_length = mjd_max-mjd_min
            # Nvisits = np.median(slice_sel[self.nexpCol])
            Nvisits = len(slice_sel)
            rv.append((float(season), cadence, season_length,
                       mjd_min, mjd_max, Nvisits))

        info_season = None
        if len(rv) > 0:
            info_season = np.rec.fromrecords(
                rv, names=['season', 'cadence', 'season_length', 'MJD_min', 'MJD_max', 'Nvisits'])

        return info_season

    def SNR(self, time_lc, band, m5_vals, flag, season_vals, T0_lc):
        """
        Estimate SNR vs time
        Parameters
        -----------
        time_lc :
        m5_vals : list(float)
           five-sigme depth values
        flag : array(bool)
          flag to be applied (example: selection from phase cut)
        season_vals : array(float)
          season values
        T0_lc : array(float)
           array of T0 for supernovae
        Returns
        -----
        fluxes_tot : list(float)
         list of (interpolated) fluxes
        snr_tab : array with the following fields:
          snr_name_ref (float) : Signal-to-Noise values
          season (float) : season num.
        """
        seasons = np.ma.array(season_vals, mask=~flag)

        fluxes_tot = {}
        snr_tab = None

        for ib, name in enumerate(self.names_ref):
            fluxes = self.lim_sn[band].fluxes[ib](np.copy(time_lc))
            if name not in fluxes_tot.keys():
                fluxes_tot[name] = fluxes
            else:
                fluxes_tot[name] = np.concatenate((fluxes_tot[name], fluxes))

            flux_5sigma = self.lim_sn[band].mag_to_flux[ib](np.copy(m5_vals))
            snr = fluxes**2/flux_5sigma**2
            snr_season = 5.*np.sqrt(np.sum(snr*flag, axis=1))
            if snr_tab is None:
                snr_tab = np.asarray(np.copy(snr_season), dtype=[
                    ('SNR_'+name, 'f8')])
            else:
                snr_tab = rf.append_fields(
                    snr_tab, 'SNR_'+name, np.copy(snr_season))
        snr_tab = rf.append_fields(snr_tab, 'season', np.mean(seasons, axis=1))

        # check if any masked value remaining
        # this would corrspond to case where no obs point has been selected
        # ie no points with phase in [phase_min,phase_max]
        # this happens when internight gaps are large (typically larger than shift)
        idmask = np.where(snr_tab.mask)
        if len(idmask) > 0:
            tofill = np.copy(snr_tab['season'])
            season_recover = self.getSeason(
                T0_lc[np.where(snr_tab.mask)])
            tofill[idmask] = season_recover
            snr_tab = np.ma.filled(snr_tab, fill_value=tofill)

        return fluxes_tot, snr_tab

    def getSeason(self, T0):
        """
        Estimate the seasons corresponding to T0 values
        Parameters
        -------
        T0 : list(float)
           set of T0 values
        Returns
        -----
        list (float) of corresponding seasons
        """

        diff_min = T0[:, None]-self.info_season['MJD_min']
        diff_max = -T0[:, None]+self.info_season['MJD_max']

        seasons = np.tile(self.info_season['season'], (len(diff_min), 1))
        flag = (diff_min >= 0) & (diff_max >= 0)
        seasons = np.ma.array(seasons, mask=~flag)

        return np.mean(seasons, axis=1)

    def detectingFraction(self, snr_obs, snr_fakes):
        """
        Estimate the time fraction(per season) for which
        snr_obs > snr_fakes = detection rate
        For regular cadences one should get a result close to 1
        Parameters
        -------
        snr_obs : array
         array estimated using snrSeason(observations)
         snr_fakes: array
           array estimated using snrSeason(fakes)
        Returns
        -----
        record array with the following fields:
          fieldRA (float)
          fieldDec (float)
          season (float)
         band (str)
         frac_obs_name_ref (float)
        """

        ra = np.mean(snr_obs['fieldRA'])
        dec = np.mean(snr_obs['fieldDec'])
        # band = np.unique(snr_obs['band'])[0]

        rtot = []

        for season in np.unique(snr_obs['season']):
            idx = snr_obs['season'] == season
            sel_obs = snr_obs[idx]
            idxb = snr_fakes['season'] == season
            sel_fakes = snr_fakes[idxb]
            sel_obs.sort(order='MJD')
            sel_fakes.sort(order='MJD')
            r = [ra, dec, season, band]
            names = [self.RaCol, self.DecCol, 'season', 'band']
            for sim in self.names_ref:
                fakes = interpolate.interp1d(
                    sel_fakes['MJD'], sel_fakes['SNR_'+sim])
                obs = interpolate.interp1d(sel_obs['MJD'], sel_obs['SNR_'+sim])
                mjd_min = np.max(
                    [np.min(sel_obs['MJD']), np.min(sel_fakes['MJD'])])
                mjd_max = np.min(
                    [np.max(sel_obs['MJD']), np.max(sel_fakes['MJD'])])
                mjd = np.arange(mjd_min, mjd_max, 1.)

                diff_res = obs(mjd)-fakes(mjd)
                idx = diff_res >= 0
                r += [len(diff_res[idx])/len(diff_res)]
                names += ['frac_obs_'+sim]
            rtot.append(tuple(r))
        return np.rec.fromrecords(rtot, names=names)

    def groupInfo(self, snr):

        # we have to make group so let us move to panda
        df = pd.DataFrame.from_records(snr)
        # these data are supposed to correspond to a given filter
        # so we have to group them according to (daymax,season)

        r = []
        for key, grp_season in df.groupby(['season']):
            # print(grp_season.columns)

            select = grp_season.groupby('daymax').apply(
                self.selectSNR).value_counts()

            values = select.keys().tolist()
            counts = select.tolist()
            nGood = 0.
            if True in values:
                nGood = counts[values.index(True)]
            r.append((np.mean(grp_season['season']), np.mean(grp_season['fieldRA']), np.mean(
                grp_season['fieldDec']), nGood/len(grp_season), self.bands))
        return np.rec.fromrecords(r, names=['season', 'fieldRA', 'fieldDec', 'frac_obs_SNCosmo', 'band'])

    def selectSNR(self, grp):

        res = True
        for band in grp['band']:
            idx = grp['band'] == band
            print(band, grp['SNR_SNCosmo'][idx], self.snr_ref[band])
            res &= (grp['SNR_SNCosmo'][idx] >= self.snr_ref[band])

        print(test)
        return res
