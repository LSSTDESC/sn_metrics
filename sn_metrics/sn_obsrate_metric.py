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
from sn_tools.sn_obs import season as seasonCalc
from sn_tools.sn_obs import getPix


class SNObsRateMetric(BaseMetric):
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
      metric name (default: SNSNRMetric)
    mjdCol : str, opt
      mjd column name (default: observationStartMJD)
    RACol : str,opt
      Right Ascension column name (default: fieldRA)
    DecCol : str,opt
      Declinaison column name (default: fieldDec)
    filterCol : str,opt
       filter column name (default: filter)
    m5Col : str, opt
       five-sigma depth column name (default: fiveSigmaDepth)
    exptimeCol : str,opt
       exposure time column name (default: visitExposureTime)
    nightCol : str,opt
       night column name (default: night)
    obsidCol : str,opt
      observation id column name (default: observationId)
    nexpCol : str,opt
      number of exposure column name (default: numExposures)
    vistimeCol : str,opt
        visit time column name(default: visitTime)
    coadd : bool,opt
       coaddition per night (and per band) (default: True)
    season : list,opt
       list of seasons to process (float) (default: -1. = all seasons)
    z : float,opt
       redshift for the study(default: 0.1)
    bands: str, opt
       bands to consider (default: griz)
    nside: int, opt
       nside parameter for healpix (default: 64)
    snr_ref: dict, opt
       snr dict reference values (default: {})


    """

    def __init__(self, lim_sn, names_ref,
                 metricName='SNObsRateMetric',
                 mjdCol='observationStartMJD', RACol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', visitExposureTimeCol='visitExposureTime',
                 season=-1, coadd=True, z=0.1, bands='griz', nside=64, snr_ref={},
                 verbose=False, **kwargs):

        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.RACol = RACol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.seasonCol = 'season'
        self.nightCol = nightCol
        self.obsidCol = obsidCol
        self.nexpCol = nexpCol
        self.vistimeCol = vistimeCol
        self.visitExposureTimeCol = visitExposureTimeCol
        self.daystep = 1.
        self.verbose = verbose

        cols = [self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol, self.visitExposureTimeCol]
        self.stacker = None

        if coadd:
            #cols += ['coadd']
            self.stacker = CoaddStacker(mjdCol=self.mjdCol,
                                        RACol=self.RACol, DecCol=self.DecCol,
                                        m5Col=self.m5Col, nightCol=self.nightCol,
                                        filterCol=self.filterCol, numExposuresCol=self.nexpCol,
                                        visitTimeCol=self.vistimeCol, visitExposureTimeCol=self.visitExposureTimeCol)
        super(SNObsRateMetric, self).__init__(
            col=cols, metricDtype='object', metricName=metricName, **kwargs)

        self.filterNames = np.array([b for b in bands])
        # self.config = config
        self.blue_cutoff = 300.
        self.red_cutoff = 800.
        self.min_rf_phase = -20.
        self.max_rf_phase = 60.
        self.z = z
        self.season = season
        self.bands = bands
        self.min_rf_phase_qual = -15.
        self.max_rf_phase_qual = 25.
        self.shift_phmin = -(1.+self.z)*self.min_rf_phase_qual
        self.shift_phmax = -(1.+self.z)*self.max_rf_phase_qual

        # self.shift_phmin = 0.
        # self.shift_phmax = 0.
        self.lim_sn = lim_sn
        self.names_ref = names_ref
        self.snr_ref = snr_ref
        self.nside = nside
        self.verbose = verbose

    def run(self, dataSlice,  slicePoint=None):
        """
        Run method for the metric.

        Parameters
        ---------------
        dataSlice: numpy array
          data to process
        slicePoint: ,opt
          (default: None)

        Returns
        -----------
        final_resu: numpy record array with the following:

          season: season num

          fieldRA: median(fieldRA) of the dataSlice

          fieldDec: median(fieldDec) of the dataSlice

          pixRA: RA of the pixel corresponding to (fieldRA,fieldDec)

          pixDec: Dec of the pixel corresponding to (fieldRA,fieldDec)

          healpixId: Id of the pixel corresponding to (fieldRA,fieldDec)

          frac_obs_band: observing fraction in the band.

        """
        if self.verbose:
            print('Running metric')

        goodFilters = np.in1d(dataSlice[self.filterCol], self.filterNames)
        dataSlice = dataSlice[goodFilters]
        if dataSlice.size == 0:
            if self.verbose:
                print('no data selected')

            return None
        dataSlice.sort(order=self.mjdCol)

        """
        time = dataSlice[self.mjdCol]-dataSlice[self.mjdCol].min()
        """

        # get seasons
        if 'season' in dataSlice.dtype.names:
            dataSlice = rf.drop_fields(dataSlice, ['season'])
            dataSlice = seasonCalc(dataSlice)

        # get fieldRA, fieldDec and healpix infos
        self.fieldRA = np.mean(dataSlice[self.RACol])
        self.fieldDec = np.mean(dataSlice[self.DecCol])

        if 'pixRA' not in dataSlice.dtype.names:
            self.healpixID, self.pixRA, self.pixDec = getPix(self.nside,
                                                             np.mean(
                                                                 dataSlice[self.RACol]),
                                                             np.mean(dataSlice[self.DecCol]))

            dataSlice = rf.append_fields(dataSlice, 'healpixID', [
                                         self.healpixID]*len(dataSlice))
            dataSlice = rf.append_fields(
                dataSlice, 'pixRA', [self.pixRA]*len(dataSlice))
            dataSlice = rf.append_fields(
                dataSlice, 'pixDec', [self.pixDec]*len(dataSlice))

        else:
            self.pixRA = np.mean(dataSlice['pixRA'])
            self.pixDec = np.mean(dataSlice['pixDec'])
            self.healpixID = np.mean(dataSlice['healpixID'])

        if self.stacker is not None:
            dataSlice = self.stacker._run(dataSlice)
            if self.verbose:
                print('after stacking', dataSlice[[
                      self.filterCol, self.m5Col, self.visitExposureTimeCol]])
        seasons = self.season
        if self.season == -1:
            seasons = np.unique(dataSlice[self.seasonCol])

        # print(np.unique(dataSlice[self.filterCol]),
        #      np.unique(dataSlice[self.seasonCol]))
        # get infos on seasons

        self.info_season = self.seasonInfo(dataSlice, seasons)

        # print('info seasons', self.info_season)
        if self.info_season is None:
            r = []
            for seas in seasons:
                r.append((int(seas), self.fieldRA, self.fieldDec,
                          self.pixRA, self.pixDec, self.healpixID,
                          0., self.bands))

            return np.rec.fromrecords(r, names=[
                'season', 'fieldRA', 'fieldDec',
                'pixRA', 'pixDec', 'healpixId',
                'frac_obs_{}'.format(self.names_ref[0]), 'band'])

        seasons = self.info_season['season']

        snr_tot = None
        for band in self.bands:
            idx = dataSlice[self.filterCol] == band
            if len(dataSlice[idx]) > 0:
                snr_rate = self.snrSeason(
                    dataSlice[idx], seasons)  # SNR for observations
                if snr_rate is not None:
                    if snr_tot is None:
                        snr_tot = snr_rate
                    else:
                        snr_tot = np.concatenate((snr_tot, snr_rate))

        res = None

        if snr_tot is not None:
            res = self.groupInfo(snr_tot)

        r = []
        for val in self.info_season:
            season = val['season']
            dates = np.arange(val['MJD_min']+self.shift_phmin,
                              val['MJD_max']+self.shift_phmax, self.daystep)
            nsn = len(dates)
            rat = 0.
            if res is not None:
                idx = np.abs(res['season']-season) < 1.e-5
                nsn_snrcut = len(res[idx])

                if nsn > 0:
                    rat = nsn_snrcut/nsn

            else:
                rat = 0.
            r.append((season, self.fieldRA, self.fieldDec,
                      self.pixRA, self.pixDec, self.healpixID,
                      rat, self.bands))

        final_resu = np.rec.fromrecords(r, names=[
            'season', 'fieldRA', 'fieldDec',
            'pixRA', 'pixDec', 'healpixId',
            'frac_obs_{}'.format(self.names_ref[0]), 'band'])

        if self.verbose:
            print('Processed', final_resu)
        return final_resu

    def snrSeason(self, dataSlice, seasons, j=-1, output_q=None):
        """
        Estimate SNR values for a dataSlice.
        Apply a selection to have only SNRs>= SNR_ref

        Parameters
        ---------------
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
        ----------
        array with the following fields (all are of f8 type, except band which is of U1)

         SNR_name_ref:  Signal-To-Noise Ratio estimator

         season : season

         cadence: cadence of the season

         season_length: length of the season

         MJD_min: min MJD of the season

         daymax: SN max luminosity MJD (aka T0)

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

        sel = np.copy(sel)
        # Get few infos: Nvisits, m5, exptime
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
            min_mjd = val['MJD_min']+self.shift_phmin
            max_mjd = val['MJD_max']+self.shift_phmax
            deltaT = max_mjd-min_mjd
            if deltaT >= 5.:
                mjds_T0 = np.arange(min_mjd, max_mjd, self.daystep)
                if dates is None:
                    dates = mjds_T0
                else:
                    dates = np.concatenate((dates, mjds_T0))

        if dates is None:
            return None

        # SN  DayMax: dates-shift where shift is chosen in the input yaml file
        T0_lc = dates

        #T0_lc = np.array([np.mean(dates)])
        # for these DayMax, estimate the phases of LC points corresponding to the current dataSlice MJDs
        # diff_time = dates[:, np.newaxis]-mjds
        time_for_lc = mjds-T0_lc[:, None]

        phase = time_for_lc/(1.+self.z)  # phases of LC points
        if self.verbose:
            print('time for lc', time_for_lc)
            print('phase for lc', phase)
        # flag: select LC points only in between min_rf_phase and max_phase
        # phase_max = self.shift/(1.+self.z)
        flag = (phase >= self.min_rf_phase) & (phase <= self.max_rf_phase)
        # tile m5, MJDs, and seasons to estimate all fluxes and SNR at once
        m5_vals = np.tile(sel[self.m5Col], (len(time_for_lc), 1))
        mjd_vals = np.tile(sel[self.mjdCol], (len(time_for_lc), 1))
        season_vals = np.tile(sel[self.seasonCol], (len(time_for_lc), 1))

        # estimate fluxes and snr in SNR function
        fluxes_tot, snr = self.calcSNR(
            time_for_lc, band, m5_vals, flag, season_vals, T0_lc)

        if self.verbose:
            print('m5', m5_vals)
            print('fluxes', fluxes_tot)
            print('snr', snr)

        # now save the results in a record array
        snr_nomask = np.ma.copy(snr)
        _, idx = np.unique(snr['season'], return_inverse=True)

        # print('seas', np.unique(snr['season']))
        # print(band, idx)
        infos = self.info_season[idx]

        """
        vars_info = ['cadence', 'season_length', 'MJD_min']
        snr = rf.append_fields(
            snr, vars_info, [infos[name] for name in vars_info])
        """
        snr = rf.append_fields(snr, 'daymax', T0_lc)

        #snr = rf.append_fields(snr, 'MJD', dates)
        snr = rf.append_fields(snr, 'm5_mean', np.mean(
            np.ma.array(m5_vals, mask=~flag), axis=1))

        if len(snr) > 0:
            # global_list = [(self.fieldRA, self.fieldDec, band, m5,
            #                Nvisits, exptime)]*len(snr)
            # names = ['fieldRA', 'fieldDec', 'band',
            #         'm5', 'Nvisits', 'ExposureTime']
            global_list = [(self.fieldRA, self.fieldDec, band,
                            )]*len(snr)
            names = ['fieldRA', 'fieldDec', 'band']

            global_info = np.rec.fromrecords(global_list, names=names)
            snr = rf.append_fields(
                snr, names, [global_info[name] for name in names])

            # print('there pal', self.snr_ref[band],snr[['season','band', 'daymax',
            #                        'SNR_{}'.format(self.names_ref[0])]])
            idx = snr['SNR_{}'.format(self.names_ref[0])] >= self.snr_ref[band]
            snr = np.copy(snr[idx])

        else:
            snr = None

        if output_q is not None:
            output_q.put({j: snr})
        else:
            return snr

    def seasonInfo(self, dataSlice, seasons):
        """
        Get info on seasons for each dataSlice

        Parameters
        ---------------
        dataSlice : array
          array of observations

        Returns
        ----------
        recordarray with the following fields:
        season, cadence, season_length, MJDmin, MJDmax
        """

        rv = []
        for season in seasons:
            idx = np.abs(dataSlice[self.seasonCol]-season) < 1.e-5
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
            rv.append((int(season), cadence, season_length,
                       mjd_min, mjd_max, Nvisits))

        info_season = None
        if len(rv) > 0:
            info_season = np.rec.fromrecords(
                rv, names=['season', 'cadence', 'season_length', 'MJD_min', 'MJD_max', 'Nvisits'])

        return info_season

    def calcSNR(self, time_lc, band, m5_vals, flag, season_vals, T0_lc):
        """
        Estimate SNR vs time.
        The SNR is estimated in the backgound dominating regime:

        .. math::

           (a + b)^2 = a^2 + 2ab + b^2 

           SNR=\sqrt{\sum(flux/\sigma_{flux})}

        with 

        .. math::

           \sigma_{flux} = \sigma_5 

           \sigma_5 = flux_5/5

           log(flux_5) = 10^{-0.4*(m_5-zp)}

        :math:`m_5` :fiveSigmaDepth (from the scheduler)

        Parameters
        ----------------
        time_lc : numpy array(floats)
          differences between mjds and T0
        m5_vals : list(float)
           five-sigme depth values
        flag : array(bool)
          flag to be applied (example: selection from phase cut)
        season_vals : array(float)
          season values
        T0_lc : array(float)
           array of T0 for supernovae

        Returns
        ----------
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
            if self.verbose:
                print('times', time_lc)
                print('fluxes', fluxes)
                print('5sigma fluxes', flux_5sigma)
                print('snr', snr*flag, np.sum(snr*flag),
                      np.sqrt(np.sum(snr*flag)))
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
        ---------------
        T0 : list(float)
           set of T0 values

        Returns
        -----------
        list (float) of corresponding seasons
        """

        diff_min = T0[:, None]-self.info_season['MJD_min']
        diff_max = -T0[:, None]+self.info_season['MJD_max']

        seasons = np.tile(self.info_season['season'], (len(diff_min), 1))
        flag = (diff_min >= 0) & (diff_max >= 0)
        seasons = np.ma.array(seasons, mask=~flag)

        return np.mean(seasons, axis=1)

    def groupInfo(self, snr):
        """
        Method to filter a numpy array.
        the input numpy array is grouped by (season,daymax)
        Only groups with len(grp)==len(self.bands) are selected


        Parameters
        ---------------
        snr: numpy array with the following cols
          SNR: SNR value 

          season : season

          m5_mean : mean m5 for obs

          fieldRA : RA of the field

          fieldDec : Dec of the field

          band : band

        Returns
        ----------
        record array with 'season', 'fieldRA', 'fieldDec', 'daymax' as fields.

        """

        # we have to make group so let us move to panda
        df = pd.DataFrame.from_records(snr)
        # these data are supposed to correspond to a given filter
        # so we have to group them according to (daymax,season)
        df.season = df.season.astype(int)
        r = []
        for key, grp_season in df.groupby(['season', 'daymax']):
            if len(grp_season) == len(self.bands):
                season = np.mean(grp_season['season'])
                daymax = np.mean(grp_season['daymax'])
                r.append((season, self.fieldRA, self.fieldDec, daymax))

        if len(r) > 0:
            return np.rec.fromrecords(r, names=['season', 'fieldRA', 'fieldDec', 'daymax'])
        else:
            return None
