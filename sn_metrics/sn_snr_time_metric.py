import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from sn_stackers.coadd_stacker import CoaddStacker
import multiprocessing
import yaml
import os
from sn_tools.sn_calcFast import LCfast, CovColor
from sn_tools.sn_telescope import Telescope
from astropy.table import Table, vstack, Column
import time
import pandas as pd
from scipy.interpolate import interp1d
from sn_tools.sn_rate import SN_Rate
from functools import wraps
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from sn_metrics.sn_plot_live import Plot_NSN_metric
from sn_tools.sn_utils import multiproc
from random import shuffle


class SNSNRTIMEMetric(BaseMetric):
    """
    Measure zlim of type Ia supernovae.

    Parameters
    --------------
    lc_reference : dict
       SN reference data (LC)(key: (x1,color); vals: sn_tools.sn_utils.GetReference
    metricName : str, opt
      metric name (default : SNSNRMetric)
    mjdCol : str, opt
      mjd column name (default : observationStartMJD)
    RACol : str,opt
      Right Ascension column name (default : fieldRA)
    DecCol : str,opt
      Declinaison column name (default : fieldDec)
    filterCol : str,opt
       filter column name (default: filter)
    m5Col : str, opt
       five-sigma depth column name (default : fiveSigmaDepth)
    exptimeCol : str,opt
       exposure time column name (default : visitExposureTime)
    nightCol : str,opt
       night column name (default : night)
    obsidCol : str,opt
      observation id column name (default : observationId)
    nexpCol : str,opt
      number of exposure column name (default : numExposures)
     vistimeCol : str,opt
        visit time column name (default : visitTime)
    season : list,opt
       list of seasons to process (float)(default: -1 = all seasons)
    coadd : bool,opt
       coaddition per night (and per band) (default : True)
    zmin : float,opt
       min redshift for the study (default: 0.0)
    zmax : float,opt
       max redshift for the study (default: 1.2)
    pixArea: float, opt
       pixel area (default: 9.6)
    outputType: str, opt
      output type requested (defauls: zlims)
    verbose: bool,opt
      verbose mode (default: False)
    ploteffi: bool, opt
      display efficiencies during processing (default:False)
    proxy_level: int, opt
     proxy level for the processing (default: 0)
    n_bef: int, opt
      number of LC points LC before T0 (default:5)
    n_aft: int, opt
      number of LC points after T0 (default: 10)
     snr_min: float, opt
       minimal SNR of LC points (default: 5.0)
     n_phase_min: int, opt
       number of LC points with phase<= -5(default:1)
    n_phase_max: int, opt
      number of LC points with phase>= 20 (default: 1)
    x1_color_dist: ,opt
     (x1,color) distribution (default: None)
    lightOutput: bool, opt
      output level of information (light or more) (default:True)
    T0s: str,opt
       T0 values for the processing (default: all)
    zlim_coeff: float, opt
      rules estimation of the redshift limit (default: 0.95)
      if >0: zlim correspond to the zlim_coeff fraction of SN with z<zlim
      if <0: zlim is estimated as the redshift corresponding to a decrease of efficiency
    ebvofMV: float, opt
      E(B-V) (default: -1 : estimated from dust map)
    obsstat: bool, opt
      to get info on observing conditions (default: True)
    bands: str, opt
      bands to consider (default: grizy)
    fig_for_movie: bool, opt
      to save figures to make a movie showing how the metric is estimated
    """

    def __init__(self, lc_reference, dustcorr,
                 metricName='SNSNRTIMEMetric',
                 mjdCol='observationStartMJD', RACol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', season=[-1], coadd=True, zmin=0.0, zmax=1.,
                 pixArea=9.6, outputType='zlims', verbose=False, timer=False, ploteffi=False, proxy_level=0,
                 n_bef=5, n_aft=10, snr_min=5., n_phase_min=1, n_phase_max=1, errmodrel=0.1, sigmaC=0.04,
                 x1_color_dist=None, lightOutput=True, T0s='all', zlim_coeff=0.95,
                 ebvofMW=-1., obsstat=True, bands='grizy', fig_for_movie=False, templateLC={}, dbName='', **kwargs):

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
        self.pixArea = pixArea
        self.pixArea = 9.6
        self.ploteffi = ploteffi
        self.x1_color_dist = x1_color_dist
        self.T0s = T0s
        self.zlim_coeff = zlim_coeff
        self.ebvofMW = ebvofMW
        self.bands = bands
        self.fig_for_movie = fig_for_movie

        cols = [self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol]

        self.stacker = None
        if coadd:
            cols += ['coadd']
            self.stacker = CoaddStacker(mjdCol=self.mjdCol, RACol=self.RACol, DecCol=self.DecCol, m5Col=self.m5Col, nightCol=self.nightCol,
                                        filterCol=self.filterCol, numExposuresCol=self.nexpCol, visitTimeCol=self.vistimeCol, visitExposureTimeCol='visitExposureTime')
        super(SNSNRTIMEMetric, self).__init__(
            col=cols, metricDtype='object', metricName=metricName, **kwargs)

        self.season = season

        telescope = Telescope(airmass=1.2)

        # LC selection parameters
        self.n_bef = n_bef  # nb points before peak
        self.n_aft = n_aft  # nb points after peak
        self.snr_min = snr_min  # SNR cut for points before/after peak
        self.n_phase_min = n_phase_min  # nb of point with phase <=-5
        self.n_phase_max = n_phase_max  # nb of points with phase >=20
        self.errmodrel = errmodrel  # relative error model for g and r bands
        self.sigmaC = sigmaC

        # print('selection', self.n_bef, self.n_aft,
        #      self.n_phase_min, self.n_phase_max)
        self.lcFast = {}

        # loading reference LC files
        for key, vals in lc_reference.items():
            self.lcFast[key] = LCfast(vals, dustcorr[key], key[0], key[1], telescope,
                                      self.mjdCol, self.RACol, self.DecCol,
                                      self.filterCol, self.exptimeCol,
                                      self.m5Col, self.seasonCol, self.nexpCol,
                                      self.snr_min, lightOutput=lightOutput)

        # loading parameters
        self.zmin = zmin  # zmin for the study
        self.zmax = zmax  # zmax for the study
        self.zstep = 0.05  # zstep
        # get redshift range for processing
        zrange = list(np.arange(self.zmin, self.zmax+self.zstep, self.zstep))
        if zrange[0] < 1.e-6:
            zrange[0] = 0.01

        self.zrange = np.unique(zrange)

        self.daymaxStep = 2.  # daymax step
        self.min_rf_phase = -20.  # min ref phase for LC points selection
        self.max_rf_phase = 60.  # max ref phase for LC points selection

        self.min_rf_phase_qual = -15.  # min ref phase for bounds effects
        self.max_rf_phase_qual = 30.  # max ref phase for bounds effects

        # snrate
        self.rateSN = SN_Rate(H0=70., Om0=0.3,
                              min_rf_phase=self.min_rf_phase_qual, max_rf_phase=self.max_rf_phase_qual)
        # snrate
        # rateSN = SN_Rate(H0=70., Om0=0.3,
        #                 min_rf_phase=self.min_rf_phase_qual, max_rf_phase=self.max_rf_phase_qual)
        """
        self.duration_ref = 110.
        grp = pd.DataFrame([1800],columns=['MJD_min'])
        grp['MJD_max'] = grp['MJD_min']+self.duration_ref
        dur_z = self.duration_z(grp)
        nsn = self.nsn_from_rate(dur_z)
        self.nsn_expected = interp1d(self.zrange, nsn['nsn_expected'], kind='linear',
                                     bounds_error=False, fill_value=0)
        """
        # verbose mode - useful for debug and code performance estimation
        self.verbose = verbose
        self.timer = timer

        # status of the pixel after processing
        self.status = dict(
            zip(['ok', 'effi', 'season_length', 'nosn', 'simu_parameters', 'low_effi'], [1, -1, -2, -3, -4, -5]))

        # supernovae parameters for fisher estimation
        self.params = ['x0', 'x1', 'daymax', 'color']

        # output type and proxy level
        self.outputType = outputType  # this is to choose the output: lc, sn, effi, zlims
        self.proxy_level = proxy_level  # proxy level chosen by the user: 0, 1, 2

        self.obsstat = obsstat
        self.bandstat = None
        if self.obsstat:
            self.bandstat = ['u', 'g', 'r', 'i', 'z', 'y', 'gr',
                             'gi', 'gz', 'iz', 'uu', 'gg', 'rr', 'ii', 'zz', 'yy']
            """
            bands = 'grizy'
            for ba in bands:
                self.bandstat.append(ba)
                for bb in bands:
                    self.bandstat.append(
                        ''.join(sorted('{}{}'.format(ba, bb))))
                    for bc in bands:
                        self.bandstat.append(
                            ''.join(sorted('{}{}{}'.format(ba, bb, bc))))
            """
        self.plotter = None
        if self.ploteffi and self.fig_for_movie:
            self.plotter = Plot_NSN_metric(self.snr_min, self.n_bef, self.n_aft,
                                           self.n_phase_min, self.n_phase_max, self.errmodrel,
                                           self.mjdCol, self.m5Col, self.filterCol, self.nightCol,
                                           templateLC=templateLC, dbName=dbName)

    def run(self, dataSlice,  slicePoint=None):
        """
        Run method of the metric

        Parameters
        --------------
        dataSlice: numpy array
          data to process (scheduler simulations)
        slicePoint: bool, opt
          (default: None)

        Returns
        ----------

        """
        obs = pd.DataFrame(np.copy(dataSlice))
        #obs = obs[obs['season'].isin(seasons)]
        sn = self.simul_sn(obs)
        print('simulation', sn)
        autogen = True
        frac = 0.5
        self.cadence_obs = self.cadence(dataSlice)
        print('cad before', self.cadence_obs)

        all_nights = np.unique(dataSlice['night'])

        all_nights.sort()
        dataSlice.sort(order='night')
        all_nights_noborder = all_nights[1:-1]

        # dataSlice = np.random.choice(dataSlice, int(
        #    frac*len(dataSlice)), replace=False)
        nights = np.random.choice(all_nights_noborder, int(
            frac*len(all_nights_noborder)), replace=False)

        nights = list(nights)
        # add first and last night
        nights += [dataSlice['night'][0], dataSlice['night'][-1]]
        goodNights = np.in1d(dataSlice['night'], nights)
        dataSlice = dataSlice[goodNights]

        print('processing pixel', np.unique(dataSlice['healpixID']))

        self.pixRA = np.unique(dataSlice['pixRA'])[0]
        self.pixDec = np.unique(dataSlice['pixDec'])[0]
        self.healpixID = np.unique(dataSlice['healpixID'])[0]

        # select observations filter
        goodFilters = np.in1d(dataSlice[self.filterCol], list(self.bands))
        dataSlice = dataSlice[goodFilters]

        # get the season durations
        seasons, dur_z = self.season_length(self.season, dataSlice)

        if not seasons or dur_z.empty:
            df = self.resError(self.status['season_length'])
            return df

        # select observations corresponding to seasons
        obs = pd.DataFrame(np.copy(dataSlice))
        obs = obs[obs['season'].isin(seasons)]

        # get SNR time here
        mjd_min = obs['observationStartMJD'].min()
        mjd_max = obs['observationStartMJD'].max()

        if autogen:
            nights = obs['night'].unique()
            night_min = nights.min()
            idx = obs['night'] == night_min
            obs = obs[idx]

        # analyze one season only
        ido = obs['season'] == 1
        obs = obs[ido]
        """
        res, obs_gen = obs.groupby(['season']).apply(
            lambda x: self.get_SNRTime(x, mjd_min, mjd_max, autogen=autogen))
        """

        res, obs_gen = self.get_SNRTime(
            obs, mjd_min, mjd_max, autogen=autogen, downtimes=True)
        if autogen:
            dataSlice = obs_gen
        sn = self.simul_sn(dataSlice)
        print('simulation new', sn)
        import matplotlib.pyplot as plt
        band = 'z'
        fig, ax = plt.subplots()
        tpl = 'SNR_{}'.format(band)
        ax.plot(res['MJD_obs'], res[tpl], 'ko')
        axb = ax.twinx()
        idx = dataSlice['filter'] == band
        sel_obs = dataSlice[idx]
        axb.plot(sel_obs['observationStartMJD'],
                 sel_obs['fiveSigmaDepth'], 'r*')
        print('hello', np.mean(res[tpl]), np.std(res[tpl]))
        plt.show()

        band = 'z'
        fig, ax = plt.subplots()
        ax.plot(res['MJD_obs'], res['nsn']/res['nsn_expected'], 'ko')

        axb = ax.twinx()
        idx = dataSlice['filter'] == band
        sel_obs = dataSlice[idx]
        axb.plot(sel_obs['observationStartMJD'],
                 sel_obs['fiveSigmaDepth'], 'r*')

        fig, ax = plt.subplots()
        ax.plot(res['cadence'], res['nsn']/res['nsn_expected'], 'ko')
        axb = ax.twinx()
        axb.plot(res['cadence'], res['Nepochs'], 'ko')
        plt.show()

        return metricValues

    def cadence(self, data, op=np.median):

        data.sort(order='night')
        idx = data['filter'] == 'z'
        seldata = data[idx]

        cad = op(np.diff(seldata['night']))

        return cad

    def add_obs(self, obs, night, MJD, template_obs):

        template_obs[self.nightCol] = night
        template_obs[self.mjdCol] = MJD

        obs = pd.concat((obs, template_obs))

        return obs

    def get_SNRTime(self, grp, mjd_min, mjd_max, autogen=False, downtimes=True, runmode='one_lc'):

        mjds = np.arange(mjd_min, mjd_max, 1)

        df_tot = pd.DataFrame()
        i_mjds = list(range(len(mjds)))
        params = {}
        params['mjds'] = mjds
        params['grp'] = grp
        params['autogen'] = autogen
        params['runmode'] = runmode
        params['downtimes'] = downtimes

        # shuffle(i_mjds)

        obs = pd.DataFrame()
        if not autogen:
            nproc = 1
            df_tot = multiproc(i_mjds, params,
                               self.get_SNRTime_loop, nproc)
        else:
            df_tot, obs = self.get_SNRTime_loop(i_mjds, params)

        return df_tot, obs

    def get_SNRTime_loop(self, i_mjds, params, j=0, output_q=None):

        mjds = params['mjds']
        grp = params['grp']
        autogen = params['autogen']
        runmode = params['runmode']
        downtimes = params['downtimes']

        obs = pd.DataFrame(grp)
        if not autogen:
            df_tot = self.run_on_survey(i_mjds, mjds, grp, runmode)
        else:
            df_tot, obs = self.run_autogen(mjds, grp, runmode, downtimes)

        if output_q is not None:
            return output_q.put({j: df_tot})
        else:
            return df_tot, obs

    def run_on_survey(self, i_mjds, mjds, grp, runmode):

        df_tot = pd.DataFrame()

        for i in i_mjds:
            # for i in [i_mjds[4]]:
            df_mean = self.get_SNRTime_single(i, mjds, grp)
            print(df_mean)
            df_tot = pd.concat((df_tot, df_mean))

        return df_tot

    def run_autogen(self, mjds, grp, runmode, downtimes=False, frac_gap=0.8):

        mjd_min, df_downtimes = self.load_downtimes(downtimes)
        window_width = 30

        mjd_max = mjd_min+180.  # season length: 180 days
        mjds = np.arange(mjd_min, mjd_max, 1.)

        mjds = pd.DataFrame(mjds, columns=['MJD'])
        mjds['night'] = mjds['MJD']-mjds['MJD'].min()+1
        mjds['night'] = mjds['night'].astype(int)

        nights_gap = self.random_gaps(
            mjds.to_records(index=False), frac_gap)
        nights_gap['night'] = nights_gap['night'].astype(int)

        print('gaps', nights_gap)

        df_tot = pd.DataFrame()
        obs = pd.DataFrame()

        mjd_start_obs = mjd_min-self.min_rf_phase_qual*self.zmin

        for i, val in mjds.iterrows():
            inight = int(val['night'])
            mjd = val['MJD']
            downtime = False
            #  check if the mjd is downtime
            if not df_downtimes.empty:
                df_downtimes['MJD'] = mjd
                idd = df_downtimes['MJD'] >= df_downtimes['MJD_min']
                idd &= df_downtimes['MJD'] <= df_downtimes['MJD_max']
                downtime = not df_downtimes[idd].empty
            # check if this mjd is a gap
            igap = nights_gap['night'].isin([inight])
            gaptime = not nights_gap[igap].empty
            #print(inight, mjd, downtime, gaptime)
            # if mjd >= mjd_start_obs:
            if inight >= window_width:
                df_mjd = self.get_SNRTime_single(
                    inight-1, mjds['MJD'].to_list(), obs, runmode, window_width)
                # print(df_mjd)
                df_tot = pd.concat((df_tot, df_mjd))
            if obs.empty and not downtime and not gaptime:
                obs = self.add_obs(obs, inight, mjd, grp)
            else:
                ilast_night = np.max(obs['night'])
                if inight-ilast_night >= self.cadence_obs and not downtime and not gaptime:
                    obs = self.add_obs(obs, inight, mjd, grp)

        return df_tot, obs

    def random_gaps(self, dataSlice, frac_gap):

        all_nights = np.unique(dataSlice['night'])

        all_nights.sort()
        dataSlice.sort(order='night')
        all_nights_noborder = all_nights[1:-1]

        # dataSlice = np.random.choice(dataSlice, int(
        #    frac*len(dataSlice)), replace=False)
        nights = np.random.choice(all_nights_noborder, int(
            frac_gap*len(all_nights_noborder)), replace=False)

        nights = list(nights)

        return pd.DataFrame(nights, columns=['night'])

    def load_downtimes(self, downtimes):

        # get mjdmin and down time from simulation
        from lsst.sims.featureScheduler.modelObservatory import Model_observatory
        import itertools
        mo = Model_observatory()
        mjd_min = mo.mjd
        df_downtimes = pd.DataFrame()
        if downtimes:
            downtim = mo.downtimes.tolist()
            df_downtimes = pd.DataFrame(
                downtim, columns=['MJD_min', 'MJD_max'])

        return mjd_min, df_downtimes

    def get_SNRTime_single(self, index, mjds, grp, runmode, window=80):

        sn = pd.DataFrame()
        if runmode != 'one_lc':
            sn = self.get_SNRTime_all_lc(index, mjds, grp, window=window)
        else:
            sn = self.get_SNRTime_one_lc(index, mjds, grp, window=window)

        return sn

    def get_SNRTime_all_lc(self, index, mjds, grp, window=80):
        T0s = mjds
        if len(mjds) < window or index < window:
            return pd.DataFrame()

        if index >= 0:
            T0s = mjds[index-window:index+1]
        mjd_max_time = np.max(T0s)
        mjd_min_time = np.min(T0s)
        season = 1

        idx = grp['observationStartMJD'] <= mjd_max_time
        idx &= grp['observationStartMJD'] >= mjd_min_time
        data = pd.DataFrame(grp[idx])
        data['season'] = season

        dd = pd.DataFrame([(season, mjd_min_time, mjd_max_time)], columns=[
            'season', 'MJD_min', 'MJD_max'])
        dur_z = dd.groupby(['season']).apply(
            lambda x: self.duration_z(x, 1.))
        if not dur_z.empty:
            gen_par = dur_z.groupby(['z', 'season']).apply(
                lambda x: self.calcDaymax(x)).reset_index()
            print('hello', gen_par)
            lc = self.step_lc(data, gen_par)
            if not lc.empty:
                lc.index = lc.index.droplevel()
            # estimate efficiencies
            sn = self.get_nsn(lc, dur_z, grp, mjd_max_time)
            sn['cadence'] = self.cadence(
                data.to_records(index=False), op=np.mean)
            # get the number of epochs
            sn['Nepochs'] = len(data['night'].unique())
            return sn

        return pd.DataFrame()

    def simul_sn(self, data):

        mjd_max_time = np.max(data['observationStartMJD'])
        mjd_min_time = np.min(data['observationStartMJD'])
        season = 1

        dd = pd.DataFrame([(season, mjd_min_time, mjd_max_time)], columns=[
            'season', 'MJD_min', 'MJD_max'])
        dur_z = dd.groupby(['season']).apply(
            lambda x: self.duration_z(x, 1.))
        if not dur_z.empty:
            gen_par = dur_z.groupby(['z', 'season']).apply(
                lambda x: self.calcDaymax(x)).reset_index()
            print('hello here', gen_par)
            lc = self.step_lc(data, gen_par)
            if not lc.empty:
                lc.index = lc.index.droplevel()
            # estimate efficiencies
            sn = self.get_nsn(lc, dur_z, data, mjd_max_time)
            sn['cadence'] = self.cadence(
                data.to_records(index=False), op=np.mean)
            # get the number of epochs
            sn['Nepochs'] = len(data['night'].unique())
            return sn

        return pd.DataFrame()

    def get_SNRTime_one_lc(self, index, mjds, grp, window=30.):

        zref = 0.6
        T0 = mjds[index]
        mjd_max_time = T0
        mjd_min_time = T0-window

        if mjd_min_time < 0:
            return pd.DataFrame()

        # select data
        season = 1
        idx = grp['observationStartMJD'] <= mjd_max_time
        idx &= grp['observationStartMJD'] >= mjd_min_time
        data = pd.DataFrame(grp[idx])
        data['season'] = season

        r = [(zref, season, T0, self.min_rf_phase, self.max_rf_phase)]
        gen_par = pd.DataFrame(
            r, columns=['z', 'season', 'daymax', 'minRFphase', 'maxRFphase'])

        # generate lc here
        lc = self.step_lc(data, gen_par)

        # estimate SNR per band for lc point with snr > 1.

        idx = lc['snr_m5'] > 1.
        lc = lc[idx]

        dicout = {}
        for b in np.unique(lc['band']):
            io = lc['band'] == b
            sel_lc = lc[io]
            dicout['SNR_{}'.format(b.split(':')[-1])
                   ] = [np.sqrt(np.sum(sel_lc['snr_m5']**2))]

        sn = pd.DataFrame.from_dict(dicout)
        sn['MJD_obs'] = T0

        return sn
        print(lc)

        print(test)

        # plt.plot(data['observationStartMJD'], data['fiveSigmaDepth'], 'r*')
        # plt.show()
        dd = pd.DataFrame([(season, mjd_min_time, mjd_max_time)], columns=[
            'season', 'MJD_min', 'MJD_max'])
        dur_z = dd.groupby(['season']).apply(
            lambda x: self.duration_z(x, 1.))
        if not dur_z.empty:
            gen_par = dur_z.groupby(['z', 'season']).apply(
                lambda x: self.calcDaymax(x)).reset_index()
            # generate LC here
            # print('generating LC', len(data), mjd_min_time, mjd_max_time)
            lc = self.step_lc(data, gen_par)
            # transform lc to sn
            # print('ahooo', len(lc), gen_par)
            if not lc.empty:
                lc.index = lc.index.droplevel()
            # estimate efficiencies
            sn = self.get_nsn(lc, dur_z, grp, mjd_max_time)
            sn['cadence'] = self.cadence(
                data.to_records(index=False), op=np.mean)
            # get the number of epochs
            sn['Nepochs'] = len(data['night'].unique())
            return sn

        return pd.DataFrame()

    def get_nsn(self, lc, dur_z, grp, mjd_max_time):

        if lc.empty:
            return pd.DataFrame()
        sn_effis = self.step_efficiencies(lc)
        # estimate nsn
        sn = self.step_nsn(sn_effis, dur_z)
        print('sn', sn)
        # sn.index = sn.index.droplevel()
        if not sn.empty:
            # sn = sn.groupby(['healpixID', 'season', 'x1', 'color'])[
            #    'nsn_expected', 'nsn'].sum().reset_index()
            sn = sn.groupby(['healpixID', 'season', 'x1', 'color'])[[
                'nsn_expected', 'nsn']].apply(sum).reset_index()
        else:
            healpixID = grp['healpixID'].unique()[0]
            season = grp['season'].unique()[0]
            x1 = grp['x1'].unique()[0]
            color = grp['color'].unique()[0]
            r = [(healpixID, season, x1, color, 1., 0.)]
            cols = ['healpixID', 'season', 'x1',
                    'color', 'nsn_expected', 'nsn']
            sn = pd.DataFrame(r, columns=cols)
        sn['MJD_obs'] = mjd_max_time

        return sn

    def cadence_gap(self, grp, cadName='cadence', gapName='gap_max'):
        """
        Method to estimate the cadence and max gap for a set of observations

        Parameters
        ---------------
        grp: pandas df
          data to process
        cadName: str, opt
          cadence col name (default: cadence)
        gapName: str, opt
          gap max col name (default: gap_max)

        Returns
        -----------
        pandas df with cadence and max gap

        """
        grp = grp.sort_values(by=['night'])
        cadence = -1
        gap_max = -1

        diff = np.diff(np.unique(grp['night']))
        if len(diff) >= 2:
            cadence = np.median(diff)
            gap_max = np.max(diff)

        return pd.DataFrame({cadName: [cadence], gapName: [gap_max]})

    def filter_allocation(self, grp):
        """
        Method to estimate filter allocation

        Parameters
        ---------------
        grp: pandas df
          data to process

        Returns
        -----------
        pandas df with filter allocation

        """

        dictres = {}
        bands = 'ugrizy'
        ntot = len(grp)
        for b in bands:
            io = grp['filter'] == b
            dictres['frac_{}'.format(b)] = [len(grp[io])/ntot]

        return pd.DataFrame(dictres)

    def season_length(self, seasons, dataSlice):
        """
        Method to estimate season lengths vs z

        Parameters
        ---------------
        seasons: list(int)
          list of seasons to process
        dataSlice: numpy array
          array of observations

        Returns
        -----------
        seasons: list(int)
          list of seasons to process
        dur_z: pandas df
          season lengths vs z
        """
        # if seasons = -1: process the seasons seen in data
        if seasons == [-1]:
            seasons = np.unique(dataSlice[self.seasonCol])

        # season infos
        dfa = pd.DataFrame(np.copy(dataSlice))
        dfa = dfa[dfa['season'].isin(seasons)]
        season_info = dfa.groupby(['season']).apply(
            lambda x: self.seasonInfo(x, min_duration=60)).reset_index()

        if season_info.empty:
            return [], pd.DataFrame()

        # get season length depending on the redshift
        dur_z = season_info.groupby(['season']).apply(
            lambda x: self.duration_z(x)).reset_index()

        return season_info['season'].to_list(), dur_z

    def step_lc(self, obs, gen_par):
        """
        Method to generate lc

        Parameters
        ---------------
        obs: array
          observations
        gen_par: array
          simulation parameters

        Returns
        ----------
        SN light curves (astropy table)

        """
        lc = obs.groupby(['season']).apply(
            lambda x: self.genLC(x, gen_par))

        # plot figs for movie
        if self.ploteffi and self.fig_for_movie and len(lc) > 0:
            self.plot_for_movie(obs, lc, gen_par)

        return lc

    def step_efficiencies(self, lc):
        """
        Method to estimate observing efficiencies

        Parameter
        -------------
        lc: pandas df
           light curves

        Returns
        -----------
        pandas df with efficiencies

        """
        sn_effis = lc.groupby(['healpixID', 'season', 'z', 'x1', 'color', 'sntype']).apply(
            lambda x: self.sn_effi(x)).reset_index()

        # estimate efficiencies
        for vv in ['healpixID', 'season']:
            sn_effis[vv] = sn_effis[vv].astype(int)
        sn_effis['effi'] = sn_effis['nsel']/sn_effis['ntot']
        sn_effis['effi_err'] = np.sqrt(
            sn_effis['nsel']*(1.-sn_effis['effi']))/sn_effis['ntot']

        if self.ploteffi:
            from sn_metrics.sn_plot_live import plotNSN_effi
            for season in sn_effis['season'].unique():
                idx = sn_effis['season'] == season
                plotNSN_effi(sn_effis[idx], 'effi', 'effi_err',
                             'Observing Efficiencies', ls='-')

        return sn_effis

    def step_nsn(self, sn_effis, dur_z):
        """
        Method to estimate the number of supernovae from efficiencies

        Parameters
        ---------------
        sn_effis: pandas df
          data with efficiencies of observation
        dur_z:  array
          array of season length

        Returns
        ----------
        initial sn_effis appended with a set of infos (duration, nsn)

        """
        # add season length here
        sn_effis = sn_effis.merge(
            dur_z, left_on=['season', 'z'], right_on=['season', 'z'])

        # estimate the number of supernovae
        sn_effis['nsn'] = sn_effis['effi']*sn_effis['nsn_expected']
        # sn_effis['nsn'] = sn_effis['effi']*self.nsn_expected(
        #    sn_effis['z'].to_list())/(1.+(sn_effis['season_length_orig']/self.duration_ref)/sn_effis['season_length'])
        return sn_effis

    def ebvofMW_calc(self):
        """
        Method to estimate E(B-V)

        Returns
        ----------
        E(B-V)

        """
        # in that case ebvofMW value is taken from a map
        coords = SkyCoord(self.pixRA, self.pixDec, unit='deg')
        try:
            sfd = SFDQuery()
        except Exception as err:
            from dustmaps.config import config
            config['data_dir'] = 'dustmaps'
            import dustmaps.sfd
            dustmaps.sfd.fetch()
            # dustmaps('dustmaps')
        sfd = SFDQuery()
        ebvofMW = sfd(coords)

        return ebvofMW

    def seasonInfo(self, grp, min_duration):
        """
        Method to estimate seasonal info (cadence, season length, ...)

        Parameters
        --------------
        grp: pandas df group
        min_duration: float
          minimal duration for a season to be considered

        Returns
        ---------
        pandas df with the following cols:
        - Nvisits: number of visits for this group
        - N_xx:  number of visits in xx where xx is defined in self.bandstat

        """
        df = pd.DataFrame([len(grp)], columns=['Nvisits'])
        df['MJD_min'] = grp[self.mjdCol].min()
        df['MJD_max'] = grp[self.mjdCol].max()
        df['season_length'] = df['MJD_max']-df['MJD_min']
        df['cadence'] = 0.

        if len(grp) > 5:
            # to = grp.groupby(['night'])[self.mjdCol].median().sort_values()
            # df['cadence'] = np.mean(to.diff())
            nights = np.sort(grp['night'].unique())
            diff = np.asarray(nights[1:]-nights[:-1])
            df['cadence'] = np.median(diff).item()

            # select seasons of at least 30 days
        idx = df['season_length'] >= min_duration

        return df[idx]

    def duration_z(self, grp, min_duration=60.):
        """
        Method to estimate the season length vs redshift
        This is necessary to take into account boundary effects
        when estimating the number of SN that can be detected

        daymin, daymax = min and max MJD of a season
        T0_min(z) =  daymin-(1+z)*min_rf_phase_qual
        T0_max(z) =  daymax-(1+z)*max_rf_phase_qual
        season_length(z) = T0_max(z)-T0_min(z)

        Parameters
        --------------
        grp: pandas df group
          data to process: season infos
        min_duration: float, opt
          min season length for a season to be processed (deafult: 60 days)

        Returns
        ----------
        pandas df with season_length, z, T0_min and T0_max cols

        """

        daymin = grp['MJD_min'].values
        daymax = grp['MJD_max'].values
        dur_z = pd.DataFrame(self.zrange, columns=['z'])
        dur_z['T0_min'] = daymin-(1.+dur_z['z'])*self.min_rf_phase_qual
        dur_z['T0_max'] = daymax-(1.+dur_z['z'])*self.max_rf_phase_qual
        dur_z['season_length'] = dur_z['T0_max']-dur_z['T0_min']
        # dur_z['season_length_orig'] = daymax-daymin
        # dur_z['season_length_orig'] = [daymax-daymin]*len(self.zrange)
        nsn = self.nsn_from_rate(dur_z)
        dur_z = dur_z.merge(nsn, left_on=['z'], right_on=['z'])

        idx = dur_z['season_length'] > min_duration
        sel = dur_z[idx]
        if len(sel) < 2:
            return pd.DataFrame()
        return dur_z

    def calcDaymax(self, grp):
        """
        Method to estimate T0 (daymax) values for simulation.

        Parameters
        --------------
        grp: group (pandas df sense)
         group of data to process with the following cols:
           T0_min: T0 min value (per season)
           T0_max: T0 max value (per season)

        Returns
        ----------
        pandas df with daymax, min_rf_phase, max_rf_phase values

        """

        if self.T0s == 'all':
            T0_max = grp['T0_max'].values
            T0_min = grp['T0_min'].values
            num = (T0_max-T0_min)/self.daymaxStep
            if T0_max-T0_min > 1.:
                df = pd.DataFrame(np.linspace(
                    T0_min, T0_max, int(num)), columns=['daymax'])
            else:
                df = pd.DataFrame([-1], columns=['daymax'])
        else:
            df = pd.DataFrame([0.], columns=['daymax'])

        df['minRFphase'] = self.min_rf_phase
        df['maxRFphase'] = self.max_rf_phase

        return df

    def genLC(self, grp, gen_par_orig):
        """
        Method to generate light curves from observations

        Parameters
        ---------------
        grp: pandas group
          observations to process
        gen_par_orig: pandas df
          simulation parameters

        Returns
        ----------
        light curves as pandas df

        """
        season = grp.name
        idx = gen_par_orig['season'] == season
        gen_par = gen_par_orig[idx].to_records(index=False)

        sntype = dict(zip([(-2.0, 0.2), (0.0, 0.0)], ['faint', 'medium']))
        res = pd.DataFrame()
        for key, vals in self.lcFast.items():
            gen_par_cp = gen_par.copy()
            if key == (-2.0, 0.2):
                idx = gen_par_cp['z'] < 0.9
                gen_par_cp = gen_par_cp[idx]
            lc = vals(grp.to_records(index=False),
                      0.0, gen_par_cp, bands='grizy')
            lc['x1'] = key[0]
            lc['color'] = key[1]
            lc['sntype'] = sntype[key]
            res = pd.concat((res, lc))
            # break
        return res

    def plot_for_movie(self, obs, lc, gen_par):
        """
        Method to make a set of plot that may be assembled as a movie

        Parameters
        ---------------
        obs: pandas df
          observations
        lc: pandas df
          set of light curves
        gen_par: pandas df
          simulation parameters for SN

        """

        for season in obs['season'].unique():
            idxa = obs['season'] == season
            idxb = lc['season'] == season
            idxc = gen_par['season'] == season
            self.plotter.plotLoop(self.healpixID, season,
                                  obs[idxa].to_records(index=False), lc[idxb], gen_par[idxc].to_records(index=False))

    def sn_effi(self, lc):
        """
        Method to transform LCs to supernovae

        Parameters
        ---------------
        lc: pandas grp
          light curve

        Returns
        ----------
        pandas df of sn efficiencies vs z
        """

        lcarr = lc.to_records(index=False)

        idx = lcarr['snr_m5'] >= self.snr_min

        lcarr = np.copy(lcarr[idx])

        T0s = np.unique(lcarr['daymax'])
        T0s.sort()

        deltaT = lcarr['daymax']-T0s[:, np.newaxis]

        flag = np.abs(deltaT) < 1.e-5
        flag_idx = np.argwhere(flag)

        resdf = pd.DataFrame(T0s, columns=['daymax'])

        # get n_phase_min, n_phase_max
        for vv in ['n_phmin', 'n_phmax', 'F_x0x0', 'F_x0x1', 'F_x0daymax', 'F_x0color', 'F_x1x1', 'F_x1daymax',
                   'F_x1color', 'F_daymaxdaymax', 'F_daymaxcolor', 'F_colorcolor']:
            resdf[vv] = self.get_sum(lcarr, vv, len(deltaT), flag)

        nights = np.tile(lcarr['night'], (len(deltaT), 1))
        phases = np.tile(lcarr['phase'], (len(deltaT), 1))

        flagph = phases >= 0.
        resdf['nepochs_aft'] = self.get_epochs(nights, flag, flagph)
        flagph = phases <= 0.
        resdf['nepochs_bef'] = self.get_epochs(nights, flag, flagph)

        # get selection efficiencies
        effis = self.efficiencies(resdf)

        return effis

    def lc_to_sn(self, lc):
        """
        Method to transform LCs to supernovae

        Parameters
        ---------------
        lc: pandas grp
          light curve

        Returns
        ----------
        pandas df of sn
        """

        lcarr = lc.to_records(index=False)

        idx = lcarr['snr_m5'] >= self.snr_min

        lcarr = np.copy(lcarr[idx])

        T0s = np.unique(lcarr['daymax'])
        T0s.sort()

        deltaT = lcarr['daymax']-T0s[:, np.newaxis]

        bands = np.unique(lcarr['band'])
        flag = np.abs(deltaT) < 1.e-5
        flag_idx = np.argwhere(flag)

        resdf = pd.DataFrame(T0s, columns=['daymax'])

        # get n_phase_min, n_phase_max
        for vv in ['n_phmin', 'n_phmax', 'F_x0x0', 'F_x0x1', 'F_x0daymax', 'F_x0color', 'F_x1x1', 'F_x1daymax',
                   'F_x1color', 'F_daymaxdaymax', 'F_daymaxcolor', 'F_colorcolor']:
            resdf[vv] = self.get_sum(lcarr, vv, len(deltaT), flag)
        for b in bands:
            flagb = lcarr['band'] == b
            resdf['SNR_{}'.format(b.split(':')[-1])] = self.get_sumsq(
                lcarr, 'snr_m5', len(deltaT), flag & flagb)

        nights = np.tile(lcarr['night'], (len(deltaT), 1))
        phases = np.tile(lcarr['phase'], (len(deltaT), 1))

        flagph = phases >= 0.
        resdf['nepochs_aft'] = self.get_epochs(nights, flag, flagph)
        flagph = phases <= 0.
        resdf['nepochs_bef'] = self.get_epochs(nights, flag, flagph)

        return resdf

    def get_sum(self, lcarr, varname, nvals, flag):
        """
        Method to get the sum of variables using broadcasting

        Parameters
        --------------
        lcarr: numpy array
          data to process
        varname: str
          col to process in lcarr
        nvals: int
          dimension for tiling
        flag: array(bool)
          flag to apply

        Returns
        ----------
        array: the sum of the corresponding variable

        """

        phmin = np.tile(lcarr[varname], (nvals, 1))
        n_phmin = np.ma.array(phmin, mask=~flag)
        n_phmin = n_phmin.sum(axis=1)

        return n_phmin

    def get_sumsq(self, lcarr, varname, nvals, flag):
        """
        Method to get the sum of variables using broadcasting

        Parameters
        --------------
        lcarr: numpy array
          data to process
        varname: str
          col to process in lcarr
        nvals: int
          dimension for tiling
        flag: array(bool)
          flag to apply

        Returns
        ----------
        array: the sum of the corresponding variable

        """

        phmin = np.tile(lcarr[varname], (nvals, 1))
        n_phmin = np.square(np.ma.array(phmin, mask=~flag))
        n_phmin = n_phmin.sum(axis=1)

        return np.sqrt(n_phmin)

    def get_epochs(self, nights, flag, flagph):
        """
        Method to get the number of epochs

        Parameters
        ---------------
        nights: array
          night number array
        flag: array(bool)
          flag to apply
        flagph: array(bool)
          flag to apply

        Returns
        -----------
        array with the number of epochs

        """
        nights_cp = np.copy(nights)
        B = np.ma.array(nights_cp, mask=~(flag & flagph))
        B.sort(axis=1)
        C = np.diff(B, axis=1) > 0
        D = C.sum(axis=1)+1
        return D

    def sigmaSNparams(self, grp):
        """
        Method to estimate variances of SN parameters
        from inversion of the Fisher matrix

        Parameters
        ---------------
        grp: pandas df of flux derivatives wrt SN parameters
        Returns
        ----------
        Diagonal elements of the inverted matrix (as pandas df)
        """

        # params = ['x0', 'x1', 'daymax', 'color']
        parts = {}
        for ia, vala in enumerate(self.params):
            for jb, valb in enumerate(self.params):
                if jb >= ia:
                    parts[ia, jb] = grp['F_'+vala+valb]

        # print(parts)
        size = len(grp)
        npar = len(self.params)
        Fisher_Big = np.zeros((npar*size, npar*size))
        Big_Diag = np.zeros((npar*size, npar*size))
        Big_Diag = []

        for iv in range(size):
            Fisher_Matrix = np.zeros((npar, npar))
            for ia, vala in enumerate(self.params):
                for jb, valb in enumerate(self.params):
                    if jb >= ia:
                        Fisher_Big[ia+npar*iv][jb+npar*iv] = parts[ia, jb][iv]

        # pprint.pprint(Fisher_Big)

        Fisher_Big = Fisher_Big + np.triu(Fisher_Big, 1).T

        detmat = np.linalg.det(Fisher_Big)
        res = pd.DataFrame()

        if detmat:
            Big_Diag = np.diag(np.linalg.inv(Fisher_Big))
            for ia, vala in enumerate(self.params):
                indices = range(ia, len(Big_Diag), npar)
                res['Cov_{}{}'.format(vala, vala)] = np.take(Big_Diag, indices)
        else:
            for ia, vala in enumerate(self.params):
                res['Cov_{}{}'.format(vala, vala)] = 9999.

        return res

    def efficiencies(self, dfo):
        """"
        Method to estimate selection efficiencies

        Parameters
        ---------------
        df: pandas df
          data to process

        """

        df = pd.DataFrame(dfo)
        df['select'] = df['n_phmin'] >= self.n_phase_min
        df['select'] &= df['n_phmax'] >= self.n_phase_max
        df['select'] &= df['nepochs_bef'] >= self.n_bef
        df['select'] &= df['nepochs_aft'] >= self.n_aft
        df['select'] = df['select'].astype(int)
        df['Cov_colorcolor'] = 100.

        idx = df['select'] == 1

        badSN = pd.DataFrame(df.loc[~idx])
        goodSN = pd.DataFrame()
        if len(df[idx]) > 0:
            goodSN = pd.DataFrame(df.loc[idx].reset_index())
            sigma_Fisher = self.sigmaSNparams(goodSN)
            goodSN['Cov_colorcolor'] = sigma_Fisher['Cov_colorcolor']

        allSN = pd.concat((goodSN, badSN))
        allSN['select'] &= allSN['Cov_colorcolor'] <= self.sigmaC**2
        idx = allSN['select'] == 1

        return pd.DataFrame({'ntot': [len(allSN)], 'nsel': [len(allSN[idx])]})

        """
        arr = df.to_records(index=False)

        T0s = np.unique(arr['daymax'])

        deltaT = arr['daymax']-T0s[:, np.newaxis]

        flag = np.abs(deltaT) < 1.e-5
        flag_idx = np.argwhere(flag)

        all_sn = np.tile(arr['all'], (len(deltaT), 1))
        sel_sn = np.tile(arr['select'], (len(deltaT), 1))

        A = np.ma.array(all_sn, mask=~flag).count(axis=0)
        B = np.ma.array(sel_sn, mask=~flag).count(axis=0)

        print(A, B)
        """

    def zlim_or_nsn(self, effi, sntype='faint', zlim=-1):
        """
        Method to estimate the redshift limit or the number of sn

        Parameters
        ---------------
        effi: pandas df
          data to process
        sntype: str, opt
          type of SN to consider for estimation (default: faint)
        zlim: float, opt
          redshift limit

        Returns
        -----------
        if zlim<0: returns the redshift limit
        if zlim>0: returns the number of sn up to zlim


        """
        seleffi = effi[effi['sntype'] == sntype]
        seleffi = seleffi.sort_values(by=['z'])
        nsn_cum = np.cumsum(seleffi['nsn'].to_list())

        res = -999
        if zlim < 0:
            zlim = interp1d(nsn_cum/nsn_cum[-1], seleffi['z'], kind='linear',
                            bounds_error=False, fill_value=0)
            res = zlim(self.zlim_coeff)
        else:
            nsn = interp1d(seleffi['z'], nsn_cum, kind='linear',
                           bounds_error=False, fill_value=0)
            res = nsn(zlim)

        return np.round(res, 6)

    def metric(self, grp):
        """
        Method to estimate the metric(zcomp, nsn)

        Parameters
        ---------------
        grp: pandas group

        Returns
        ------------
        pandas df with the metric as cols
        """
        zcomp = -1
        nsn = -1
        if grp['effi'].mean() > 0.02:
            zcomp = self.zlim_or_nsn(grp, 'faint', -1)
            nsn = self.zlim_or_nsn(grp, 'medium', zcomp)

        if self.ploteffi:
            from sn_metrics.sn_plot_live import plot_zlim, plot_nsn
            plot_zlim(grp, 'faint', self.zmin,
                      self.zmax, self.zlim_coeff)
            plot_nsn(grp, 'medium', self.zmin, self.zmax, zcomp)

        return pd.DataFrame({'zcomp': [zcomp], 'nsn': [nsn]})

    def nsn_from_rate(self, grp):
        """
        Method to estimate the expected number of supernovae

        Parameters
        ---------------
        grp: pandas df
          data to process

        Returns
        -----------
        pandas df with z and nsn_expected as cols

        """
        durinterp_z = interp1d(grp['z'], grp['season_length'], kind='linear',
                               bounds_error=False, fill_value=0)
        zz, rate, err_rate, nsn, err_nsn = self.rateSN(zmin=self.zmin,
                                                       zmax=self.zmax,
                                                       dz=self.zstep,
                                                       duration_z=durinterp_z,
                                                       # duration=self.duration_ref,
                                                       survey_area=self.pixArea,
                                                       account_for_edges=False)

        nsn_expected = interp1d(zz, nsn, kind='linear',
                                bounds_error=False, fill_value=0)
        nsn_res = nsn_expected(grp['z'])

        return pd.DataFrame({'nsn_expected': nsn_res, 'z': grp['z'].to_list()})

    def resError(self, istatus):
        """
        Method to return a dataframe corresponding to anomalous result

        Parameters
        --------------
        istatus: int

        Returns
        ----------
        pandas df with sn_status

        """

        df = pd.DataFrame([self.healpixID], columns=['healpixID'])
        df['pixRA'] = self.pixRA
        df['pixDec'] = self.pixDec

        cols = ['season', 'zcomp', 'nsn', 'cadence', 'gap_max', 'frac_u', 'frac_g',
                'frac_r', 'frac_i', 'frac_z', 'frac_y', 'cadence_sn', 'gap_max_sn']

        for col in cols:
            df[col] = -1

        df['status'] = istatus

        return df