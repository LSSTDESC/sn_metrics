import numpy as np
from rubin_sim.maf.metrics import BaseMetric
from sn_stackers.coadd_stacker import CoaddStacker
import healpy as hp
import numpy.lib.recfunctions as rf
import multiprocessing
import yaml
from scipy import interpolate
import os
from sn_tools.sn_calcFast import LCfast, CovColor
from sn_tools.sn_telescope import Telescope
from astropy.table import Table, vstack, Column
import time
import pandas as pd
from scipy.interpolate import interp1d
from sn_tools.sn_rate import SN_Rate
from scipy.interpolate import RegularGridInterpolator
from functools import wraps
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from sn_metrics.sn_plot_live import Plot_NSN_metric
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Define decorators

# estimate processing time


def time_this(arg):
    def time_init(original_function):
        @wraps(original_function)
        def new_function(*args, **kwargs):
            if kwargs['timer']:
                import datetime
                before = datetime.datetime.now()
            x = original_function(*args, **kwargs)
            if kwargs['timer']:
                after = datetime.datetime.now()
                print("Elapsed Time for {} = {}".format(arg, after-before))
            # x.__doc__ = original_function.__doc__
            return x
        # new_function.__doc__ = x.__doc__
        return new_function
    # time_init.__doc__ = new_function.__doc__
    return time_init


# verbose mode
def verbose_this(arg):
    def verbose(original_function):
        @wraps(original_function)
        def new_function(*args, **kwargs):
            if kwargs['verbose']:
                print(arg)
            x = original_function(*args, **kwargs)
            return x
        # new_function.__doc__ = x.__doc__
        return new_function
    # verbose.__doc__ = new_function.__doc__
    return verbose


class SNNSNMetric(BaseMetric):
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
                 metricName='SNNSNMetric',
                 mjdCol='observationStartMJD', RACol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures', seeingCol='seeingFwhmEff',
                 vistimeCol='visitTime', season=[-1], coadd=True, zmin=0.0, zmax=1.2, zStep=0.03,
                 daymaxStep=4., pixArea=9.6, outputType='zlims', verbose=False, timer=False, ploteffi=False, proxy_level=0,
                 n_bef=5, n_aft=10, snr_min=5., n_phase_min=1, n_phase_max=1, errmodrel=0.1,
                 x1_color_dist=None, lightOutput=True, T0s='all', zlim_coeff=0.95,
                 ebvofMW=-1., obsstat=True, bands='grizy', fig_for_movie=False, templateLC={}, mjd_LSST_Start=60218.83514, dbName='', timeIt=False, **kwargs):

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
        self.seeingCol = seeingCol
        self.pixArea = pixArea
        self.ploteffi = ploteffi
        self.x1_color_dist = x1_color_dist
        self.T0s = T0s
        self.zlim_coeff = zlim_coeff
        self.ebvofMW = ebvofMW
        self.bands = bands
        self.fig_for_movie = fig_for_movie
        self.timeIt = timeIt

        cols = [self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol]

        self.stacker = None
        if coadd:
            cols += ['coadd']
            self.stacker = CoaddStacker(col_sum=[self.nexpCol, self.vistimeCol, 'visitExposureTime'],
                                        col_mean=[self.mjdCol, self.RACol, self.DecCol,
                                                  self.m5Col, 'pixRA', 'pixDec', 'healpixID', 'season'],
                                        col_median=['airmass',
                                                    'sky', 'moonPhase'],
                                        col_group=[
                                            self.filterCol, self.nightCol],
                                        col_coadd=[self.m5Col, 'visitExposureTime'])
        super(SNNSNMetric, self).__init__(
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

        # print('selection', self.n_bef, self.n_aft,
        #      self.n_phase_min, self.n_phase_max)
        self.lcFast = {}

        # loading reference LC files
        for key, vals in lc_reference.items():
            self.lcFast[key] = LCfast(vals, dustcorr[key], key[0], key[1], telescope,
                                      self.mjdCol, self.RACol, self.DecCol,
                                      self.filterCol, self.exptimeCol,
                                      self.m5Col, self.seasonCol, self.nexpCol, self.seeingCol,
                                      self.snr_min, lightOutput=lightOutput)

        # loading parameters
        self.zmin = zmin  # zmin for the study
        self.zmax = zmax  # zmax for the study
        self.zStep = zStep  # zstep
        # get redshift range for processing
        zRange = list(np.arange(self.zmin, self.zmax, self.zStep))
        if zRange[0] < 1.e-6:
            zRange[0] = 0.01

        self.zRange = np.unique(zRange)
        self.daymaxStep = daymaxStep  # daymax step
        self.min_rf_phase = -20.  # min ref phase for LC points selection
        self.max_rf_phase = 60.  # max ref phase for LC points selection

        self.min_rf_phase_qual = -15.  # min ref phase for bounds effects
        self.max_rf_phase_qual = 30.  # max ref phase for bounds effects

        # snrate
        self.rateSN = SN_Rate(H0=70., Om0=0.3,
                              min_rf_phase=self.min_rf_phase_qual, max_rf_phase=self.max_rf_phase_qual)

        # verbose mode - useful for debug and code performance estimation
        self.verbose = verbose
        self.timer = timer

        # status of the pixel after processing
        self.status = dict(
            zip(['ok', 'effi', 'season_length', 'nosn', 'simu_parameters', 'low_effi'], [1, -1, -2, -3, -4, -5]))

        # supernovae parameters
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
        # this is to plot live estimation of the metric
        if self.ploteffi and self.fig_for_movie:
            self.plotter = Plot_NSN_metric(self.snr_min, self.n_bef, self.n_aft,
                                           self.n_phase_min, self.n_phase_max, self.errmodrel,
                                           self.mjdCol, self.m5Col, self.filterCol, self.nightCol,
                                           templateLC=templateLC, dbName=dbName)

        # get reference time for LC night
        """
        from astropy.time import Time
        t = Time('2023-10-01T20:02:36', format='isot')
        self.mjd_LSST_Start = t.mjd
        print('hello', t.mjd)
        """
        self.mjd_LSST_Start = mjd_LSST_Start

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
        Depend on self.outputType:
          self.outputType == 'zlims':  numpy array

          self.outputType == 'sn':  numpy array

          self.outputType == 'lc':  numpy array

          self.outputType == 'effi':  numpy array
        """

        """
        import matplotlib.pyplot as plt
        plt.plot(dataSlice[self.RACol], dataSlice[self.DecCol], 'ko')
        print('data', len(dataSlice))
        plt.show()
        """

        """
        for seas in np.unique(dataSlice['season']):
            idx = dataSlice['season'] == seas
            print(seas, len(dataSlice[idx]))
        """
        #dataSlice = np.load('../DB_Files/pixel_35935.npy', allow_pickle=True)
        # time 0 for performance estimation purpose
        time_ref = time.time()
        goodFilters = np.in1d(dataSlice[self.filterCol], list(self.bands))
        dataSlice = dataSlice[goodFilters]

        healpixID = np.unique(dataSlice['healpixID'])

        if self.verbose:
            print('Observations')
            print(dataSlice[[self.mjdCol, self.exptimeCol,
                  self.filterCol, self.nightCol, self.nexpCol]])
        dataSlice = self.stacker._run(dataSlice)

        if self.verbose:
            print('Observations - after coadd')
            print(dataSlice[[self.mjdCol, self.exptimeCol,
                             self.filterCol, self.nightCol, self.nexpCol]])

        if not healpixID:
            zlimsdf = pd.DataFrame()
            # print(zlimsdf.columns, len(zlimsdf.columns))
            return zlimsdf

        # Get ebvofMW here
        ebvofMW = self.ebvofMW
        self.pixRA = np.unique(dataSlice['pixRA'])[0]
        self.pixDec = np.unique(dataSlice['pixDec'])[0]
        self.healpixID = np.unique(dataSlice['healpixID'])[0]

        if ebvofMW < 0:
            ebvofMW = self.ebvofMW_calc()

        # get the seasons
        seasons = self.season

        # if seasons = -1: process the seasons seen in data
        if self.season == [-1]:
            seasons = np.unique(dataSlice[self.seasonCol])

        # season infos
        dfa = pd.DataFrame(np.copy(dataSlice))
        dfa = dfa[dfa['season'].isin(seasons)]

        season_info = dfa.groupby(['season']).apply(
            lambda x: self.seasonInfo(x)).reset_index()

        # select seasons of at least 60 days
        idx = season_info['season_length'] >= 60
        season_info = season_info[idx]

        if self.verbose:
            print('season infos', season_info)

        if season_info.empty:
            zlimsdf = self.nooutput(self.pixRA, self.pixDec, self.healpixID)
            # print(zlimsdf.columns, len(zlimsdf.columns))
            return zlimsdf

        # get season length depending on the redshift
        dur_z = season_info.groupby(['season']).apply(
            lambda x: self.duration_z(x)).reset_index()

        if self.verbose:
            print('duration vs z', dur_z)

        if dur_z.empty:
            zlimsdf = self.nooutput(self.pixRA, self.pixDec, self.healpixID)
            # print(zlimsdf.columns, len(zlimsdf.columns))
            return zlimsdf

        # get simulation parameters

        gen_par = dur_z.groupby(['z', 'season']).apply(
            lambda x: self.calcDaymax(x)).reset_index()

        if self.verbose:
            print('getting simulation parameters')
            print(gen_par, len(gen_par))

        # prepare pandas DataFrames for output
        vara_totdf = pd.DataFrame()
        varb_totdf = pd.DataFrame()

        # loop on seasons
        for seas in seasons:
            # get seasons processing
            idx = season_info['season'] == seas
            if len(season_info[idx]) > 0:
                cadence = season_info[idx]['cadence'].item()
                season_length = season_info[idx]['season_length'].item()
                Nvisits = {}
                """
                for b in 'ugrizy':
                    Nvisits[b] = season_info[idx]['Nvisits_{}'.format(
                        b)].item()
                """
                if self.obsstat:
                    # Nvisits['filters_night'] = season_info[idx]['filters_night'].item()
                    for b in self.bandstat:
                        Nvisits[b] = season_info[idx]['N_{}'.format(b)].item()

                Nvisits['total'] = season_info[idx]['Nvisits'].item()
                vara_df, varb_df = self.run_seasons(
                    dataSlice, [seas], gen_par, dur_z, ebvofMW, cadence, season_length, Nvisits, verbose=self.verbose, timer=self.timer)

                vara_totdf = pd.concat([vara_totdf, vara_df], sort=False)
                varb_totdf = pd.concat([varb_totdf, varb_df], sort=False)

        # estimate time of processing
        if self.verbose:
            print('finally - eop', time.time()-time_ref)
            print(varb_totdf.columns)
            toshow = ['pixRA', 'pixDec', 'healpixID', 'season', 'x1_faint', 'color_faint', 'zlim_faint',
                      'zmean_faint', 'zpeak_faint',
                      'nsn_zlim_faint', 'nsn_zmean_faint', 'nsn_zpeak_faint']
            if self.obsstat:
                # toshow += ['N_filters_night']
                for b in self.bandstat:
                    toshow += ['N_{}'.format(b)]
            print(varb_totdf[toshow])

        # return the output as chosen by the user (outputType)
        if self.outputType == 'lc':
            return varb_totdf

        if self.outputType == 'sn':
            return vara_totdf

        if self.outputType == 'effi':
            return vara_totdf

        if self.verbose:
            print('final result', varb_totdf[[
                  'season', 'zlim_faint', 'nsn_zlim_faint']])
            print('Summary zlim_med:', np.median(
                varb_totdf['zlim_faint']), 'NSN', np.sum(varb_totdf['nsn_zlim_faint']))

        # print('final result', varb_totdf[[
        #      'season', 'zlim_faint', 'nsn_zlim_faint']])
        varb_totdf['timeproc'] = time.time()-time_ref
        if self.timeIt:
            print('processing time', time.time()-time_ref)

        return varb_totdf

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

    def nooutput(self, pixRA, pixDec, healpixID, val='season_length'):
        """
        Method to return a dataframe when no data could be processed

        Parameters
        ---------------
        pixRA: float
          pixel RA
        pixDec: float
          pixel Dec
        healpixID: int
           healpix number
        val: str, opt
          reason of the missing data (default: season_length)
        Returns
        ----------
        dummy df

        """
        m5_med = {}
        Nvisits = {}
        for b in 'ugrizy':
            m5_med[b] = 0.
            Nvisits[b] = 0
            Nvisits['{}{}'.format(b, b)] = 0
        Nvisits['gr'] = 0
        Nvisits['gi'] = 0
        Nvisits['gz'] = 0
        Nvisits['iz'] = 0
        zlimsdf = self.errordf(
            pixRA, pixDec, healpixID, 1,
            self.status[val],
            m5_med, -1., -1., -1., -1., -1., Nvisits)

        return zlimsdf

    @verbose_this('Processing season')
    @time_this('Processing season')
    def run_seasons(self, dataSlice, seasons, gen_par, dura_z, ebvofMW, cadence, season_length, Nvisits, **kwargs):
        """
        Method to run on seasons

        Parameters
        --------------
        dataSlice: numpy array, opt
          data to process (scheduler simulations)
        seasons: list(int)
          list of seasons to process
        gen_par: numpy array
           parameters for generation
        dura_z: array
          season duration vs z
        ebvofMW: float
           ebvofMW for dust effect

        Returns
        ---------
        effi_seasondf: pandas df
          efficiency curves
        zlimsdf: pandas df
          redshift limits and number of supernovae
        """

        time_ref = time.time()

        if self.verbose:
            print('#### Processing season', seasons, self.healpixID)

        groupnames = ['pixRA', 'pixDec', 'healpixID', 'season', 'x1', 'color']

        gen_p = gen_par[gen_par['season'].isin(seasons)]
        if gen_p.empty:
            if self.verbose:
                print('No generator parameter found')
            return None, None
        else:
            if self.verbose:
                print('generator parameters', gen_p, len(gen_p))

        dur_z = dura_z[dura_z['season'].isin(seasons)]
        obs = pd.DataFrame(np.copy(dataSlice))
        obs = obs[obs['season'].isin(seasons)]

        if self.timer:
            time_refb = time.time()

        # coaddition per night and per band (if requested by the user)
        # done earlier now
        """
        if self.stacker is not None:
            if self.verbose:
                print('before stacking', seasons,
                      obs[[self.m5Col, 'visitExposureTime']])

            obs = self.stacker._run(obs.to_records(index=False))
            if self.verbose:
                print('after stacking', seasons,
                      obs[[self.m5Col, 'visitExposureTime']])
            if self.timer:
                print('timing:', time.time()-time_refb)
        else:
            obs = obs.to_records(index=False)
        """

        obs = obs.to_records(index=False)
        obs.sort(order='night')
        if self.verbose:
            print('obs season', seasons,
                  obs[[self.mjdCol, self.nightCol, self.filterCol, self.RACol, self.DecCol, self.m5Col]], len(obs))

        # estimate m5 median and gaps
        m5_med, gap_max, gap_med = self.getInfos_obs(obs)

        # simulate supernovae and lc
        if self.verbose:
            print("LC generation")

        sn = pd.DataFrame()
        sn_infos = pd.DataFrame()
        if ebvofMW < 0.25:
            sn, lc, sn_infos = self.gen_LC_SN(obs, 0.0, gen_p.to_records(
                index=False), verbose=self.verbose, timer=self.timer)

            # print('sn here', sn[['x1', 'color', 'z', 'daymax', 'Cov_colorcolor']])
            if self.verbose:
                print(
                    'sn here', sn[['x1', 'color', 'z', 'daymax', 'Cov_colorcolor']])
                idx = np.abs(sn['x1']+2.0) < 1.e-5
                idx &= sn['z'] <= 0.2
                sel = sn[idx]
                sel = sel.sort_values(by=['z', 'daymax'])
                print('sn and lc', len(sn),
                      sel[['x1', 'color', 'z', 'daymax', 'Cov_colorcolor', 'n_bef', 'n_aft']])

        # estimate m5 median and gaps
        m5_med, gap_max, gap_med = self.getInfos_obs(obs)

        # simulate supernovae and lc
        if self.verbose:
            print("LC generation")

        sn = pd.DataFrame()
        sn_infos = pd.DataFrame()
        if ebvofMW < 0.25:
            sn, lc, sn_infos = self.gen_LC_SN(obs, 0.0, gen_p.to_records(
                index=False), verbose=self.verbose, timer=self.timer)

            if self.verbose:
                print(
                    'sn here', sn[['x1', 'color', 'z', 'daymax', 'Cov_colorcolor']])
                idx = np.abs(sn['x1']+2.0) < 1.e-5
                idx &= sn['z'] <= 0.2
                sel = sn[idx]
                sel = sel.sort_values(by=['z', 'daymax'])

                print('sn and lc', len(sn),
                      sel[['x1', 'color', 'z', 'daymax', 'Cov_colorcolor', 'n_bef', 'n_aft']])
                lc['test'] = lc[self.exptimeCol]/lc[self.nexpCol]
                print(lc[['snr_m5', 'z', 'flux_5', 'mag', 'band',
                          'test', self.nexpCol, self.m5Col]])

            if self.outputType == 'lc' or self.outputType == 'sn':
                return sn, lc

        if sn.empty:
            # no LC could be simulated -> fill output with errors
            if self.verbose:
                print('no simulation possible!!')
            for seas in seasons:
                zlimsdf = self.errordf(
                    self.pixRA, self.pixDec, self.healpixID, seas,
                    self.status['nosn'],
                    m5_med, gap_max, gap_med, ebvofMW, cadence, season_length, Nvisits)
                effi_seasondf = self.erroreffi(
                    self.pixRA, self.pixDec, self.healpixID, seas)
            return effi_seasondf, zlimsdf
        else:
            # LC could be simulated -> estimate efficiencies
            if self.verbose:
                idx = np.abs(sn["x1"] + 2) < 1.0e-5
                idx &= sn["z"] <= 0.1
                sel = sn[idx]
                sel = sel.sort_values(by=["z", "daymax"])

                print(
                    "sn and lc",
                    len(sn),
                    sel.columns,
                    sel[["x1", "color", "z", "daymax",
                         "Cov_colorcolor", "n_bef", "n_aft"]],
                )
                print('effidf', self.effidf(
                    sel, verbose=self.verbose, timer=self.timer))

            effi_seasondf = self.effidf(
                sn, verbose=self.verbose, timer=self.timer)

            # zlims can only be estimated if efficiencies are ok
            idx = effi_seasondf['z'] <= 0.2
            idx &= effi_seasondf['z'] >= 0.05
            x1ref = -2.0
            colorref = 0.2
            idx &= np.abs(effi_seasondf['x1']-x1ref) < 1.e-5
            idx &= np.abs(effi_seasondf['color']-colorref) < 1.e-5
            sel = effi_seasondf[idx]

            if np.mean(sel['effi']) > 0.02:
                # estimate zlims
                zlimsdf = self.zlims(
                    effi_seasondf, dur_z, groupnames, verbose=self.verbose, timer=self.timer)

                if self.verbose:
                    print('zlims', zlimsdf)
                # estimate number of medium supernovae
                """
                zlimsdf['nsn_med'],  zlimsdf['err_nsn_med'] = zlimsdf.apply(lambda x: self.nsn_typedf(
                    x, 0.0, 0.0, effi_seasondf, dur_z), axis=1, result_type='expand').T.values
                """
                zlimsdf['nsn_zlim'],  zlimsdf['nsn_zmean'], zlimsdf['nsn_zpeak'] = zlimsdf.apply(lambda x: self.nsn_typedf(
                    x, 0.0, 0.0, effi_seasondf, dur_z), axis=1, result_type='expand').T.values

                dfa = pd.DataFrame([zlimsdf.iloc[0]], columns=zlimsdf.columns)
                dfb = pd.DataFrame([zlimsdf.iloc[1]], columns=zlimsdf.columns)

                on = ['healpixID', 'pixRA', 'pixDec', 'season']
                zlimsdf = dfa.merge(
                    dfb, left_on=on, right_on=on, suffixes=('_faint', '_medium'))

                if self.verbose:
                    print(zlimsdf.columns)
                    print('result here', zlimsdf[[
                          'zlim_faint', 'zmean_faint', 'nsn_zlim_faint']])

                # add observing stat if requested
                if self.obsstat:
                    # add median m5
                    for key, vals in m5_med.items():
                        zlimsdf.loc[:, 'm5_med_{}'.format(key)] = vals
                    zlimsdf.loc[:, 'gap_max'] = gap_max
                    zlimsdf.loc[:, 'gap_med'] = gap_med
                    zlimsdf.loc[:, 'ebvofMW'] = ebvofMW
                    zlimsdf.loc[:, 'cadence'] = cadence
                    zlimsdf.loc[:, 'season_length'] = season_length
                    for b, vals in Nvisits.items():
                        zlimsdf.loc[:, 'N_{}'.format(b)] = vals
                    zlimsdf.loc[:,
                                'cad_sn_mean'] = sn_infos['cad_sn_mean'].mean()
                    zlimsdf.loc[:, 'cad_sn_std'] = np.sqrt(
                        np.sum(sn_infos['cad_sn_std']**2))
                    zlimsdf.loc[:,
                                'gap_sn_mean'] = sn_infos['gap_sn_mean'].mean()
                    zlimsdf.loc[:, 'gap_sn_std'] = np.sqrt(
                        np.sum(sn_infos['gap_sn_std']**2))

            else:

                for seas in seasons:
                    zlimsdf = self.errordf(
                        self.pixRA, self.pixDec, self.healpixID, seas,
                        self.status['low_effi'],
                        m5_med, gap_max, gap_med, ebvofMW, cadence, season_length, Nvisits)
                    effi_seasondf = self.erroreffi(
                        self.pixRA, self.pixDec, self.healpixID, seas)

            if self.proxy_level == 2:
                zlimsdf['nsn'] = -1
                zlimsdf['var_nsn'] = -1
                return effi_seasondf, zlimsdf

            # estimate number of supernovae - total - proxy_level = 0 or 1
            zlimsdf['nsn'], zlimsdf['var_nsn'] = self.nsn_tot(
                effi_seasondf, zlimsdf, dur_z, verbose=self.verbose, timer=self.timer)

            if self.verbose:
                print('final result ', zlimsdf)
            if self.timer:
                print('timing:', time.time()-time_refb)

        if self.verbose:
            print('#### SEASON processed', time.time()-time_ref,
                  seasons)

        return effi_seasondf, zlimsdf

    def getInfos_obs(self, obs):
        """
        Method to get infos from observations

        Parameters
        --------------
        obs: array
           observations

        Returns
        ----------
        m5_med: dict
          median m5 per band
        gap_max: float
          max gap
        gap_med: float
          med gap

        """
        m5_med = {}
        for b in 'ugrizy':
            m5_med[b] = 0.

        for b in np.unique(obs[self.filterCol]):
            io = obs[self.filterCol] == b
            sel = obs[io]
            m5_med[b] = np.median(sel[self.m5Col])

        obs.sort(order=self.mjdCol)
        diffs = np.diff(obs[self.mjdCol])
        gap_max = np.max(diffs)
        gap_med = np.median(diffs)

        return m5_med, gap_max, gap_med

    def duration_z(self, grp):
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

        Returns
        ----------
        pandas df with season_length, z, T0_min and T0_max cols

        """

        daymin = grp['MJD_min'].values
        daymax = grp['MJD_max'].values
        dur_z = pd.DataFrame(self.zRange, columns=['z'])
        dur_z['T0_min'] = daymin-(1.+dur_z['z'])*self.min_rf_phase_qual
        dur_z['T0_max'] = daymax-(1.+dur_z['z'])*self.max_rf_phase_qual
        dur_z['season_length'] = dur_z['T0_max']-dur_z['T0_min']
        # dur_z['season_length'] = [daymax-daymin]*len(self.zRange)

        idx = dur_z['season_length'] > 60.
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
            if T0_max-T0_min > 10:
                df = pd.DataFrame(np.linspace(
                    T0_min, T0_max, int(num)), columns=['daymax'])
            else:
                df = pd.DataFrame([-1], columns=['daymax'])
        else:
            df = pd.DataFrame([0.], columns=['daymax'])

        df['minRFphase'] = self.min_rf_phase
        df['maxRFphase'] = self.max_rf_phase

        return df

    def errordf(self, pixRA, pixDec, healpixID, season, errortype,
                m5_med, gap_max, gap_med, ebvofMW, cadence, season_length, Nvisits):
        """
        Method to return error df related to zlims values

        Parameters
        --------------
        pixRA: float
          pixel RA
        pixDec: float
          pixel Dec
        healpixID: int
          healpix ID
        season: int
          season
        errortype: str
          type of error
        m5_med: dict
          median m5 values per band
        gap_max: float
          max internight gap
        gap_med: float
          median internight gap
        ebvofMW: float
            E(B-V) of MW
        cadence: float
           cadence of observation
        season_length: float
          length of the season
        Nvisits: dict
           total number of visits per (combi of) bands and per season
        """

        df = pd.DataFrame({'pixRA': [np.round(pixRA, 4)],
                           'pixDec': [np.round(pixDec, 4)],
                          'healpixID': [healpixID],
                           'nsn': [-1.0],
                           'var_nsn': [-1.0],
                           'season': [int(season)],
                           # 'm5_med': [m5_med],
                           'gap_max': [gap_max],
                           'gap_med': [gap_med],
                           'ebvofMW': [ebvofMW],
                           'cadence': [cadence],
                           'season_length': [season_length]})

        for key, val in m5_med.items():
            df['m5_med_{}'.format(key)] = val

        for key, val in Nvisits.items():
            df['N_{}'.format(key)] = val

        # for vv in ['x1', 'color', 'zlim', 'zlimp', 'zlimm', 'nsn_med', 'err_nsn_med']:
        for vv in ['x1', 'color', 'zlim', 'zmean', 'zpeak', 'nsn_zlim', 'nsn_zmean', 'nsn_zpeak']:
            for ko in ['faint', 'medium']:
                df['{}_{}'.format(vv, ko)] = [-1.0]

        for vv in ['cad_sn_mean', 'cad_sn_std', 'gap_sn_mean', 'gap_sn_std']:
            df[vv] = [-1]

        for ko in ['faint', 'medium']:
            df['status_{}'.format(ko)] = [int(errortype)]

        df['N_total'] = 0.
        return df

    def erroreffi(self, pixRA, pixDec, healpixID, season):
        """
        Method to return error df related to efficiencies

        Parameters
        --------------
        pixRA: float
          pixel RA
        pixDec: float
          pixel Dec
        healpixID: int
          healpix ID
        season: int
          season
        errortype: str
          type of error

        """
        return pd.DataFrame({'pixRA': [np.round(pixRA)],
                             'pixDec': [np.round(pixDec)],
                             'healpixID': [healpixID],
                             'season': [int(season)],
                             'x1': [-1.0],
                             'color': [-1.0],
                             'z': [-1.0],
                             'effi': [-1.0],
                             'effi_err': [-1.0],
                             'effi_var': [-1.0]})

    @ verbose_this('Estimate efficiencies')
    @ time_this('Efficiencies')
    def effidf(self, sn_tot, color_cut=0.04, **kwargs):
        """
        Method estimating efficiency vs z for a sigma_color cut

        Parameters
        ---------------
        sn_tot: pandas df
          data used to estimate efficiencies
        color_cut: float, opt
          color selection cut (default: 0.04)

        Returns
        ----------
        effi: pandas df with the following cols:
          season: season
          pixRA: RA of the pixel
          pixDec: Dec of the pixel
          healpixID: pixel ID
          x1: SN stretch
          color: SN color
          z: redshift
          effi: efficiency
          effi_err: efficiency error (binomial)
        """

        sndf = pd.DataFrame(sn_tot)

        listNames = ['season', 'pixRA', 'pixDec', 'healpixID', 'x1', 'color']
        groups = sndf.groupby(listNames)

        # estimating efficiencies
        effi = groups[['Cov_colorcolor', 'z']].apply(
            lambda x: self.effiObsdf(x, color_cut)).reset_index(level=list(range(len(listNames))))

        if self.verbose:
            print('efficiencies', effi)
        # this is to plot efficiencies and also sigma_color vs z
        if self.ploteffi:

            import matplotlib.pylab as plt
            fig, ax = plt.subplots()

            self.plot_old(ax, effi, 'effi', 'effi_err',
                          'Observing Efficiencies', ls='-')
            figb, axb = plt.subplots()
            sndf['sigma_color'] = np.sqrt(sndf['Cov_colorcolor'])
            self.plot_old(axb, sndf, 'sigma_color', None, '$\sigma_{color}$')

            plt.show()

            from sn_metrics.sn_plot_live import plotNSN_effi
            plotNSN_effi(effi, 'effi', 'effi_err',
                         'Observing Efficiencies', ls='-')
        return effi

    def plot_old(self, ax, effi, vary, erry=None, legy='', ls='None'):
        """
        Simple method to plot vs z

        Parameters
        --------------
        ax:
          axis where to plot
        effi: pandas df
          data to plot
        vary: str
          variable (column of effi) to plot
        erry: str, opt
          error on y-axis (default: None)
        legy: str, opt
          y-axis legend (default: '')

        """
        grb = effi.groupby(['x1', 'color'])
        yerr = None
        for key, grp in grb:
            x1 = grp['x1'].unique()[0]
            color = grp['color'].unique()[0]
            if erry is not None:
                yerr = grp[erry]
            ax.errorbar(grp['z'], grp[vary], yerr=yerr,
                        marker='o', label='(x1,color)=({},{})'.format(x1, color), lineStyle=ls)

        ftsize = 15
        ax.set_xlabel('z', fontsize=ftsize)
        ax.set_ylabel(legy, fontsize=ftsize)
        ax.xaxis.set_tick_params(labelsize=ftsize)
        ax.yaxis.set_tick_params(labelsize=ftsize)
        ax.legend(fontsize=ftsize)

    def zlimdf(self, grp, duration_z):
        """
        Method to estimate redshift limits

        Parameters
        --------------
        grp: pandas df group
          efficiencies to estimate redshift limits;
          columns:
           season: season
           pixRA: RA of the pixel
           pixDec: Dec of the pixel
           healpixID: pixel ID
           x1: SN stretch
           color: SN color
           z: redshift
           effi: efficiency
           effi_err: efficiency error (binomial)
        duration_z: pandas df with the following cols:
           season: season
           z: redshift
           T0_min: min daymax
           T0_max: max daymax
            season_length: season length

        Returns
        ----------
        pandas df with the following cols:
         zlimit: redshift limit
        zmean: mean redshift (weighted by NSN)
        zpeak: peak redshift (corresponding to max(NSN(z)))
         status: status of the processing

        """

        zlimit = 0.0
        zmean = 0.0
        zpeak = 0.0
        status = self.status['effi']

        # z range for the study
        zplot = np.arange(self.zmin, self.zmax, 0.01)

        # print(grp['z'], grp['effi'])

        if len(grp['z']) <= 3:
            return pd.DataFrame({'zlim': [zlimit],
                                 'zmean': [zmean],
                                 'zpeak': [zpeak],
                                 'status': [int(status)]})
            """
            return pd.DataFrame({'zlim': [zlimit],
                                 'zlimp': [zlimit],
                                 'zlimm': [zlimit],
                                 'status': [int(status)]})
            """
        # interpolate efficiencies vs z
        effiInterp = interp1d(
            grp['z'], grp['effi'], kind='linear', bounds_error=False, fill_value=0.)
        """
        effiInterp_plus = interp1d(
            grp['z'], grp['effi']+grp['effi_err'], kind='linear', bounds_error=False, fill_value=0.)
        effiInterp_minus = interp1d(
            grp['z'], grp['effi']-grp['effi_err'], kind='linear', bounds_error=False, fill_value=0.)
        """
        effiInterp_err = interp1d(
            grp['z'], grp['effi_err'], kind='linear', bounds_error=False, fill_value=0.)

        if self.zlim_coeff < 0.:
            # in that case zlim is estimated from efficiencies
            # first step: identify redshift domain with efficiency decrease
            zlimit, status = self.zlim_from_effi(effiInterp, zplot)

        else:
            """
            zlimit, zlimit_plus, zlimit_minus, status = self.zlim_from_cumul(
                grp, duration_z, effiInterp, effiInterp_err, zplot, rate='SN_rate')
            """
            zlimit, zmean, zpeak, status = self.zlim_from_cumul(
                grp, duration_z, effiInterp, effiInterp_err, zplot, rate='SN_rate')

        return pd.DataFrame({'zlim': [zlimit],
                             'zmean': [zmean],
                             'zpeak': [zpeak],
                             'status': [int(status)]})
        """
        return pd.DataFrame({'zlim': [zlimit],
                             'zlimp': [zlimit_plus],
                             'zlimm': [zlimit_minus],
                             'status': [int(status)]})
        """

    def zlim_from_cumul(self, grp, duration_z, effiInterp, effiInterp_err, zplot, rate='cte'):
        """
        Method to estimate the redshift limit from the cumulative
        The redshift limit is estimated to be the z value corresponding to:
        frac(NSN(z<zlimit))=zlimi_coeff

        Parameters
        ---------------
        grp: pandas group
          data to process
        duration_z: array
           duration as a function of the redshift
        effiInterp: interp1d
          interpolator for efficiencies
        zplot: interp1d
          interpolator for redshift values
        rate: str, opt
          rate to estimate the number of SN to estimate zlimit
          rate = cte: rate independent of z
          rate = SN_rate: rate from SN_Rate class

        Returns
        ----------
        zlimit: float
          the redshift limit
        status: str
          status of the estimation
        """

        if rate == 'SN_rate':
            # get rate
            season = np.median(grp['season'])
            idx = duration_z['season'] == season
            seas_duration_z = duration_z[idx]

            durinterp_z = interp1d(
                seas_duration_z['z'], seas_duration_z['season_length'], bounds_error=False, fill_value=0.)

            # estimate the rates and nsn vs z
            zz, rate, err_rate, nsn, err_nsn = self.rateSN(zmin=self.zmin,
                                                           zmax=self.zmax,
                                                           duration_z=durinterp_z,
                                                           # duration = np.mean(seas_duration_z['season_length']),
                                                           survey_area=self.pixArea,
                                                           account_for_edges=False)

            # rate interpolation
            rateInterp = interp1d(zz, nsn, kind='linear',
                                  bounds_error=False, fill_value=0)
            rateInterp_err = interp1d(zz, err_nsn, kind='linear',
                                      bounds_error=False, fill_value=0)

        else:
            # this is for a rate z-independent
            nsn = np.ones(len(zplot))
            rateInterp = interp1d(zplot, nsn, kind='linear',
                                  bounds_error=False, fill_value=0)
            rateInterp_err = interp1d(zplot, 0.01*nsn, kind='linear',
                                      bounds_error=False, fill_value=0)

        nsn_z = effiInterp(zplot)*rateInterp(zplot)
        nsn_cum = np.cumsum(nsn_z)
        if self.verbose:

            print('Estimating zlim', self.pixArea, self.zlim_coeff)
            print('season', duration_z[['z', 'season_length']])
            print('zplot', zplot)
            print('effis', effiInterp(zplot))
            print('rate', rateInterp(zplot))
            print('nsn', nsn_cum)

        nsn_cum_err = []
        for i in range(len(zplot)):
            siga = effiInterp_err(zplot[:i+1])*rateInterp(zplot[:i+1])
            sigb = effiInterp(zplot[:i+1])*rateInterp_err(zplot[:i+1])
            nsn_cum_err.append(np.cumsum(
                np.sqrt(np.sum(siga**2 + sigb**2))).item())

        if nsn_cum[-1] >= 1.e-5:
            nsn_cum_norm = nsn_cum/nsn_cum[-1]  # normalize
            nsn_cum_norm_err = nsn_cum_err/nsn_cum[-1]  # normalize
            zlim = interp1d(nsn_cum_norm, zplot,
                            bounds_error=False, fill_value=-1.)
            """
            zlim_plus = interp1d(nsn_cum_norm+nsn_cum_norm_err,
                                 zplot, bounds_error=False, fill_value=-1.)
            zlim_minus = interp1d(
                nsn_cum_norm-nsn_cum_norm_err, zplot, bounds_error=False, fill_value=-1.)
            """
            zlimit = zlim(self.zlim_coeff).item()
            """
            zlimit_minus = zlim_plus(self.zlim_coeff).item()
            zlimit_plus = zlim_minus(self.zlim_coeff).item()
            """

            status = self.status['ok']

            norm = np.cumsum(nsn_z)[-1]
            zmean = np.round(np.sum(nsn_z*zplot)/norm, 2)
            io = np.argmax(nsn_z)
            zpeak = np.round(zplot[io], 2)

            if self.ploteffi:
                from sn_metrics.sn_plot_live import plotNSN_cumul, plotNSN_z
                plotNSN_cumul(grp, nsn_cum_norm, nsn_cum_norm_err,
                              zplot, self.zlim_coeff, zlimit, zmean, zpeak)

                # plotNSN_z(grp, zplot, nsn_z, self.zlim_coeff,
                #          zlimit, zmean, zpeak)
        else:
            zlimit = 0.
            zpeak = 0
            zmean = 0
            status = self.status['low_effi']

        # return zlimit, zlimit_plus, zlimit_minus, status
        return zlimit, zmean, zpeak, status

    def zlim_from_effi(self, effiInterp, zplot):
        """
        Method to estimate the redshift limit from efficiency curves
        The redshift limit is defined here as the redshift value beyond
        which efficiency decreases up to zero.

        Parameters
        ---------------
        effiInterp: interpolator
         use to get efficiencies
        zplot: numpy array
          redshift values

        Returns
        -----------
        zlimit: float
          the redshift limit

        """

        # get efficiencies
        effis = effiInterp(zplot)
        if len(effis) < 1:
            return 0.0, self.status['low_effi']
        # select data with efficiency decrease
        idx = np.where(np.diff(effis) < -0.005)
        if len(zplot[idx]) < 1:
            return 0.0, self.status['low_effi']

        z_effi = np.array(zplot[idx], dtype={
            'names': ['z'], 'formats': [np.float]})
        # from this make some "z-periods" to avoid accidental zdecrease at low z
        z_gap = 0.05
        seasoncalc = np.ones(z_effi.size, dtype=int)
        diffz = np.diff(z_effi['z'])
        flag = np.where(diffz > z_gap)[0]

        if len(flag) > 0:
            for i, indx in enumerate(flag):
                seasoncalc[indx+1:] = i+2
        z_effi = rf.append_fields(z_effi, 'season', seasoncalc)

        # now take the highest season (end of the efficiency curve)
        idd = z_effi['season'] == np.max(z_effi['season'])
        zlimit = np.min(z_effi[idd]['z'])

        return zlimit, self.status['ok']

    def nsn_typedf(self, grp, x1, color, effi_tot, duration_z, search=True):
        """
        Method to estimate the number of supernovae for a given type of SN

        Parameters
        --------------
        grp: pandas series with the following infos:
         pixRA: pixelRA
         pixDec: pixel Dec
         healpixID: pixel ID
         season: season
         x1: SN stretch
         color: SN color
         zlim: redshift limit
         status: processing status
        x1, color: SN params to estimate the number
        effi_tot: pandas df with columns:
           season: season
           pixRA: RA of the pixel
           pixDec: Dec of the pixel
           healpixID: pixel ID
           x1: SN stretch
           color: SN color
           z: redshift
           effi: efficiency
           effi_err: efficiency error (binomial)
        duration_z: pandas df with the following cols:
           season: season
           z: redshift
           T0_min: min daymax
           T0_max: max daymax
            season_length: season length

        Returns
        ----------
        nsn: float
           number of supernovae
        """

        # get rate
        season = np.median(grp['season'])
        idx = duration_z['season'] == season
        seas_duration_z = duration_z[idx]

        durinterp_z = interp1d(
            seas_duration_z['z'], seas_duration_z['season_length'], bounds_error=False, fill_value=0.)

        if search:
            effisel = effi_tot.loc[lambda dfa: (
                dfa['x1'] == x1) & (dfa['color'] == color), :]
        else:
            effisel = effi_tot

        """
        nsn, var_nsn = self.nsn(
            effisel, grp['zlim'], grp['zlimp'], grp['zlimm'], durinterp_z)
        """
        zlims = [grp['zlim'], grp['zmean'], grp['zpeak']]
        nsn = self.nsn_simple(effisel, zlims, durinterp_z)

        """
        nsn, var_nsn = self.nsn(
            effisel, grp['zlim'], grp['zlimp'], grp['zlimm'], seas_duration_z)
        """
        return nsn

    def nsn_typedf_weight(self, effi, duration_z, zlims):
        """
        Method to estimate the number of supernovae weigthed

        Parameters
        --------------
        effi: pandas df with efficiencies:
          effi: efficiencies
          effi_err: efficiency errors
          z: redshift
          weight: weight
        duration_z: pandas df with the following cols:
          season: season
          z: redshift
          T0_min: min daymax
          T0_max: max daymax
          season_length: season length
        zlims: pandas df with the cols:
          pixRA: RA pixel
          pixDec:  Dec pixel
          healpixID: pixel ID
          season: season
          x1: SN stretch
          color: SN color
          zlim: redshift limit
          status: status of the processing
          nsn_med: number of medium supernovae

        Returns
        ----------
        pandas df with weighted number of supernovae and variances

        """
        x1 = effi.name[0]
        color = effi.name[1]
        weight = np.mean(effi['weight'])
        nsn, var_nsn = zlims.apply(lambda x: self.nsn_typedf(
            x, x1, color, effi, duration_z, search=True), axis=1, result_type='expand').T.values

        return pd.DataFrame({'nsn': [nsn*weight], 'var_nsn': [var_nsn*weight*weight]})

    @ verbose_this('Estimating the total number of supernovae')
    @ time_this('Total number of supernovae')
    def nsn_tot(self, effi, zlim, duration_z, **kwargs):
        """
        Method to estimate the total number of supernovae

        Parameters
        ---------------
        effi: pandas df with the following cols:
          season: season
          pixRA: RA of the pixel
          pixDec: Dec of the pixel
          healpixID: pixel ID
          x1: SN stretch
          color: SN color
          z: redshift
          effi: efficiency
          effi_err: efficiency error (binomial)
        zlim: pandas df with the cols:
          pixRA: RA pixel
          pixDec:  Dec pixel
          healpixID: pixel ID
          season: season
          x1: SN stretch
          color: SN color
          zlim: redshift limit
          status: status of the processing
          nsn_med: number of medium supernovae
       duration_z: pandas df with the following cols:
          season: season
          z: redshift
          T0_min: min daymax
          T0_max: max daymax
          season_length: season length


        Returns
        -----------
        nsn, var_nsn: float,float
          total number of supernovae and associated variance

        """

        # the first thing is to interpolate efficiencies to have a regular grid
        zvals = np.arange(0.0, 1.2, 0.05)

        # take the medium SN as witness
        # if not enough measurements for this type -> return -1
        idx = np.abs(effi['x1']) < 1.e-5
        idx &= np.abs(effi['color']) < 1.e-5
        if len(effi[idx]['z']) < 3 or np.mean(effi[idx]['effi']) < 1.e-5:
            return 0.0, 0.0

        # get interpolated efficiencies for the set of reference SN
        effi_grp = effi.groupby(['x1', 'color'])[['x1', 'color', 'effi', 'effi_err', 'effi_var', 'z']].apply(
            lambda x: self.effi_interp(x, zvals)).reset_index().to_records(index=False)

        # print('hello', self.x1_color_dist)

        if self.proxy_level == 1:
            grpdf = pd.DataFrame(effi_grp)
            effidf = pd.DataFrame(self.x1_color_dist)

            totdf = grpdf.merge(
                effidf, left_on=['x1', 'color'], right_on=['x1', 'color'])

            totdf = totdf.rename(columns={'weight_tot': 'weight'})
            # print(totdf[['x1', 'color', 'weight_tot']])
            # print(totdf.columns)
            season = np.median(zlim['season'])
            idxb = duration_z['season'] == season
            duration = duration_z[idxb]

            # get the weighted number of supernovae
            dfsn = totdf.groupby(['x1', 'color']).apply(lambda x: self.nsn_typedf_weight(
                x, duration_z[idxb], zlim))

            nsn_tot = dfsn['nsn']
            var_tot = dfsn['var_nsn']

            return nsn_tot.sum(axis=0), var_tot.sum(axis=0)

        # Now construct the griddata

        # get x1, color and z values
        x1_vals = np.unique(effi_grp['x1'])
        color_vals = np.unique(effi_grp['color'])
        z_vals = np.unique(effi_grp['z'])

        n_x1 = len(x1_vals)
        n_color = len(color_vals)
        n_z = len(z_vals)

        # build the griddata - be careful of the order here
        index = np.lexsort((effi_grp['z'], effi_grp['color'], effi_grp['x1']))
        effi_resh = np.reshape(effi_grp[index]['effi'], (n_x1, n_color, n_z))
        # effi_resh = effi_grp[index]['effi']
        effi_var_resh = np.reshape(
            effi_grp[index]['effi_var'], (n_x1, n_color, n_z))

        effi_grid = RegularGridInterpolator(
            (x1_vals, color_vals, z_vals), effi_resh, method='linear', bounds_error=False, fill_value=0.)

        effi_var_grid = RegularGridInterpolator(
            (x1_vals, color_vals, z_vals), effi_var_resh, method='linear', bounds_error=False, fill_value=0.)

        nsnTot = None
        ip = -1
        weight_sum = 0.

        # select only sn with |x1|<2 and |color|<0.2
        idx = np.abs(self.x1_color_dist['x1']) <= 2
        idx &= np.abs(self.x1_color_dist['color']) <= 0.2

        # now estimate efficiencies from this griddata

        time_ref = time.time()
        x1_tile = np.repeat(self.x1_color_dist[idx]['x1'], len(zvals))
        color_tile = np.repeat(self.x1_color_dist[idx]['color'], len(zvals))
        z_tile = np.tile(zvals, len(self.x1_color_dist[idx]))
        weight_tile = np.repeat(
            self.x1_color_dist[idx]['weight_tot'], len(zvals))

        df_test = pd.DataFrame()

        df_test.loc[:, 'effi'] = effi_grid((x1_tile, color_tile, z_tile))
        df_test.loc[:, 'effi_var'] = effi_var_grid(
            (x1_tile, color_tile, z_tile))
        df_test.loc[:, 'x1'] = np.round(x1_tile, 2)
        df_test.loc[:, 'color'] = np.round(color_tile, 2)
        df_test.loc[:, 'z'] = z_tile
        # df_test.loc[:, 'weight'] = np.round(weight_tile, 2)
        df_test.loc[:, 'weight'] = weight_tile
        season = np.median(zlim['season'])
        idxb = duration_z['season'] == season

        # this is a check
        """
        idx = np.abs(df_test['x1']) < 1.e-8
        idx &= np.abs(df_test['color']) < 1.e-8
        print('tttt', df_test[idx][['x1', 'color', 'z', 'effi', 'weight']])

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        idxf = np.abs(effi_grp['x1']) < 1.e-8
        idxf &= np.abs(effi_grp['color']) < 1.e-8
        print('tttt', effi_grp[idxf][['x1', 'color', 'z', 'effi']])
        ax.plot(df_test[idx]['z'], df_test[idx]['effi'], 'ko')
        ax.plot(effi_grp[idxf]['z'], effi_grp[idxf]['effi'], 'r*')
        test_effi = effi_grid((
            effi_grp['x1'][idxf], effi_grp['color'][idxf], effi_grp['z'][idxf]))
        # ax.plot(effi_grp['z'][idxf], test_effi,
        #        'b.', mfc='None')

        plt.show()
        """
        # get the weighted number of supernovae
        resdf = df_test.groupby(['x1', 'color']).apply(lambda x: self.nsn_typedf_weight(
            x, duration_z[idxb], zlim))

        return resdf['nsn'].sum(axis=0), resdf['var_nsn'].sum(axis=0)

    def effi_interp(self, grp, zvals):
        """
        Method to interpolate efficiencies and associated errors

        Parameters
        --------------
        grp: pandas df with the following cols:
          x1: SN stretch
          color: SN color
          effi: efficiency
          effi_err: efficiency error
           z: redshift
        zvals: list(float)
          list of redshift for interpolation

        Returns
        ----------
        pandas df with the cols:
          effi: efficiency
          effi_err: efficiency error
          z: redshift value

        """

        interp = interp1d(grp['z'], grp['effi'],
                          bounds_error=False, fill_value=0.)
        interp_err = interp1d(grp['z'], grp['effi_err'],
                              bounds_error=False, fill_value=0.)
        interp_var = interp1d(grp['z'], grp['effi_var'],
                              bounds_error=False, fill_value=0.)

        return pd.DataFrame({'effi': interp(zvals),
                             'effi_err': interp_err(zvals),
                             'effi_var': interp_var(zvals),
                             'z': zvals})

    def nsn(self, effi, zlim, zlimp, zlimm, duration_z):
        """
        Method to estimate the number of supernovae

        Parameters
        --------------
        effi: pandas df grp of efficiencies
          season: season
          pixRA: RA of the pixel
          pixDec: Dec of the pixel
          healpixID: pixel ID
          x1: SN stretch
          color: SN color
          z: redshift
          effi: efficiency
          effi_err: efficiency error (binomial)
        zlim: float
          redshift limit value
        duration_z: pandas df with the following cols:
           season: season
           z: redshift
           T0_min: min daymax
           T0_max: max daymax
            season_length: season length

        Returns
        ----------
        nsn, var_nsn : float
          number of supernovae (and variance) with z<zlim

        """

        if zlim < 1.e-3:
            return -1.0, -1.0

        dz = 0.001
        zplot = list(np.arange(self.zmin, self.zmax, dz))
        # interpolate efficiencies vs z
        effiInterp = interp1d(
            effi['z'], effi['effi'], kind='linear', bounds_error=False, fill_value=0.)
        # interpolate variance efficiency vs z
        effiInterp_err = interp1d(
            effi['z'], effi['effi_err'], kind='linear', bounds_error=False, fill_value=0.)

        # estimate the cumulated number of SN vs z
        zz, rate, err_rate, nsn, err_nsn = self.rateSN(zmin=self.zmin,
                                                       zmax=self.zmax,
                                                       dz=dz,
                                                       duration_z=duration_z,
                                                       # duration = np.mean(duration_z['season_length']),
                                                       survey_area=self.pixArea,
                                                       account_for_edges=False)

        # rate interpolation
        rateInterp = interp1d(zz, nsn, kind='linear',
                              bounds_error=False, fill_value=0)
        rateInterp_err = interp1d(zz, err_nsn, kind='linear',
                                  bounds_error=False, fill_value=0)
        nsn_cum = np.cumsum(effiInterp(zplot)*nsn)

        err_cum = []
        """
        err_cum = np.cumsum(nsn)*effiInterp(zplot) * \
            (1.-effiInterp(zplot))+np.cumsum(err_nsn**2)
        """
        for i in range(len(zplot)):
            erra = effiInterp_err(zplot[:i+1])*rateInterp(zplot[:i+1])
            errb = effiInterp(zplot[:i+1])*rateInterp_err(zplot[:i+1])
            err_cum.append(np.cumsum(
                np.sqrt(np.sum(erra**2 + errb**2))).item())

        nsn_interp = interp1d(
            zplot, nsn_cum, bounds_error=False, fill_value=0.)
        err_interp = interp1d(
            zplot, err_cum, bounds_error=False, fill_value=0.)
        """
        # estimate numbers if we had efficiencies equal to one...
        nsn_all = interp1d(zplot, np.cumsum(nsn))
        err_all = interp1d(zplot, np.cumsum(err_nsn*err_nsn))
        print('ici all', nsn_all(zlim), np.sqrt(err_all(zlim)))

        """
        nsn = nsn_interp(zlim).item()
        err_nsn = err_interp(zlim).item()
        nsnp = nsn_interp(zlimp).item()
        nsnm = nsn_interp(zlimm).item()

        err_nsn_zlim = (nsnp-nsnm)/2.
        if nsnp < 1.e-5:
            err_nsn_zlim = (nsn-nsnm)

        err_nsn = np.sqrt(err_nsn**2+err_nsn_zlim**2)
        return [nsn, err_nsn]

    def nsn_simple(self, effi, zlim, duration_z):
        """
        Method to estimate the number of supernovae corresponding to zlim

        Parameters
        --------------
        effi: pandas df grp of efficiencies
          season: season
          pixRA: RA of the pixel
          pixDec: Dec of the pixel
          healpixID: pixel ID
          x1: SN stretch
          color: SN color
          z: redshift
          effi: efficiency
          effi_err: efficiency error (binomial)
        zlim: float
          redshift limit value
        duration_z: pandas df with the following cols:
           season: season
           z: redshift
           T0_min: min daymax
           T0_max: max daymax
            season_length: season length

        Returns
        ----------
        nsn, var_nsn : float
          number of supernovae (and variance) with z<zlim

        """
        if zlim[0] < 1.e-3:
            return [-1.0]*len(zlim)

        dz = 0.001
        zplot = list(np.arange(self.zmin, self.zmax, dz))
        # interpolate efficiencies vs z
        effiInterp = interp1d(
            effi['z'], effi['effi'], kind='linear', bounds_error=False, fill_value=0.)
        # interpolate variance efficiency vs z
        effiInterp_err = interp1d(
            effi['z'], effi['effi_err'], kind='linear', bounds_error=False, fill_value=0.)

        # estimate the cumulated number of SN vs z
        zz, rate, err_rate, nsn, err_nsn = self.rateSN(zmin=self.zmin,
                                                       zmax=self.zmax,
                                                       dz=dz,
                                                       duration_z=duration_z,
                                                       # duration = np.mean(duration_z['season_length']),
                                                       survey_area=self.pixArea,
                                                       account_for_edges=False)
        # rate interpolation
        rateInterp = interp1d(zz, nsn, kind='linear',
                              bounds_error=False, fill_value=0)
        nsn_cum = np.cumsum(effiInterp(zplot)*nsn)

        nsn_interp = interp1d(
            zplot, nsn_cum, bounds_error=False, fill_value=0.)

        nsn_res = nsn_interp(zlim)
        return nsn_res

    def seasonInfo(self, grp):
        """
        Method to estimate seasonal info (cadence, season length, ...)

        Parameters
        --------------
        grp: pandas df group

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

        """
        for band in 'ugrizy':
            Nvisits = 0
            idx = grp[self.filterCol] == band
            if len(grp[idx]) > 0:
                Nvisits = grp[idx][self.nexpCol].sum()
            df['Nvisits_{}'.format(band)] = Nvisits
        """

        if self.obsstat:
            grpb = grp.groupby(['night']).apply(
                lambda x: pd.DataFrame({'filter': [''.join(sorted(x[self.filterCol]*x[self.nexpCol].astype(int).values))]})).reset_index()

            dfcomb = grpb.groupby('filter').apply(
                lambda x: pd.DataFrame(({'Nvisits': [len(x)]}))).reset_index()

            dfcomb = dfcomb.sort_values(by=['Nvisits'], ascending=False)

            for vv in self.bandstat:
                count = 0
                for io, row in dfcomb.iterrows():
                    for b in vv:
                        ca = row['filter'].count(b)
                        count += row['Nvisits']*np.min([1, ca])/len(vv)
                df['N_{}'.format(vv)] = count

            """
            print(count)



            filtcombi = ''
            for i, row in dfcomb.iterrows():
                filtcombi += '{}*{}/'.format(row['Nvisits'],row['filter'])

            df['filters_night'] = filtcombi
            """
            """
            # old code with bandstat
            for val in self.bandstat:
                # print(val, grpb[self.filterCol].str.count(val).sum())
                idx = grpb[self.filterCol]==val
                # df['N_{}'.format(val)] = grpb[self.filterCol].str.count(val).sum()
                df['N_{}'.format(val)] = len(grpb[idx])
            """

        if len(grp) > 5:
            # to = grp.groupby(['night'])[self.mjdCol].median().sort_values()
            # df['cadence'] = np.mean(to.diff())
            nights = np.sort(grp['night'].unique())
            diff = np.asarray(nights[1:]-nights[:-1])
            df['cadence'] = np.median(diff).item()

        return df

    @ verbose_this('Selecting')
    @ time_this('Selection')
    def process(self, tab, **kwargs):
        """
        Method to process LC: sigma_color estimation and LC selection

        Parameters
        --------------
        tab: pandas df of LC points with the following cols:
          flux:  flux
          fluxerr: flux error
          phase:  phase
          snr_m5: Signal-to-Noise Ratio
          time: time (MJD)
          mag: magnitude
          m5:  five-sigma depth
          magerr: magnitude error
          exposuretime: exposure time
          band: filter
          zp:  zero-point
          season: season number
          healpixID: pixel ID
          pixRA: pixel RA
          pixDec: pixel Dec
          z: redshift
          daymax: T0
          flux_e_sec: flux (in photoelec/sec)
          flux_5: 5-sigma flux (in photoelec/sec)
          F_x0x0, ...F_colorcolor: Fisher matrix elements
          x1: x1 SN
          color: color SN
          n_aft: number of LC points before daymax
          n_bef: number of LC points after daymax
          n_phmin: number of LC points with a phase<-5
          n_phmax:  number of LC points with a phase > 20

        Returns
        ----------

        """

        # remove points with too high errormodel
        tab['snr_model'] = tab['fluxerr_model']/tab['flux']
        if self.errmodrel > 0.:
            tab = self.select_error_model(tab)

        if self.verbose:
            print('after sel errmodel', len(tab))
            print(tab[['band', 'snr_m5',
                  'fluxerr_photo']])

        # select LC points with min snr
        idx = tab['snr_m5'] >= self.snr_min
        tab = tab[idx]

        if self.verbose:
            print('LC', tab.columns)
            print(tab[['band', 'snr_m5',
                  'fluxerr_photo']])
        # now groupby
        tab = tab.round({'pixRA': 4, 'pixDec': 4, 'daymax': 5,
                         'z': 3, 'x1': 2, 'color': 2})
        groups = tab.groupby(
            ['pixRA', 'pixDec', 'daymax', 'season', 'z', 'healpixID', 'x1', 'color'])

        tosum = []
        for ia, vala in enumerate(self.params):
            for jb, valb in enumerate(self.params):
                if jb >= ia:
                    tosum.append('F_'+vala+valb)
        # tosum += ['n_aft', 'n_bef', 'n_phmin', 'n_phmax']
        tosum += ['n_phmin', 'n_phmax']
        # apply the sum on the group
        # sums = groups[tosum].sum().reset_index()

        sums = groups.apply(
            lambda x: self.sumIt(x, tosum)).reset_index()

        if self.verbose:
            print('jjj', sums)
            idx = np.abs(sums['daymax']-60881.94101) < 1.e-5
            print('for sel', sums[idx])
        # select LC according to the number of points bef/aft peak
        idx = sums['n_aft'] >= self.n_aft
        idx &= sums['n_bef'] >= self.n_bef
        idx &= sums['n_phmin'] >= self.n_phase_min
        idx &= sums['n_phmax'] >= self.n_phase_max

        if self.verbose:
            print('selection parameters', self.n_bef,
                  self.n_aft, self.n_phase_min, self.n_phase_max)
            print('goodsn', len(sums.loc[idx]))
        finalsn = pd.DataFrame()
        goodsn = pd.DataFrame(sums.loc[idx])

        # estimate the color for SN that passed the selection cuts
        if len(goodsn) > 0:
            goodsn.loc[:, 'Cov_colorcolor'] = CovColor(goodsn).Cov_colorcolor
            finalsn = pd.concat([finalsn, goodsn], sort=False)
            if self.verbose:
                print('goodSN', len(goodsn))
                idb = goodsn['z'] <= 0.1
                print('goodSN', goodsn[idb])

        badsn = pd.DataFrame(sums.loc[~idx])

        # Supernovae that did not pass the cut have a sigma_color=10
        if len(badsn) > 0:
            badsn.loc[:, 'Cov_colorcolor'] = 100.
            finalsn = pd.concat([finalsn, badsn], sort=False)
            if self.verbose:
                print('finalsn', len(finalsn))
                idx = finalsn['z'] <= 0.1
                print('finalsn', finalsn[idx])

        return finalsn

    def sumIt_nonight(self, grpa, tosum):
        """
        Method to estimate the sum of some of the group col
        and also of the number of epochs before and after peak

        Parameters
        --------------
        grp: pandas df
          pandas group to process
        tosum: list(str)
          list of the columns to sum

        Returns
        ----------
        pandas df with the summed columns and the estimation of the number of epochs (n_bef and n_aft)

        """

        grp = pd.DataFrame(grpa)
        grp = grp.reset_index()
        grp = grp.sort_values(by=['time'])

        gap = 12./24.  # in days
        df_time = grp['time'].diff()
        index = list((df_time[df_time > gap].index))
        index.insert(0, grp.index.min())
        index.insert(len(index), grp.index.max())
        grp['epoch'] = 1

        for i in range(0, len(index)-1):
            # grp.loc[index[i]: index[i+1], 'epoch'] = i+1
            grp.loc[index[i]:, 'epoch'] = i+1

        """
        print('iii', grp.index)
        print('jjj', index, grp.index.min())
        for i in range(0, len(index)-1):
            grp.loc[index[i]: index[i+1], 'epoch'] = i+1
        """

        # grp = grp.sort_values(by=['epoch'])
        idx = grp['phase'] <= 0
        sel = grp[idx]

        n_bef = len(grp[idx]['epoch'].unique())

        idx = grp['phase'] >= 0
        n_aft = len(grp[idx]['epoch'].unique())

        sums = grpa[tosum].sum()
        sums['n_bef'] = n_bef
        sums['n_aft'] = n_aft

        # print('result', n_bef, n_aft, sums['n_phmin'], sums['n_phmax'])
        return sums

    def sumIt(self, grp, tosum):
        """
        Method to estimate the sum of some of the group col
        and also of the number of epochs before and after peak

        Parameters
        --------------
        grp: pandas df
          pandas group to process
        tosum: list(str)
          list of the columns to sum

        Returns
        ----------
        pandas df with the summed columns and the estimation of the number of epochs (n_bef and n_aft)

        """
        sums = grp[tosum].sum()

        grp['night'] = grp['time']-self.mjd_LSST_Start+1.
        grp['night'] = grp['night'].astype(int)

        """
        idx = np.abs(grp['daymax']-61016.17090) < 1.e-3
        if len(grp[idx]) > 0:
            sel = grp[idx]
            sel = sel.sort_values(by='phase')
            print('aoooo', sel[['phase', 'time', 'night', 'snr_m5']])
        """

        grp = grp.sort_values(by=['night'])
        idx = grp['phase'] <= 0
        sel = grp[idx]

        n_bef = len(grp[idx]['night'].unique())

        idx = grp['phase'] >= 0
        n_aft = len(grp[idx]['night'].unique())

        sums['n_bef'] = n_bef
        sums['n_aft'] = n_aft

        # print('result', n_bef, n_aft, sums['n_phmin'], sums['n_phmax'])
        return sums

    def effiObsdf(self, data, color_cut=0.04):
        """
        Method to estimate observing efficiencies for supernovae

        Parameters
        --------------
        data: pandas df - grp
          data to process

        Returns
        ----------
        pandas df with the following cols:
          - cols used to make the group
          - effi, effi_err: observing efficiency and associated error

        """

        if self.verbose:
            data = data.sort_values(by=['z'])
            print('effiobsdf', data)
        # reference df to estimate efficiencies
        df = data.loc[lambda dfa:  np.sqrt(dfa['Cov_colorcolor']) < 100000., :]

        # selection on sigma_c<= 0.04
        df_sel = df.loc[lambda dfa:  np.sqrt(
            dfa['Cov_colorcolor']) <= color_cut, :]

        # make groups (with z)
        group = df.groupby('z')
        group_sel = df_sel.groupby('z')

        # Take the ratio to get efficiencies
        rb = (group_sel.size()/group.size())
        err = np.sqrt(rb*(1.-rb)/group.size())
        var = rb*(1.-rb)*group.size()

        rb = rb.array
        err = err.array
        var = var.array

        rb[np.isnan(rb)] = 0.
        err[np.isnan(err)] = 0.
        var[np.isnan(var)] = 0.

        return pd.DataFrame({group.keys: list(group.groups.keys()),
                             'effi': rb,
                             'effi_err': err,
                             'effi_var': var})

    @ verbose_this('Simulation SN')
    @ time_this('Simulation SN')
    def gen_LC_SN(self, obs, ebvofMW, gen_par, **kwargs):
        """
        Method to simulate LC and supernovae

        Parameters
        ---------------
        obs: numpy array
          array of observations (from scheduler)
        ebvofMW: float
           e(B-V) of MW for dust effects
        gen_par: numpy array
          array of parameters for simulation


        """

        time_ref = time.time()
        # LC estimation

        sn_tot = pd.DataFrame()
        lc_tot = pd.DataFrame()
        sn_info = pd.DataFrame()
        for key, vals in self.lcFast.items():
            time_refs = time.time()
            gen_par_cp = np.copy(gen_par)
            if key == (-2.0, 0.2):
                idx = gen_par_cp['z'] < 0.9
                #idx &= np.abs(gen_par_cp['daymax']-61016.17090) < 1.e-5
                gen_par_cp = gen_par_cp[idx]

            lc = vals(obs, ebvofMW, gen_par_cp, bands='grizy')
            """
            if key == (-2.0, 0.2):
                obs.sort(order=self.mjdCol)
                print('obs', obs[[self.mjdCol, self.filterCol]])
                self.plotLC_debug(lc, daymax=61016.17090)
            """
            tt = lc.groupby(['z', 'daymax']).apply(
                lambda x: self.sn_cad_gap(x)).reset_index()
            sn_info = tt.groupby(['z']).apply(
                lambda x: self.sn_cad_gap_sum(x)).reset_index()

            if self.ploteffi and self.fig_for_movie and len(lc) > 0 and key == (-2.0, 0.2):
                for season in np.unique(obs['season']):
                    idxa = obs['season'] == season
                    idxb = lc['season'] == season
                    idxc = gen_par['season'] == season
                    self.plotter.plotLoop(self.healpixID, season,
                                          obs[idxa], lc[idxb], gen_par[idxc])

            if self.verbose:
                print('End of simulation', key, time.time()-time_refs)
            """
            if self.ploteffi and len(lc) > 0:
                self.plotLC(lc)
            """
            # if self.outputType == 'lc':
            lc_tot = pd.concat((lc_tot, lc), sort=False)
            if self.verbose:
                print('End of simulation after concat',
                      key, time.time()-time_refs)

            # estimate SN

            sn = pd.DataFrame()
            if len(lc) > 0:
                # sn = self.process(Table.from_pandas(lc))
                sn = self.process(pd.DataFrame(
                    lc), verbose=self.verbose, timer=self.timer)

            if self.verbose:
                print('End of supernova', time.time()-time_refs)

            """
            if sn is not None:
                if sn_tot is None:
                    sn_tot = sn
                else:
                    sn_tot = np.concatenate((sn_tot, sn))
            """
            if not sn.empty:
                sn_tot = pd.concat([sn_tot, pd.DataFrame(sn)], sort=False)

        if self.verbose:
            print('End of supernova - all', time.time()-time_ref)

        return sn_tot, lc_tot, sn_info

    def sn_cad_gap_sum(self, grp):

        res = pd.DataFrame()
        if len(grp) > 0:
            res = pd.DataFrame({'cad_sn_mean': [grp['cad_sn'].mean()],
                                'cad_sn_std': [grp['cad_sn'].std()],
                               'gap_sn_mean': [grp['gap_sn'].mean()],
                                'gap_sn_std': [grp['gap_sn'].std()]})

        return res

    def sn_cad_gap(self, grp):
        """
        Method to estimate cadence and gap for a set od grp points

        Parameters
        --------------
        grp: pandas group
          data to process


        """
        nights = np.unique(grp[self.nightCol])

        nights.sort()
        diff = np.diff(nights)
        if len(diff) >= 2:
            return pd.DataFrame({'cad_sn': [np.median(diff)], 'gap_sn': [np.max(diff)]})
        else:
            return pd.DataFrame()

    def plotLC(self, lc, zref=0.5):
        """
        Method to plot LC

        Parameters
        --------------
        lc: pandas df of lc points
        zref: redshift value chosen for the display

        """

        import matplotlib.pyplot as plt

        if self.verbose:
            print('lc', lc.columns)
        lc = lc.round({'daymax': 6})
        sel = lc[np.abs(lc['z']-zref) < 1.e-5]
        # sel = sel[sel['band']=='LSST::g']

        # print(sel['daymax'].unique())
        fig, ax = plt.subplots(ncols=2, nrows=3)
        pos = dict(
            zip('ugrizy', [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]))

        fig.suptitle('(x1,color)=({},{}) - z={}'.format(
            sel['x1'].unique().item(), sel['color'].unique().item(), np.round(zref, 2)))
        for band in sel['band'].unique():
            idxb = sel['band'] == band
            selb = sel[idxb].to_records(index=False)
            ix = pos[band.split(':')[-1]][0]
            iy = pos[band.split(':')[-1]][1]
            for daymax in np.unique(selb['daymax']):
                selc = selb[selb['daymax'] == daymax]
                # ax[ix][iy].plot(selc['phase'], selc['flux_e_sec'])
                ax[ix][iy].errorbar(selc['phase'], selc['flux_e_sec'],
                                    yerr=selc['flux_e_sec']/selc['snr_m5'])

        plt.show()

    @ verbose_this('Estimating redshift limits')
    @ time_this('redshift limits')
    def zlims(self, effi_seasondf, dur_z, groupnames, **kwargs):
        """
        Method to estimate redshift limits

        Parameters
        --------------
        effi_seasondf: pandas df
            season: season
          pixRA: RA of the pixel
          pixDec: Dec of the pixel
          healpixID: pixel ID
          x1: SN stretch
          color: SN color
          z: redshift
          effi: efficiency
          effi_err: efficiency error (binomial)
        dur_z: pandas df with the following cols:
           season: season
           z: redshift
           T0_min: min daymax
           T0_max: max daymax
            season_length: season length
        groupnames: list(str)
          list of columns to use to define the groups

        Returns
        ----------
        pandas df with the following cols: pixRA: RA of the pixel
           pixDec: Dec of the pixel
           healpixID: pixel ID
           season: season number
           x1: SN stretch
           color: SN color
           zlim: redshift limit
           status: status of the processing
        """

        res = effi_seasondf.groupby(groupnames).apply(
            lambda x: self.zlimdf(x, dur_z)).reset_index(level=list(range(len(groupnames))))

        return res

    def select_error_model(self, grp):
        """
        function to select LCs

        Parameters
        ---------------
        grp : pandas df
          lc to consider

        Returns
        ----------
        lc with filtered values (pandas df)

       """

        lc = Table.from_pandas(grp)
        if self.errmodrel < 0.:
            return lc.to_pandas()

        # first: select iyz bands

        bands_to_keep = []

        lc_sel = Table()
        for b in 'izy':
            bands_to_keep.append('LSST::{}'.format(b))
            idx = lc['band'] == 'LSST::{}'.format(b)
            lc_sel = vstack([lc_sel, lc[idx]])

        # now apply selection on g band for z>=0.25
        sel_g = self.sel_band(lc, 'g', 0.25)

        # now apply selection on r band for z>=0.6
        sel_r = self.sel_band(lc, 'r', 0.6)

        lc_sel = vstack([lc_sel, sel_g])
        lc_sel = vstack([lc_sel, sel_r])

        return lc_sel.to_pandas()

    def sel_band(self, tab, b, zref):
        """
        Method to perform selections depending on the band and z

        Parameters
        ---------------
        tab: astropy table
          lc to process
        b: str
          band to consider
        zref: float
           redshift below wiwh the cut wwill be applied

        Returns
        ----------
        selected lc
        """

        idx = tab['band'] == 'LSST::{}'.format(b)
        sel = tab[idx]
        if len(sel) == 0:
            return Table()

        ida = sel['z'] < zref

        idb = sel['z'] >= zref
        idb &= sel['fluxerr_model']/sel['flux'] <= self.errmodrel

        tabres = vstack([sel[idb], sel[ida]])

        return tabres
        """
        if z >= zref:
            idb = sel['fluxerr_model']/sel['flux'] <= self.errmodrel
            selb = sel[idb]
            return selb

        return sel
        """

    def plotLC_debug(self, lc, daymax=60896.88112):
        """
        Method to plot LC for a given daymax (debug purpose)

        Parameters
        --------------
        lc: pandas df
          set of light curves
        daymax: float, opt
          T0 value (default: 60896.88112)

        """
        print('bands', np.unique(lc['band']))
        ido = np.abs(lc['daymax']-daymax) < 1.e-5
        ido &= np.abs(lc['x1']+2.0) < 1.e-5
        ido &= lc['snr_m5'] >= 1
        lcsel = lc[ido]
        lcsel = lcsel.sort_values(by=['phase'])
        print('light curve ',
              lcsel[['phase', 'flux', 'band', 'night', 'daymax', 'snr_m5', 'time']])
        """
        for ir, row in lcsel.iterrows():
            print(row['phase'], row['flux'],
                  row['band'], row['night'], row['z'], row['x1'], row['daymax'])
            print('light curve ',
                  lcsel[['phase', 'flux', 'band', 'night', 'daymax']])
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(lcsel['phase'], lcsel['flux'], 'ko')
        plt.show()
