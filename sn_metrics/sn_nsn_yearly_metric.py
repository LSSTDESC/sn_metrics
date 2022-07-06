import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from sn_stackers.coadd_stacker import CoaddStacker
import yaml
import os
from sn_tools.sn_calcFast import LCfast, CovColor
from sn_tools.sn_telescope import Telescope
from astropy.table import Table, vstack, Column
import time
import pandas as pd
from scipy.interpolate import interp1d
from sn_tools.sn_rate import SN_Rate
# from functools import wraps
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from sn_metrics.sn_plot_live import Plot_NSN_metric


class SNNSNYMetric(BaseMetric):
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
    timeIt: bool, opt
      to estimate processing time per pixel (default: False)
    """

    def __init__(self, lc_reference, dustcorr,
                 metricName='SNNSNYMetric',
                 mjdCol='observationStartMJD', RACol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', seeingCol='seeingFwhmEff', season=[-1], coadd=True, zmin=0.0, zmax=1.2, zStep=0.03,
                 daymaxStep=4., pixArea=9.6, outputType='zlims', verbose=False, timer=False, ploteffi=False, proxy_level=0,
                 n_bef=5, n_aft=10, snr_min=5., n_phase_min=1, n_phase_max=1, errmodrel=0.1, sigmaC=0.04,
                 x1_color_dist=None, lightOutput=True, T0s='all', zlim_coeff=0.95,
                 ebvofMW=-1., obsstat=True, bands='grizy', fig_for_movie=False, templateLC={}, dbName='', timeIt=False, slower=False, **kwargs):

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
        self.slower = slower

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
        super(SNNSNYMetric, self).__init__(
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
                                      self.m5Col, self.seasonCol, self.nexpCol, self.seeingCol,
                                      self.snr_min, lightOutput=lightOutput)
        # loading parameters
        self.zmin = zmin  # zmin for the study
        self.zmax = zmax  # zmax for the study
        self.zstep = zStep  # zstep
        # get redshift range for processing
        zrange = list(np.arange(self.zmin, self.zmax, self.zstep))
        if zrange[0] < 1.e-6:
            zrange[0] = 0.01

        self.zrange = np.unique(zrange)

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

        time_ref = time.time()

        healpixID = np.unique(dataSlice['healpixID']).tolist()

        if not healpixID:
            zlimsdf = pd.DataFrame()
            return zlimsdf

        print('processing pixel', np.unique(dataSlice['healpixID']))

        self.pixRA = np.mean(dataSlice['pixRA'])
        self.pixDec = np.mean(dataSlice['pixDec'])
        self.healpixID = healpixID[0]

        # get ebvofMW
        ebvofMW = self.ebvofMW
        if ebvofMW < 0:
            ebvofMW = self.ebvofMW_calc()
        if ebvofMW > 0.25:
            return pd.DataFrame()

        """
        # get infos on obs - filter allocation (before stacking)
        obs_alloc = pd.DataFrame(np.copy(dataSlice)).groupby(['season']).apply(
            lambda x: self.filter_allocation(x)).reset_index()
        """
        # print(obs_alloc)

        # select observations filter
        goodFilters = np.in1d(dataSlice[self.filterCol], list(self.bands))
        dataSlice = dataSlice[goodFilters]

        # coaddition per night and per band (if requested by the user)
        if self.stacker is not None:
            # obs = pd.DataFrame(self.stacker._run(obs.to_records(index=False)))
            dataSlice = self.stacker._run(dataSlice)

        dataSlice.sort(order=self.mjdCol)
        if self.verbose:
            print('Observations')
            print(dataSlice[[self.mjdCol, self.exptimeCol,
                  self.filterCol, self.nightCol, self.nexpCol]])

        # get redshift values per season
        zseason = self.z_season(self.season, dataSlice)
        zseason_allz = self.z_season_allz(zseason)

        metricValues = self.metric(
            dataSlice, zseason_allz, x1=-2.0, color=0.2, zlim=-1, metric='zlim')

        zseason = metricValues[['season', 'zcomp']]
        zseason.loc[:, 'zmin'] = 0.01
        zseason.loc[:, 'zstep'] = self.zstep
        zseason = zseason.rename(columns={"zcomp": "zmax"})
        zseason['zmax'] += self.zstep
        zseason_allz = self.z_season_allz(zseason)

        nsn_zcomp = self.metric(
            dataSlice, zseason_allz, x1=0.0, color=0.0, zlim=metricValues[['season', 'zcomp']], metric='nsn')

        metricValues = metricValues.merge(
            nsn_zcomp, left_on=['season'], right_on=['season'])

        """
        # get the season durations
        seasons, dur_z = self.season_length(self.season, dataSlice, zseason)

        if not seasons or dur_z.empty:
            df = self.resError(self.status['season_length'])
            return df

        # get simulation parameters
        gen_par = dur_z.groupby(['z', 'season']).apply(
            lambda x: self.calcDaymax(x, self.daymaxStep)).reset_index()

        if gen_par.empty:
            df = self.resError(self.status['simu_parameters'])
            return df

        # select observations corresponding to seasons
        obs = pd.DataFrame(np.copy(dataSlice))
        obs = obs[obs['season'].isin(seasons)]
        """
        """
        # get infos on obs: cadence, max gap
        cad_gap = self.add_infos(
            obs_alloc, obs, grpCol='season', cadCol='cadence', gapCol='gap_max')

        goodFilters = obs[self.filterCol].isin(['g', 'r', 'i'])
        obs_gri = obs[goodFilters]

        cad_gap = self.add_infos(
            cad_gap, obs_gri, grpCol='season', cadCol='cadence_gri', gapCol='gap_max_gri')
        """
        """
        metricValues = pd.DataFrame()

        # generate LC here
        lc = self.step_lc(obs, gen_par, x1=-2.0, color=0.2)

        print('lc here', lc)

        if self.verbose:
            print(lc['daymax'].unique())

        if len(lc) == 0:
            df = self.resError(self.status['nosn'])
            return df

        # get observing efficiencies and build sn for metric
        lc.index = lc.index.droplevel()
        """
        """
        # get infos on lc (cadence, gap)

        cad_gap_lc_all = lc.groupby(['season', 'daymax', 'z']).apply(
            lambda x: self.cadence_gap(x, 'cadence_sn', 'gap_max_sn'))
        cad_gap_lc = cad_gap_lc_all.groupby(
            ['season']).mean().reset_index()
        # print(cad_gap_lc)
        cad_gap = cad_gap.merge(cad_gap_lc, left_on=[
            'season'], right_on=['season'])
        """
        """
        # estimate efficiencies
        sn_effis = self.step_efficiencies(lc)
        # estimate nsn
        sn = self.step_nsn(sn_effis, dur_z)
        # estimate redshift completeness
        metricValues = sn.groupby(['season']).apply(
            lambda x: self.zlim(x)).reset_index()
        """
        # add ID parameters here
        metricValues['healpixID'] = self.healpixID
        metricValues['pixRA'] = self.pixRA
        metricValues['pixDec'] = self.pixDec
        # merge with all parameters
        metricValues['status'] = self.status['ok']
        metricValues['timeproc'] = time.time()-time_ref
        """
        metricValues = metricValues.merge(
            cad_gap, left_on=['season'], right_on=['season'])
        """
        if self.verbose:
            print('metricValues', metricValues[[
                'season', 'zcomp', 'nsn', 'status', 'timeproc']])
            print('mm', metricValues, metricValues.columns)

        if self.timeIt:
            print('processing time', self.healpixID, time.time()-time_ref)
        return metricValues

    def add_infos(self, infos, obs, grpCol='season', cadCol='cadence', gapCol='gap_max'):
        """
        Estimate cadence and gap max and update a df

        Parameters
        --------------
        infos: pandas df
           df to update
        obs: pandas df
          observations for estimation of cadence and gap max
        grpCol: str, opt
          col to group the df (default: season)
        cadenceCol: str, opt
          name of the (output) cadence col name (default: cadence)
        gapCol: str, opt
           name of the (output) gap col name (default: gap_max)

        """
        cad_gap_res = obs.groupby([grpCol]).apply(lambda x:
                                                  self.cadence_gap(x, cadCol, gapCol)).reset_index()

        # merge cad_gap with obs_alloc
        cad_gap_res = cad_gap_res.merge(
            infos, left_on=[grpCol], right_on=[grpCol])

        return cad_gap_res

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
        mjdmin = grp[self.mjdCol].min()
        mjdmax = grp[self.mjdCol].min()
        dictres['season_length'] = mjdmax-mjdmin

        for b in bands:
            io = grp['filter'] == b
            dictres['frac_{}'.format(b)] = [len(grp[io])/ntot]
            dictres['N{}'.format(b)] = len(grp[io])

        return pd.DataFrame(dictres)

    def z_season(self, seasons, dataSlice):
        """
        Fill the z values per season

        Parameters
        --------------
        seasons: list
          seasons to process
        dataSlice: array
          data to process

        """
        # if seasons = -1: process the seasons seen in data
        if seasons == [-1]:
            seasons = np.unique(dataSlice[self.seasonCol])

        # pandas df with zmin, zmax, zstep per season
        zseason = pd.DataFrame(seasons, columns=['season'])
        zseason['zmin'] = self.zmin
        zseason['zmax'] = self.zmax
        zseason['zstep'] = self.zstep

        return zseason

    def z_season_allz(self, zseason):

        zseason_allz = zseason.groupby(['season']).apply(lambda x: pd.DataFrame(
            {'z': list(np.arange(x['zmin'].mean(), x['zmax'].mean(), x['zstep'].mean()))})).reset_index()

        return zseason_allz[['season', 'z']]

    def season_length(self, seasons, dataSlice, zseason):
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
        dfa = pd.DataFrame(dfa[dfa['season'].isin(seasons)])

        season_info = self.get_season_info(dfa, zseason)

        if season_info.empty:
            return [], pd.DataFrame()

        season_info_cp = pd.DataFrame(season_info)
        # get season length depending on the redshift
        season_info_cp['season'] = season_info_cp['season'].astype(int)
        #season_info_cp = season_info_cp.droplevel('level_1', axis=1)

        """
        dur_z = season_info_cp.groupby('season').apply(
            lambda x: self.duration_z(x)).reset_index()
        """
        dur_z = season_info.groupby('season').apply(
            lambda x: self.nsn_expected_z(x)).reset_index()
        return season_info['season'].to_list(), dur_z

    def step_lc(self, obs, gen_par, x1=-2.0, color=0.2):
        """
        Method to generate lc

        Parameters
        ---------------
        obs: array
          observations
        gen_par: array
          simulation parameters
        x1: float, opt
           stretch value (default: -2.0)
        color: float, opt
          color value (default: 0.2)

        Returns
        ----------
        SN light curves (astropy table)

        """
        lc = obs.groupby(['season']).apply(
            lambda x: self.genLC(x, gen_par, x1, color))

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
                print('effis', sn_effis[idx])
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
        """
        df['cadence'] = 0.

        if len(grp) > 5:
            # to = grp.groupby(['night'])[self.mjdCol].median().sort_values()
            # df['cadence'] = np.mean(to.diff())
            nights = np.sort(grp['night'].unique())
            diff = np.asarray(nights[1:]-nights[:-1])
            df['cadence'] = np.median(diff).item()
        """
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

        # daymin = grp['MJD_min'].values
        # daymax = grp['MJD_max'].values
        dur_z = pd.DataFrame(grp)
        """
        dur_z['T0_min'] = dur_z['MJD_min'] - \
            (1.+dur_z['z'])*self.min_rf_phase_qual
        dur_z['T0_max'] = dur_z['MJD_max'] - \
            (1.+dur_z['z'])*self.max_rf_phase_qual
        dur_z['season_length'] = dur_z['T0_max']-dur_z['T0_min']
        print('dur_z', grp.name, dur_z)
        """
        nsn = self.nsn_from_rate(dur_z)
        print('nsn expected', nsn)
        if self.verbose:
            print('dur_z', dur_z)
            print('nsn expected', nsn)
        dur_z = dur_z.merge(nsn, left_on=['z'], right_on=['z'])

        idx = dur_z['season_length'] > min_duration
        sel = dur_z[idx]
        if len(sel) < 2:
            return pd.DataFrame()

        return sel

    def nsn_expected_z(self, grp):
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

        nsn = self.nsn_from_rate(grp)

        print('nsn expected', nsn)
        if self.verbose:
            print('nsn expected', nsn)

        dur_z = pd.DataFrame(grp)
        dur_z = dur_z.merge(nsn, left_on=['z'], right_on=['z'])

        if 'season' in dur_z.columns:
            dur_z = dur_z.drop(columns=['season'])

        return dur_z

    def calcDaymax(self, grp, daymaxStep):
        """
        Method to estimate T0 (daymax) values for simulation.

        Parameters
        --------------
        grp: group (pandas df sense)
         group of data to process with the following cols:
           T0_min: T0 min value (per season)
           T0_max: T0 max value (per season)
        daymaxStep: float
          step for T0 simulation

        Returns
        ----------
        pandas df with daymax, min_rf_phase, max_rf_phase values

        """

        if self.T0s == 'all':
            T0_max = grp['T0_max'].values
            T0_min = grp['T0_min'].values
            num = (T0_max-T0_min)/daymaxStep
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

    def genLC_deprecated(self, grp, gen_par_orig):
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

    def genLC(self, grp, gen_par_orig, x1, color):
        """
        Method to generate light curves from observations

        Parameters
        ---------------
        grp: pandas group
          observations to process
        gen_par_orig: pandas df
          simulation parameters
        x1: float
          SN stretch
        color: float
          SN color

        Returns
        ----------
        light curves as pandas df

        """
        season = grp.name
        idx = gen_par_orig['season'] == season
        gen_par = gen_par_orig[idx].to_records(index=False)

        sntype = dict(zip([(-2.0, 0.2), (0.0, 0.0)], ['faint', 'medium']))
        res = pd.DataFrame()
        key = (np.round(x1, 1), np.round(color, 1))
        vals = self.lcFast[key]

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
        Big_Diag = np.diag(np.linalg.inv(Fisher_Big))

        res = pd.DataFrame()
        for ia, vala in enumerate(self.params):
            indices = range(ia, len(Big_Diag), npar)
            res['Cov_{}{}'.format(vala, vala)] = np.take(Big_Diag, indices)

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

    def zlim(self, grp, snType='faint'):
        """
        Method to estimate the metric zcomp

        Parameters
        ---------------
        grp: pandas group
        snType: str, opt
          type of SN to estimate zlim (default: faint)

        Returns
        ------------
        pandas df with the metric as cols
        """
        zcomp = -1
        if grp['effi'].mean() > 0.02:
            zcomp = self.zlim_or_nsn(grp, snType, -1)

        if self.ploteffi:
            from sn_metrics.sn_plot_live import plot_zlim, plot_nsn
            plot_zlim(grp, snType, self.zmin,
                      self.zmax, self.zlim_coeff)

        return pd.DataFrame({'zcomp': [zcomp]})

    def nsn(self, grp, snType='medium'):
        """
        Method to estimate the metric nsn up to zlim

        Parameters
        ---------------
        grp: pandas group
        snType: str, opt
          type of SN to estimate zlim (default: medium)

        Returns
        ------------
        pandas df with the metric as cols
        """
        nsn = -1

        if grp['effi'].mean() > 0.02:
            nsn = self.zlim_or_nsn(grp, snType, grp['zcomp'].mean())

        if self.ploteffi:
            from sn_metrics.sn_plot_live import plot_zlim, plot_nsn
            plot_nsn(grp, snType, self.zmin, self.zmax, zlim)

        return pd.DataFrame({'nsn': [nsn]})

    def metric(self, dataSlice, zseason, x1=-2.0, color=0.2,  zlim=-1, metric='zlim'):

        snType = 'medium'
        if np.abs(x1+2.0) <= 1.e-5:
            snType = 'faint'

        # get the season durations
        seasons, dur_z = self.season_length(self.season, dataSlice, zseason)

        if not seasons or dur_z.empty:
            df = self.resError(self.status['season_length'])
            return df

        # get simulation parameters
        gen_par = dur_z.groupby(['z', 'season']).apply(
            lambda x: self.calcDaymax(x, self.daymaxStep)).reset_index()

        if gen_par.empty:
            df = self.resError(self.status['simu_parameters'])
            return df

        # select observations corresponding to seasons
        obs = pd.DataFrame(np.copy(dataSlice))
        obs = obs[obs['season'].isin(seasons)]

        # metric values in a DataFrame
        metricValues = pd.DataFrame()

        # generate LC here
        lc = self.step_lc(obs, gen_par, x1=x1, color=color)

        if self.verbose:
            print(lc['daymax'].unique())

        if len(lc) == 0:
            df = self.resError(self.status['nosn'])
            return df

        # get observing efficiencies and build sn for metric
        lc.index = lc.index.droplevel()

        # get infos on lc (cadence, gap)

        if snType == 'faint' and self.slower:
            obs_alloc = pd.DataFrame(np.copy(dataSlice)).groupby(['season']).apply(
                lambda x: self.filter_allocation(x)).reset_index()
            # get infos on obs: cadence, max gap
            cad_gap = self.add_infos(
                obs_alloc, obs, grpCol='season', cadCol='cadence', gapCol='gap_max')

            goodFilters = obs[self.filterCol].isin(['g', 'r', 'i'])
            obs_gri = obs[goodFilters]

            cad_gap = self.add_infos(
                cad_gap, obs_gri, grpCol='season', cadCol='cadence_gri', gapCol='gap_max_gri')
            cad_gap_lc_all = lc.groupby(['season', 'daymax', 'z']).apply(
                lambda x: self.cadence_gap(x, 'cadence_sn', 'gap_max_sn'))
            cad_gap_lc = cad_gap_lc_all.groupby(
                ['season']).mean().reset_index()
            # print(cad_gap_lc)
            cad_gap = cad_gap.merge(cad_gap_lc, left_on=[
                'season'], right_on=['season'])

        # estimate efficiencies
        sn_effis = self.step_efficiencies(lc)
        # estimate nsn
        sn = self.step_nsn(sn_effis, dur_z)
        # estimate redshift completeness
        if metric == 'zlim':
            metricValues = sn.groupby(['season']).apply(
                lambda x: self.zlim(x)).reset_index()

        if metric == 'nsn':
            sn = sn.merge(zlim, left_on=['season'], right_on=['season'])
            metricValues = sn.groupby(['season']).apply(
                lambda x: self.nsn(x)).reset_index()

        if snType == 'faint' and self.slower:
            metricValues = metricValues.merge(
                cad_gap, left_on=['season'], right_on=['season'])

        return metricValues

    def metric_deprecated(self, grp):
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
        zmin = np.round(grp['z'].min(), 2)
        zmax = np.round(grp['z'].max(), 2)
        grp = grp.sort_values(by=['z'])
        zstep = np.round(np.median(np.diff(grp['z'])), 2)

        #zmin, zmax, zstep = self.zmin, self.zmax, self.zstep
        zz, rate, err_rate, nsn, err_nsn = self.rateSN(zmin=zmin,
                                                       zmax=zmax+zstep,
                                                       dz=zstep,
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

        cols = ['season', 'zcomp', 'nsn', 'cadence', 'timeproc']
        if self.slower:
            cols += ['gap_max', 'frac_u', 'frac_g',
                     'frac_r', 'frac_i', 'frac_z', 'frac_y', 'Ng', 'Nr', 'Ni', 'Nz', 'Ny', 'cadence_sn', 'gap_max_sn', 'season_length', 'cadence_gri', 'gap_max_gri']

        for col in cols:
            df[col] = -1

        df['status'] = istatus

        return df

    def get_season_info(self, dfa, zseason, min_duration=60.):
        """
        method to get season infos vs z

        Parameters
        --------------
        dfa: pandas df
          dat to process
        zseason: pandas df
          redshift infos per season
        min_duration: float, opt
          min season length to be accepted (default: 60 days)

        Returns
        ----------
        pandas df with season length infos

        """

        season_info = dfa.groupby('season').apply(
            lambda x: self.seasonInfo(x, min_duration=min_duration)).reset_index()

        #season_info.index = season_info.index.droplevel()
        season_info = season_info.drop(columns=['level_1'])

        season_info = season_info.merge(
            zseason, left_on=['season'], right_on=['season'])

        season_info['T0_min'] = season_info['MJD_min'] - \
            (1.+season_info['z'])*self.min_rf_phase_qual
        season_info['T0_max'] = season_info['MJD_max'] - \
            (1.+season_info['z'])*self.max_rf_phase_qual
        season_info['season_length'] = season_info['T0_max'] - \
            season_info['T0_min']

        idx = season_info['season_length'] >= min_duration

        return pd.DataFrame(season_info[idx])
