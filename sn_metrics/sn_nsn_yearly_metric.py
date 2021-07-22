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
    """

    def __init__(self, lc_reference, dustcorr,
                 metricName='SNNSNYMetric',
                 mjdCol='observationStartMJD', RACol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', season=[-1], coadd=True, zmin=0.0, zmax=1.2,
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
                                      self.m5Col, self.seasonCol, self.nexpCol,
                                      self.snr_min, lightOutput=lightOutput)

        # loading parameters
        self.zmin = zmin  # zmin for the study
        self.zmax = zmax  # zmax for the study
        self.zStep = 0.03  # zstep
        # get redshift range for processing
        zRange = list(np.arange(self.zmin, self.zmax, self.zStep))
        if zRange[0] < 1.e-6:
            zRange[0] = 0.01

        self.zRange = np.unique(zRange)

        self.daymaxStep = 2.  # daymax step
        self.min_rf_phase = -20.  # min ref phase for LC points selection
        self.max_rf_phase = 60.  # max ref phase for LC points selection

        self.min_rf_phase_qual = -15.  # min ref phase for bounds effects
        self.max_rf_phase_qual = 30.  # max ref phase for bounds effects

        # snrate
        rateSN = SN_Rate(H0=70., Om0=0.3,
                         min_rf_phase=self.min_rf_phase_qual, max_rf_phase=self.max_rf_phase_qual)
        self.duration_ref = 180.
        zz, rate, err_rate, nsn, err_nsn = rateSN(zmin=0.01,
                                                  zmax=self.zmax,
                                                  # duration_z=durinterp_z,
                                                  duration=self.duration_ref,
                                                  survey_area=self.pixArea,
                                                  account_for_edges=True)

        self.nsn_expected = interp1d(zz, nsn, kind='linear',
                                     bounds_error=False, fill_value=0)

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

        # time 0 for performance estimation purpose
        time_ref = time.time()
        goodFilters = np.in1d(dataSlice[self.filterCol], list(self.bands))
        dataSlice = dataSlice[goodFilters]

        print('processing pixel', np.unique(dataSlice['healpixID']))

        self.pixRA = np.unique(dataSlice['pixRA'])[0]
        self.pixDec = np.unique(dataSlice['pixDec'])[0]
        self.healpixID = np.unique(dataSlice['healpixID'])[0]

        # Get ebvofMW here
        ebvofMW = self.ebvofMW
        if ebvofMW < 0:
            ebvofMW = self.ebvofMW_calc()

        print('hello', self.pixRA, self.pixDec, self.healpixID, ebvofMW)

        # get the seasons
        seasons = self.season

        # if seasons = -1: process the seasons seen in data
        if self.season == [-1]:
            seasons = np.unique(dataSlice[self.seasonCol])

        # season infos
        dfa = pd.DataFrame(np.copy(dataSlice))
        dfa = dfa[dfa['season'].isin(seasons)]
        season_info = dfa.groupby(['season']).apply(
            lambda x: self.seasonInfo(x, min_duration=60)).reset_index()

        print(season_info)

        # get season length depending on the redshift
        dur_z = season_info.groupby(['season']).apply(
            lambda x: self.duration_z(x)).reset_index()

        print(dur_z)

        # get simulation parameters
        gen_par = dur_z.groupby(['z', 'season']).apply(
            lambda x: self.calcDaymax(x)).reset_index()

        print(gen_par)
        obs = pd.DataFrame(np.copy(dataSlice))
        obs = obs[obs['season'].isin(seasons)]
        # coaddition per night and per band (if requested by the user)
        if self.stacker is not None:
            obs = pd.DataFrame(self.stacker._run(obs.to_records(index=False)))

        # generate LC here
        if ebvofMW < 0.25:
            # generate LC here
            lc = obs.groupby(['season']).apply(
                lambda x: self.genLC(x, gen_par))

            # plot figs for movie
            if self.ploteffi and self.fig_for_movie and len(lc) > 0:
                self.plot_for_movie(obs, lc, gen_par)

            # get observing efficiencies
            if len(lc) >= 0:
                # lc = lc.rename_axis(None)
                lc.index = lc.index.droplevel()
                print(lc.columns)
                sn_effis = lc.groupby(['healpixID', 'season', 'z', 'x1', 'color', 'sntype']).apply(
                    lambda x: self.sn_effi(x)).reset_index()

            # estimate efficiencies
            for vv in ['healpixID', 'season']:
                sn_effis[vv] = sn_effis[vv].astype(int)
            sn_effis['effi'] = sn_effis['nsel']/sn_effis['ntot']
            sn_effis['effi_err'] = np.sqrt(
                sn_effis['nsel']*(1.-sn_effis['effi']))/sn_effis['ntot']

            # add season length here
            sn_effis = sn_effis.merge(
                dur_z, left_on=['season', 'z'], right_on=['season', 'z'])

            if self.ploteffi:
                from sn_metrics.sn_plot_live import plotNSN_effi
                for season in sn_effis['season'].unique():
                    idx = sn_effis['season'] == season
                    plotNSN_effi(sn_effis[idx], 'effi', 'effi_err',
                                 'Observing Efficiencies', ls='-')

            # estimate the number of supernovae
            sn_effis['nsn'] = sn_effis['effi']*self.nsn_expected(
                sn_effis['z'].to_list())*sn_effis['season_length']/self.duration_ref

            # get redshift limit and nsn
            metric = sn_effis.groupby(['season']).apply(
                lambda x: self.metric(x)).reset_index()
            print('ici', metric)

            if self.ploteffi:
                from sn_metrics.sn_plot_live import plot_zlim
                for season in sn_effis['season'].unique():
                    idx = sn_effis['season'] == season
                    plot_zlim(sn_effis[idx], 'faint', self.zmin,
                              self.zmax, self.zlim_coeff)

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
        dur_z = pd.DataFrame(self.zRange, columns=['z'])
        dur_z['T0_min'] = daymin-(1.+dur_z['z'])*self.min_rf_phase_qual
        dur_z['T0_max'] = daymax-(1.+dur_z['z'])*self.max_rf_phase_qual
        dur_z['season_length'] = dur_z['T0_max']-dur_z['T0_min']
        # dur_z['season_length'] = [daymax-daymin]*len(self.zRange)

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

        phmin = np.tile(lcarr[varname], (nvals, 1))
        n_phmin = np.ma.array(phmin, mask=~flag)
        n_phmin = n_phmin.sum(axis=1)

        return n_phmin

    def get_epochs(self, nights, flag, flagph):

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

        params = ['x0', 'x1', 'daymax', 'color']
        parts = {}
        for ia, vala in enumerate(params):
            for jb, valb in enumerate(params):
                if jb >= ia:
                    parts[ia, jb] = grp['F_'+vala+valb]

        # print(parts)
        size = len(grp)
        npar = len(params)
        Fisher_Big = np.zeros((npar*size, npar*size))
        Big_Diag = np.zeros((npar*size, npar*size))
        Big_Diag = []

        for iv in range(size):
            Fisher_Matrix = np.zeros((npar, npar))
            for ia, vala in enumerate(params):
                for jb, valb in enumerate(params):
                    if jb >= ia:
                        Fisher_Big[ia+npar*iv][jb+npar*iv] = parts[ia, jb][iv]

        # pprint.pprint(Fisher_Big)

        Fisher_Big = Fisher_Big + np.triu(Fisher_Big, 1).T
        Big_Diag = np.diag(np.linalg.inv(Fisher_Big))

        res = pd.DataFrame()
        for ia, vala in enumerate(self.params):
            indices = range(ia, len(Big_Diag), npar)
            # restab.add_column(
            #    Column(np.take(Big_Diag, indices), name='Cov_{}{}'.format(vala,vala)))
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

        return res

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

        return pd.DataFrame({'zcomp': [zcomp], 'nsn': [nsn]})
