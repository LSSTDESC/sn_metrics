import numpy as np
from lsst.sims.maf.metrics import BaseMetric
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
    """

    def __init__(self, lc_reference, dustcorr,
                 metricName='SNNSNMetric',
                 mjdCol='observationStartMJD', RACol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', season=[-1], coadd=True, zmin=0.0, zmax=1.2,
                 pixArea=9.6, outputType='zlims', verbose=False, timer=False, ploteffi=False, proxy_level=0,
                 n_bef=5, n_aft=10, snr_min=5., n_phase_min=1, n_phase_max=1, errmodrel=0.1,
                 x1_color_dist=None, lightOutput=True, T0s='all', zlim_coeff=0.95, ebvofMW=-1., obsstat=True, **kwargs):

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

        cols = [self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol]

        self.stacker = None
        if coadd:
            cols += ['coadd']
            self.stacker = CoaddStacker(mjdCol=self.mjdCol, RACol=self.RACol, DecCol=self.DecCol, m5Col=self.m5Col, nightCol=self.nightCol,
                                        filterCol=self.filterCol, numExposuresCol=self.nexpCol, visitTimeCol=self.vistimeCol, visitExposureTimeCol='visitExposureTime')
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
                                      self.m5Col, self.seasonCol, self.nexpCol,
                                      self.snr_min, lightOutput=lightOutput)

        # loading parameters
        self.zmin = zmin  # zmin for the study
        self.zmax = zmax  # zmax for the study
        self.zStep = 0.05  # zstep
        self.daymaxStep = 2.  # daymax step
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
        # time 0 for performance estimation purpose
        time_ref = time.time()

        # Get ebvofMW here
        ebvofMW = self.ebvofMW
        if ebvofMW < 0.:
            RA = np.mean(dataSlice[self.RACol])
            Dec = np.mean(dataSlice[self.DecCol])
            pixRA = np.unique(dataSlice['pixRA'])[0]
            pixDec = np.unique(dataSlice['pixDec'])[0]
            # in that case ebvofMW value is taken from a map
            coords = SkyCoord(pixRA, pixDec, unit='deg')
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

        # get the seasons
        seasons = self.season

        # if seasons = -1: process the seasons seen in data
        if self.season == [-1]:
            seasons = np.unique(dataSlice[self.seasonCol])

        # get redshift range for processing
        zRange = list(np.arange(self.zmin, self.zmax, self.zStep))
        if zRange[0] < 1.e-6:
            zRange[0] = 0.01

        self.zRange = np.unique(zRange)

        # season infos
        dfa = pd.DataFrame(np.copy(dataSlice))
        dfa = dfa[dfa['season'].isin(seasons)]
        season_info = dfa.groupby(['season']).apply(
            lambda x: self.seasonInfo(x)).reset_index()

        # select seasons of at least 30 days
        idx = season_info['season_length'] >= 60
        season_info = season_info[idx]

        if season_info.empty:
            return None

        if self.verbose:
            print('season infos', season_info[['season', 'season_length']])

        # get season length depending on the redshift
        dur_z = season_info.groupby(['season']).apply(
            lambda x: self.duration_z(x)).reset_index()

        # remove dur_z with negative season lengths
        idx = dur_z['season_length'] >= 10.
        dur_z = dur_z[idx]

        if self.verbose:
            print('duration vs z', dur_z)

        if dur_z.empty:
            return None
        # get simulation parameters
        if self.verbose:
            print('getting simulation parameters')
        gen_par = dur_z.groupby(['z', 'season']).apply(
            lambda x: self.calcDaymax(x)).reset_index()

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
            toshow = ['pixRA', 'pixDec', 'healpixID', 'season', 'x1_faint', 'color_faint', 'zlim_faint',
                      'zlimp_faint', 'zlimm_faint',
                      'nsn_med_faint', 'err_nsn_med_faint']
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

        return varb_totdf

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

        # get pixel id
        pixRA = np.unique(dataSlice['pixRA'])[0]
        pixDec = np.unique(dataSlice['pixDec'])[0]
        healpixID = int(np.unique(dataSlice['healpixID'])[0])

        if self.verbose:
            print('#### Processing season', seasons, healpixID)

        groupnames = ['pixRA', 'pixDec', 'healpixID', 'season', 'x1', 'color']

        gen_p = gen_par[gen_par['season'].isin(seasons)]
        if gen_p.empty:
            if self.verbose:
                print('No generator parameter found')
            return None, None
        dur_z = dura_z[dura_z['season'].isin(seasons)]
        obs = pd.DataFrame(np.copy(dataSlice))
        obs = obs[obs['season'].isin(seasons)]

        if self.timer:
            time_refb = time.time()

        # coaddition per night and per band (if requested by the user)
        if self.stacker is not None:
            obs = self.stacker._run(obs.to_records(index=False))
            if self.verbose:
                print('after stacking', seasons)
            if self.timer:
                print('timing:', time.time()-time_refb)
        else:
            obs = obs.to_records(index=False)

        # print('after stacker', seasons, len(obs))
        obs.sort(order='night')
        # print('data', obs[['night', 'filter',
        #                  'observationStartMJD', 'fieldRA', 'fieldDec']])
        # estimate m5 median and gaps
        m5_med = np.median(obs[self.m5Col])
        obs.sort(order=self.mjdCol)
        diffs = np.diff(obs[self.mjdCol])
        gap_max = np.max(diffs)
        gap_med = np.median(diffs)

        # simulate supernovae and lc
        if self.verbose:
            print("LC generation")

        sn = pd.DataFrame()
        if ebvofMW < 0.25:
            sn, lc = self.gen_LC_SN(obs, ebvofMW, gen_p.to_records(
                index=False), verbose=self.verbose, timer=self.timer)

            # print('sn here', sn[['x1', 'color', 'z', 'daymax', 'Cov_colorcolor']])
            if self.verbose:
                idx = np.abs(sn['x1']+2) < 1.e-5
                idx &= np.abs(sn['z']-0.2) < 1.e-5
                sel = sn[idx]
                sel = sel.sort_values(by=['z', 'daymax'])

                print('sn and lc', len(sn),
                      sel[['x1', 'color', 'z', 'daymax', 'Cov_colorcolor', 'n_bef', 'n_aft']])

            if self.outputType == 'lc' or self.outputType == 'sn':
                return sn, lc

        if sn.empty:
            # no LC could be simulated -> fill output with errors
            if self.verbose:
                print('no simulation possible!!')
            for seas in seasons:
                zlimsdf = self.errordf(
                    pixRA, pixDec, healpixID, seas,
                    self.status['nosn'],
                    m5_med, gap_max, gap_med, ebvofMW, cadence, season_length, Nvisits)
                effi_seasondf = self.erroreffi(
                    pixRA, pixDec, healpixID, seas)
            return effi_seasondf, zlimsdf
        else:
            # LC could be simulated -> estimate efficiencies
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

                # estimate number of medium supernovae
                zlimsdf['nsn_med'],  zlimsdf['err_nsn_med'] = zlimsdf.apply(lambda x: self.nsn_typedf(
                    x, 0.0, 0.0, effi_seasondf, dur_z), axis=1, result_type='expand').T.values

                dfa = pd.DataFrame([zlimsdf.iloc[0]], columns=zlimsdf.columns)
                dfb = pd.DataFrame([zlimsdf.iloc[1]], columns=zlimsdf.columns)

                on = ['healpixID', 'pixRA', 'pixDec', 'season']
                zlimsdf = dfa.merge(
                    dfb, left_on=on, right_on=on, suffixes=('_faint', '_medium'))

                if self.verbose:
                    print('result here', zlimsdf[[
                          'zlim_faint', 'zlim_medium', 'nsn_med_faint']])

                # add observing stat if requested
                if self.obsstat:
                    # add median m5
                    zlimsdf.loc[:, 'm5_med'] = m5_med
                    zlimsdf.loc[:, 'gap_max'] = gap_max
                    zlimsdf.loc[:, 'gap_med'] = gap_med
                    zlimsdf.loc[:, 'ebvofMW'] = ebvofMW
                    zlimsdf.loc[:, 'cadence'] = cadence
                    zlimsdf.loc[:, 'season_length'] = season_length
                    for b, vals in Nvisits.items():
                        zlimsdf.loc[:, 'N_{}'.format(b)] = vals

            else:

                for seas in seasons:
                    zlimsdf = self.errordf(
                        pixRA, pixDec, healpixID, seas,
                        self.status['low_effi'],
                        m5_med, gap_max, gap_med, ebvofMW, cadence, season_length, Nvisits)
                    effi_seasondf = self.erroreffi(
                        pixRA, pixDec, healpixID, seas)

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
        m5_med: float
          median m5 value
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
                           'm5_med': [m5_med],
                           'gap_max': [gap_max],
                           'gap_med': [gap_med],
                           'ebvofMW': [ebvofMW],
                           'cadence': [cadence],
                           'season_length': [season_length]})

        for key, val in Nvisits.items():
            df['N_{}'.format(key)] = val

        for vv in ['x1', 'color', 'zlim', 'zlimp', 'zlimm', 'nsn_med', 'err_nsn_med']:
            for ko in ['faint', 'medium']:
                df['{}_{}'.format(vv, ko)] = [-1.0]

        for ko in ['faint', 'medium']:
            df['status_{}'.format(ko)] = [int(errortype)]

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

    @verbose_this('Estimate efficiencies')
    @time_this('Efficiencies')
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

        # this is to plot efficiencies and also sigma_color vs z
        if self.ploteffi:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots()
            # figb, axb = plt.subplots()

            self.plot(ax, effi, 'effi', 'effi_err',
                      'Observing Efficiencies', ls='-')
            # sndf['sigma_color'] = np.sqrt(sndf['Cov_colorcolor'])
            # self.plot(axb, sndf, 'sigma_color', None, '$\sigma_{color}$')

            plt.show()

        return effi

    def plot(self, ax, effi, vary, erry=None, legy='', ls='None'):
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
         status: status of the processing

        """

        zlimit = 0.0
        status = self.status['effi']

        # z range for the study
        zplot = np.arange(self.zmin, self.zmax, 0.01)

        # print(grp['z'], grp['effi'])

        if len(grp['z']) <= 3:
            return pd.DataFrame({'zlim': [zlimit],
                                 'zlimp': [zlimit],
                                 'zlimm': [zlimit],
                                 'status': [int(status)]})
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
            zlimit, zlimit_plus, zlimit_minus, status = self.zlim_from_cumul(
                grp, duration_z, effiInterp, effiInterp_err, zplot, rate='SN_rate')

        return pd.DataFrame({'zlim': [zlimit],
                             'zlimp': [zlimit_plus],
                             'zlimm': [zlimit_minus],
                             'status': [int(status)]})

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

        nsn_cum = np.cumsum(effiInterp(zplot)*rateInterp(zplot))
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
            zlim_plus = interp1d(nsn_cum_norm+nsn_cum_norm_err,
                                 zplot, bounds_error=False, fill_value=-1.)
            zlim_minus = interp1d(
                nsn_cum_norm-nsn_cum_norm_err, zplot, bounds_error=False, fill_value=-1.)
            zlimit = zlim(self.zlim_coeff).item()
            zlimit_minus = zlim_plus(self.zlim_coeff).item()
            zlimit_plus = zlim_minus(self.zlim_coeff).item()

            status = self.status['ok']

            if self.ploteffi:
                self.plot_NSN_cumul(grp, nsn_cum_norm, nsn_cum_norm_err, zplot)
        else:
            zlimit = 0.
            status = self.status['low_effi']

        return zlimit, zlimit_plus, zlimit_minus, status

    def plot_NSN_cumul(self, grp, nsn_cum_norm, nsn_cum_norm_err, zplot):
        """
        Method to plot the NSN cumulative vs redshift

        Parameters
        --------------
        grp: pandas group
         data to process

        """
        import matplotlib.pylab as plt
        fig, ax = plt.subplots()
        x1 = grp['x1'].unique()[0]
        color = grp['color'].unique()[0]

        ax.plot(zplot, nsn_cum_norm,
                label='(x1,color)=({},{})'.format(x1, color), color='r')
        ax.fill_between(zplot, nsn_cum_norm-nsn_cum_norm_err,
                        nsn_cum_norm+nsn_cum_norm_err, color='y')
        ftsize = 15
        ax.set_ylabel('NSN ($z<$)', fontsize=ftsize)
        ax.set_xlabel('z', fontsize=ftsize)
        ax.xaxis.set_tick_params(labelsize=ftsize)
        ax.yaxis.set_tick_params(labelsize=ftsize)
        ax.set_xlim((0.0, 0.8))
        ax.set_ylim((0.0, 1.05))
        ax.plot([0., 1.2], [self.zlim_coeff, self.zlim_coeff],
                ls='--', color='k')
        plt.legend(fontsize=ftsize)
        plt.show()

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

        nsn, var_nsn = self.nsn(
            effisel, grp['zlim'], grp['zlimp'], grp['zlimm'], durinterp_z)
        """
        nsn, var_nsn = self.nsn(
            effisel, grp['zlim'], grp['zlimp'], grp['zlimm'], seas_duration_z)
        """
        return (nsn, var_nsn)

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

    @verbose_this('Estimating the total number of supernovae')
    @time_this('Total number of supernovae')
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

    @verbose_this('Selecting')
    @time_this('Selection')
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
        if self.errmodrel > 0.:
            tab = self.select_error_model(tab)

        # define a 'night' column
        tab['nnight'] = np.sign(tab['phase'])*tab['time']
        tab['nnight'] = tab['nnight'].astype(int)

        # now groupby
        tab = tab.round({'pixRA': 4, 'pixDec': 4, 'daymax': 3,
                         'z': 3, 'x1': 2, 'color': 2})
        groups = tab.groupby(
            ['pixRA', 'pixDec', 'daymax', 'season', 'z', 'healpixID', 'x1', 'color'])

        tosum = []
        for ia, vala in enumerate(self.params):
            for jb, valb in enumerate(self.params):
                if jb >= ia:
                    tosum.append('F_'+vala+valb)
        #tosum += ['n_aft', 'n_bef', 'n_phmin', 'n_phmax']
        tosum += ['n_phmin', 'n_phmax']
        # apply the sum on the group
        #sums = groups[tosum].sum().reset_index()
        sums = groups.apply(lambda x: self.sumIt(x, tosum)).reset_index()

        # select LC according to the number of points bef/aft peak
        idx = sums['n_aft'] >= self.n_aft
        idx &= sums['n_bef'] >= self.n_bef
        idx &= sums['n_phmin'] >= self.n_phase_min
        idx &= sums['n_phmax'] >= self.n_phase_max

        finalsn = pd.DataFrame()
        goodsn = pd.DataFrame(sums.loc[idx])

        # estimate the color for SN that passed the selection cuts
        if len(goodsn) > 0:
            goodsn.loc[:, 'Cov_colorcolor'] = CovColor(goodsn).Cov_colorcolor
            finalsn = pd.concat([finalsn, goodsn], sort=False)

        badsn = pd.DataFrame(sums.loc[~idx])

        # Supernovae that did not pass the cut have a sigma_color=10
        if len(badsn) > 0:
            badsn.loc[:, 'Cov_colorcolor'] = 100.
            finalsn = pd.concat([finalsn, badsn], sort=False)

        return finalsn

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

        epochs = grp['nnight'].unique()

        n_bef = len(epochs[epochs <= 0.])
        n_aft = len(epochs)-n_bef

        sums['n_bef'] = n_bef
        sums['n_aft'] = n_aft

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

    @verbose_this('Simulation SN')
    @time_this('Simulation SN')
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
        for key, vals in self.lcFast.items():
            time_refs = time.time()
            gen_par_cp = np.copy(gen_par)
            if key == (-2.0, 0.2):
                idx = gen_par_cp['z'] < 0.9
                gen_par_cp = gen_par_cp[idx]
            lc = vals(obs, ebvofMW, gen_par_cp, bands='grizy')
            if self.verbose:
                print('End of simulation', key, time.time()-time_refs)
            if self.ploteffi and len(lc) > 0:
                self.plotLC(lc)
            if self.outputType == 'lc':
                lc_tot = pd.concat([lc_tot, lc], sort=False)
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

        return sn_tot, lc_tot

    def plotLC(self, lc, zref=0.01):
        """
        Method to plot LC

        Parameters
        --------------
        lc: pandas df of lc points
        zref: redshift value chosen for the display

        """

        import matplotlib.pyplot as plt

        lc = lc.round({'daymax': 6})
        sel = lc[np.abs(lc['z']-zref) < 1.e-5]
        # sel = sel[sel['band']=='LSST::g']

        # print(sel['daymax'].unique())
        fig, ax = plt.subplots(ncols=2, nrows=3)
        pos = dict(
            zip('ugrizy', [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]))

        fig.suptitle('(x1,color)=({},{})'.format(
            sel['x1'].unique(), sel['color'].unique()))
        for band in sel['band'].unique():
            idxb = sel['band'] == band
            selb = sel.loc[idxb]
            ix = pos[band.split(':')[-1]][0]
            iy = pos[band.split(':')[-1]][1]
            for daymax in selb['daymax'].unique():
                selc = selb[selb['daymax'] == daymax]
                ax[ix][iy].plot(selc['phase'], selc['flux_e_sec'])

        plt.show()

    @verbose_this('Estimating redshift limits')
    @time_this('redshift limits')
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
