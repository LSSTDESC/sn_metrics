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
from sn_metrics.sn_plot_live import Plot_Saturation_Metric
from astropy.table import Table, vstack, Column
import time
import pandas as pd
from scipy.interpolate import interp1d
from sn_tools.sn_rate import SN_Rate
from scipy.interpolate import RegularGridInterpolator
from functools import wraps
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
import sncosmo
from astropy import (cosmology, units as u, constants as const)
from sn_fitter.fit_sn_cosmo import Fit_LC
from sn_fit.sn_utils import Selection

filtercolors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))

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


class SNSaturationMetric(BaseMetric):
    """
    Estimate saturation metric for low-z supernovae

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
    verbose: bool,opt
      verbose mode (default: False)
    plotMetric: bool, opt
      display metric output during processing (default:False)
    proxy_level: int, opt
     proxy level for the processing (default: 0)
     snr_min: float, opt
       minimal SNR of LC points (default: 5.0)
    n_bef: int, opt
      number of LC points LC before T0 (default:5)
    n_aft: int, opt
      number of LC points after T0 (default: 10)
     n_phase_min: int, opt
       number of LC points with phase<= -5(default:1)
    n_phase_max: int, opt
      number of LC points with phase>= 20 (default: 1)
    lightOutput: bool, opt
      output level of information (light or more) (default:True)
    fracpixel: numpyarray, opt
      array of max frac pixel signal vs seeing (default: None)
    fullwell: float, opt
       ccd full well limit for saturation
    saturationLevel: float, opt
      fraction of saturated LC point in a night (epoch) to consider this night as saturated. (default: 0.99)

    """

    def __init__(self, lc_reference, dustcorr,
                 metricName='SNSaturationMetric',
                 mjdCol='observationStartMJD', RACol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', seeingCol='seeingFwhmEff', season=[-1], coadd=True, zmin=0.0, zmax=0.05,
                 verbose=False, timer=False, plotmetric=False, snr_min=5., nbef=4, naft=10, n_phase_min=1, n_phase_max=1, lightOutput=False, ebvofMW=-1., fracpixel=None, fullwell=90000., saturationLevel=0.99, figs_for_movie=False, model='salt2-extended', version=1.0, telescope=None, display=False, bands='gr', obsstat=False, **kwargs):

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
        self.plotmetric = plotmetric
        self.ebvofMW = ebvofMW
        self.fullwell = fullwell
        self.saturationLevel = saturationLevel
        self.figs_for_movie = figs_for_movie
        self.bands = bands
        # self.figs_for_movie = True
        cols = [self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol, self.seeingCol]

        self.stacker = None
        if coadd:
            cols += ['coadd']
            self.stacker = CoaddStacker(mjdCol=self.mjdCol, RACol=self.RACol, DecCol=self.DecCol, m5Col=self.m5Col, nightCol=self.nightCol,
                                        filterCol=self.filterCol, numExposuresCol=self.nexpCol, visitTimeCol=self.vistimeCol, seeingaCol=self.seeingCol, visitExposureTimeCol='visitExposureTime')
        super(SNSaturationMetric, self).__init__(
            col=cols, metricDtype='object', metricName=metricName, **kwargs)

        self.season = season

        telescope = Telescope(airmass=1.2)

        # LC selection parameters
        self.snr_min = snr_min  # SNR cut for points before/after peak
        self.n_bef = nbef
        self.n_aft = naft
        self.n_phase_min = n_phase_min
        self.n_phase_max = n_phase_max

        # print('selection', self.n_bef, self.n_aft,
        #      self.n_phase_min, self.n_phase_max)
        self.lcFast = {}

        # loading reference LC files
        for key, vals in lc_reference.items():
            self.lcFast[key] = LCfast(vals, dustcorr[key], key[0], key[1], telescope,
                                      self.mjdCol, self.RACol, self.DecCol,
                                      self.filterCol, self.exptimeCol,
                                      self.m5Col, self.seasonCol, self.nexpCol,
                                      self.seeingCol,
                                      self.snr_min, lightOutput=lightOutput)

        # loading parameters
        self.zmin = zmin  # zmin for the study
        self.zmax = zmax  # zmax for the study
        self.zStep = 0.005  # zstep
        self.daymaxStep = 2.  # daymax step
        self.min_rf_phase = -20.  # min ref phase for LC points selection
        self.max_rf_phase = 60.  # max ref phase for LC points selection

        self.min_rf_phase_qual = -15.  # min ref phase for bounds effects
        self.max_rf_phase_qual = 30.  # max ref phase for bounds effects

        # verbose mode - useful for debug and code performance estimation
        self.verbose = verbose
        self.timer = timer

        # supernovae parameters
        self.params = ['x0', 'x1', 'daymax', 'color']

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

        if fracpixel is not None:
            self.pixel_max = interp1d(
                fracpixel['seeing'], fracpixel['pixel_frac_med'], fill_value=0.0, bounds_error=False)

        self.selection = Selection(snr_min, nbef, naft, -1, phase_min=-5,
                                   phase_max=20, nphase_min=n_phase_min, nphase_max=n_phase_max)
        self.fit_lc = Fit_LC(model='salt2-extended', version=1.0, telescope=telescope, display=False, bands=bands, snrmin=snr_min, nbef=0,
                             naft=0, nbands=-1, phasemin=-5, phasemax=20, nphasemin=0, nphasemax=0, errmodrel=-1., include_errmodel_in_lcerror=False)

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
        pandas df of metric values
        """

        """
        import matplotlib.pyplot as plt
        plt.plot(dataSlice[self.RACol], dataSlice[self.DecCol], 'ko')
        print('data', len(dataSlice))
        plt.show()
        """

        # time 0 for performance estimation purpose
        time_ref = time.time()

        healpixID = np.unique(dataSlice['healpixID'])

        if not healpixID:
            return pd.DataFrame()

        pixRA = np.unique(dataSlice['pixRA'])[0]
        pixDec = np.unique(dataSlice['pixDec'])[0]
        healpixID = int(np.unique(dataSlice['healpixID'])[0])

        self.healpixID = healpixID
        self.pixRA = pixRA
        self.pixDec = pixDec

        print('processing', healpixID)
        # Get ebvofMW here
        ebvofMW = self.ebvofMW
        if ebvofMW < 0.:
            # in that case ebvofMW value is taken from a map
            ebvofMW = self.get_ebv()

        if ebvofMW > 0.25:
            return pd.DataFrame()

        if self.figs_for_movie:
            self.plot_live = Plot_Saturation_Metric(
                self.healpixID, 0.02, self.snr_min,
                self.mjdCol, self.m5Col, self.filterCol, self.fullwell, self.saturationLevel)

        # get the seasons
        seasons = self.season

        # if seasons = -1: process the seasons seen in data
        if self.season == [-1]:
            seasons = np.unique(dataSlice[self.seasonCol])

        # get redshift range for processing
        zRange = list(np.arange(self.zmin, self.zmax, self.zStep))

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
            return pd.DataFrame()

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
            return pd.DataFrame()

        # get simulation parameters
        gen_par = dur_z.groupby(['z', 'season']).apply(
            lambda x: self.calcDaymax(x)).reset_index()

        if self.verbose:
            print('getting simulation parameters')
            print(gen_par)

        # prepare pandas DataFrames for output
        vara_totdf = pd.DataFrame()

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

                vara_df = self.run_seasons(
                    dataSlice, [seas], gen_par, dur_z, ebvofMW, cadence, season_length, Nvisits, verbose=self.verbose, timer=self.timer)

                vara_totdf = pd.concat([vara_totdf, vara_df], sort=False)

        # estimate time of processing
        if self.verbose:
            print('finally - eop', time.time()-time_ref)
            toshow = ['pixRA', 'pixDec', 'healpixID', 'season', 'probasat', 'probasat_err',
                      'deltaT_sat', 'deltaT_befsat', 'nbef_sat', 'effipeak', 'effipeak_err', 'effipeak_sat', 'effipeak_sat_err', 'fractwi', 'twiphase']
            """
            if self.obsstat:
                # toshow += ['N_filters_night']
                for b in self.bandstat:
                    toshow += ['N_{}'.format(b)]
            """
            print(vara_totdf[toshow])

        print('finally', type(vara_totdf), vara_totdf.dtypes)
        return vara_totdf

    def get_ebv(self):
        """
        Method to estimate E(b-V)

        Returns
        -------
        E(B-V)

        """

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

        goodseasons = np.in1d(
            dataSlice['season'], np.array(seasons))
        dataSlice = dataSlice[goodseasons]

        goodfilters = np.in1d(
            dataSlice[self.filterCol], np.array([b for b in self.bands]))
        dataSlice = dataSlice[goodfilters]

        if len(dataSlice) <= 5:
            if self.verbose:
                print('Obs sample too small')
            return pd.DataFrame()

        if self.verbose:
            print('#### Processing season', seasons,
                  np.unique(dataSlice['healpixID']))

        groupnames = ['pixRA', 'pixDec', 'healpixID', 'season', 'x1', 'color']

        gen_p = gen_par[gen_par['season'].isin(seasons)]
        T0_min = np.min(gen_p['daymax'])
        T0_max = np.max(gen_p['daymax'])

        if gen_p.empty:
            if self.verbose:
                print('No generator parameter found')
            return pd.DataFrame()
        dur_z = dura_z[dura_z['season'].isin(seasons)]
        obs = pd.DataFrame(np.copy(dataSlice))
        # obs = obs[obs['season'].isin(seasons)]

        if self.timer:
            time_refb = time.time()

        # coaddition per night and per band (if requested by the user)
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

        # print('after stacker', seasons, len(obs))
        obs.sort(order='night')
        # print('data', obs[['night', 'filter',
        #                  'observationStartMJD', 'fieldRA', 'fieldDec']])
        # estimate m5 median and gaps
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

        # simulate supernovae and lc
        if self.verbose:
            print("LC generation")
            print('filters', np.unique(obs['filter']))

        sn = pd.DataFrame()
        if ebvofMW < 0.25:
            time_ref = time.time()
            # generate light curve
            lc = self.gen_LC(obs, ebvofMW, gen_p.to_records(
                index=False), verbose=self.verbose, timer=self.timer)
            # print('lc', np.unique(lc[self.exptimeCol]), gen_p)
            # estimate the total flux here
            lc['flux_e_max_pixel'] = lc['flux_e_sec'] * \
                self.pixel_max(lc[self.seeingCol]) * \
                lc['visitExposureTime']/lc['numExposures']
            lc['flux_e'] = lc['flux_e_sec']*lc['visitExposureTime']
            # add Poison noise (we may not be background dominated)
            lc['sigma_noise'] = lc['flux_e']/lc['snr_m5']
            lc['sigma_poisson'] = np.sqrt(lc['flux_e'])
            lc['sigma_all'] = lc['sigma_noise']**2+lc['sigma_poisson']**2
            lc['snr_all'] = lc['flux_e']/np.sqrt(lc['sigma_all'])
            lc['fluxerr'] = lc['flux']/lc['snr_all']
            # tag saturated points
            lc['sat'] = 0
            idx = lc['flux_e_max_pixel'] >= self.fullwell
            lc.loc[idx, 'sat'] = 1

            if self.figs_for_movie:
                self.plot_live(obs, lc.to_records(index=False), seasons[0],
                               T0_min, T0_max)

            lc = lc.round({'daymax': 3, 'z': 3})
            res = lc.groupby(['daymax', 'z']).apply(
                lambda x: self.calcLC(x)).reset_index()

            res['season'] = seasons[0]
            res['healpixID'] = self.healpixID
            res['pixRA'] = self.pixRA
            res['pixDec'] = self.pixDec

            r = []
            cols = res.columns
            for vv in cols:
                if 'mask' in vv:
                    r.append(vv)
            res = res.drop(columns=r)

            return res

            resb = res.groupby(['z']).apply(
                lambda x: self.proba(x)).reset_index()

            resb['season'] = seasons[0]
            resb['healpixID'] = self.healpixID
            resb['pixRA'] = self.pixRA
            resb['pixDec'] = self.pixDec

            if self.plotmetric:
                self.plotSat(resb.to_records(index=False))

            if self.verbose:
                print('processed', time.time()-time_ref)
            return resb
        return pd.DataFrame()

    def proba(self, grp):

        nevts = len(grp)
        nsat = np.sum(grp['sat'])
        npeak = np.sum(grp['ipeak'])
        npeak_sat = np.sum(grp['ipeak_sat'])

        dictout = {}

        dictout['probasat'] = [nsat/nevts]
        dictout['probasat_err'] = [np.sqrt(nsat*(1.-nsat/nevts)/nevts)]
        dictout['effipeak'] = [npeak/nevts]
        dictout['effipeak_sat'] = [npeak_sat/nsat]
        dictout['effipeak_err'] = [np.sqrt(npeak*(1.-npeak/nevts)/nevts)]
        dictout['effipeak_sat_err'] = [
            np.sqrt(npeak_sat*(1.-npeak_sat/nsat)/nsat)]

        for vv in ['deltaT_sat', 'deltaT_befsat', 'nbef_sat', 'fractwi', 'twiphase']:
            io = grp[vv] < 500.
            dictout[vv] = [np.nanmedian(grp[io][vv])]

        dictout['fractwi'] = [np.mean(grp['fractwi'])]

        return pd.DataFrame.from_dict(dictout)

    def plotExptime(self, data):
        """
        Method to plot visit exposure time vs MJD

        Parameters
        ---------------
        data: pandas df
          data to plot

        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(data[self.mjdCol], data[self.exptimeCol], 'ko')
        ax.set_xlabel('MJD [day]')
        ax.set_ylabel('Visit Exposure Time [sec]')

        plt.show()

    def calcLC(self, grp):
        """
        Method to estimate saturation parameters from LC

        Parameters
        ---------------
        grp: pandas grp

        Returns
        ----------
        pandas df with the following columns:
         'sat': 0 (if sat) or 1 (if sat)
        'nbef_sat': number of epochs before saturation
        'deltaT_sat': time of saturation(wrt begin LC)
        'deltaT_befsat': time before saturation(wrt begin LC)

        """

        # select LC points with SNR>=SNRmin

        SNID = 'SN_{}'.format(self.healpixID)

        T0 = grp.name[0]
        # select lc points with minimal SNR
        idx = grp['flux']/grp['fluxerr'] >= self.snr_min
        sel = grp[idx]
        # get lc info
        infos = self.lc_info(sel, T0)
        infos['SNID'] = SNID
        lcsel = Table.from_pandas(sel)
        for vv in ['z', 'daymax']:
            lcsel.meta[vv] = np.unique(lcsel[vv])[0]
        lcsel.meta['ebvofMW'] = 0.0
        lcsel.meta['status'] = 1
        """
        lcsel.meta['sn_type'] = 'SN_Ia'
        lcsel.meta['x0'] = -1
        lcsel.meta['x1'] = 0.0
        lcsel.meta['color'] = 0.0
        """
        lcsel = self.selection.select(lcsel)
        res = pd.DataFrame()

        col_rm = ['daymax', 'z']
        if lcsel is None:
            infos['isel'] = 0
            fita = self.dummy_fit()
            ra = pd.concat([fita.to_pandas(), infos], axis=1)
            ra = ra.drop(columns=col_rm)
            return ra
        else:
            infos['isel'] = 1
            fita, fitb = self.fit_it(sel)
            ra = pd.concat([fita.to_pandas(), infos], axis=1)
            rb = pd.concat([fitb.to_pandas(), infos], axis=1)
            res = pd.concat([ra, rb])
            res = res.drop(columns=col_rm)

        return res

    def fit_it(self, sel):
        """
        Method to fit lc with and without saturated points

        Parameters
        --------------
        sel: pandas df
           data (lc points) to fit

        Returns
        ----------
        fita, fitb: astropy tables
          fitted values

        """
        lcf = Table.from_pandas(sel)
        for vv in ['z', 'daymax']:
            lcf.meta[vv] = np.unique(lcf[vv])[0]
        lcf.meta['ebvofMW'] = 0.0
        lcf.meta['status'] = 1
        lcf.meta['sn_type'] = 'SN_Ia'
        lcf.meta['x0'] = -1
        lcf.meta['x1'] = 0.0
        lcf.meta['color'] = 0.0
        fita = self.fit_lc(lcf)
        fitb = self.fit_lc(lcf[lcf['sat'] == 0])

        fita['data'] = 'all'
        fitb['data'] = 'nosat'

        return fita, fitb

    def dummy_fit(self):

        dd = [('z', '<f8'), ('daymax', '<f8'), ('ebvofMW', '<f8'), ('status', '<i8'), ('z_fit', '<f8'),
              ('Cov_zz', '<f8'), ('Cov_zt0', '<f8'), ('Cov_zx0', '<f8'), ('Cov_zx1', '<f8'), ('Cov_zcolor', '<f8'), ('Cov_t0t0', '<f8'), ('Cov_t0x0', '<f8'), ('Cov_t0x1', '<f8'), (
            'Cov_t0color', '<f8'), ('Cov_x0x0', '<f8'), ('Cov_x0x1', '<f8'), ('Cov_x0color', '<f8'), ('Cov_x1x1', '<f8'), ('Cov_x1color', '<f8'), ('Cov_colorcolor', '<f8'),
            ('t0_fit', '<f8'), ('x0_fit', '<f8'), ('x1_fit', '<f8'), ('color_fit', '<f8'), ('hostebv_fit', '<f8'), ('hostr_v_fit', '<f8'), ('mwebv_fit', '<f8'), ('mwr_v_fit', '<f8'), ('mbfit', '<f8'), ('fitstatus', '<U5'), ('chisq', '<f8'), ('ndof', '<i8'), ('data', '<U3'), ('x0', '<f8'), ('x1', '<f8'), ('color', '<f8'), ('sn_type', '<U5')]

        dd = [('daymax', '<f8'), ('z', '<f8'), ('level_2', '<i8'), ('ebvofMW', '<f8'), ('status', '<i8'), ('sn_type', '<U5'), ('x0', '<i8'), ('x1', '<f8'), ('color', '<f8'), ('z_fit', '<f8'), ('Cov_t0t0', '<f8'), ('Cov_t0x0', '<f8'), ('Cov_t0x1', '<f8'), ('Cov_t0color', '<f8'), ('Cov_x0x0', '<f8'), ('Cov_x0x1', '<f8'), ('Cov_x0color', '<f8'), ('Cov_x1x1', '<f8'), ('Cov_x1color', '<f8'), ('Cov_colorcolor', '<f8'), ('t0_fit', '<f8'), ('x0_fit', '<f8'), ('x1_fit', '<f8'), ('color_fit', '<f8'), ('mbfit', '<f8'), ('fitstatus', '<U6'), ('chisq', '<f8'),
              ('ndof', '<i8'), ('data', '<U5'), ('sat', '<i8'), ('nbef_sat', '<i8'), ('deltaT_sat', '<f8'), ('deltaT_befsat', '<f8'), ('ipeak', '<i8'), ('ipeak_sat', '<i8'), ('fractwi', '<i8'), ('twiphase', '<f8'), ('satphase', '<f8'), ('nlcsat', '<i8'), ('SNID', '<U8'), ('isel', '<i8'), ('Cov_zz', '<f8'), ('Cov_zt0', '<f8'), ('Cov_zx0', '<f8'), ('Cov_zx1', '<f8'), ('Cov_zcolor', '<f8'), ('hostebv_fit', '<f8'), ('hostr_v_fit', '<f8'), ('mwebv_fit', '<f8'), ('mwr_v_fit', '<f8'), ('season', '<i8'), ('healpixID', '<i8'), ('pixRA', '<f8'), ('pixDec', '<f8')]
        pp = Table()

        for d in dd:
            ref_data = -99.0
            if 'i8' in d[1]:
                ref_data = 999
            if 'U3' in d[1]:
                ref_data = 'noo'
            if 'U5' in d[1]:
                ref_data = 'nodat'
            if 'U8' in d[1]:
                ref_data = 'SN_00000'
            c = Column([ref_data], name=d[0])
            pp.add_column(c)

        return pp

    def lc_info(self, sel, T0):
        # sort by mjd
        sel = sel.sort_values(by=['time'])

        # get mjdmin
        mjd_min = np.min(sel['time'])

        ipeak = 0
        satphase = -100.
        nlcsat = 0

        io = sel['sat'] == 1
        if len(sel[io]) > 0:
            satphase = np.median(sel[io]['phase'])
            nlcsat = len(sel[io])
        lcnosat = sel[sel['sat'] == 0]
        idxt = np.abs(lcnosat['time']-T0) <= 5.
        np_near_max = len(lcnosat[idxt])
        if np_near_max >= 3:
            ipeak = 1

        selsat = sel.groupby(['night']).apply(
            lambda x: self.satinfo(x)).reset_index()

        frac_twi = 0
        twiphase = np.median(selsat['twiphase'])
        idt = selsat['twivisits'] > 0
        if len(selsat[idt]) > 0:
            frac_twi = len(selsat[idt])/len(selsat)

        isat = 0
        ipeak_sat = 0
        nbef_sat = 999
        deltaT_sat = 999.
        deltaT_befsat = 999.

        idx = selsat['sat'] >= self.saturationLevel
        selsatb = selsat[idx]
        if len(selsatb) > 0:
            isat = 1
            time_sat = np.min(selsatb['time'])
            deltaT_sat = time_sat-mjd_min
            ido = sel['time'] < time_sat
            selnosat = sel[ido]
            nbef_sat = len(np.unique(selnosat['night']))
            deltaT_befsat = np.max(selnosat['time'])-mjd_min
            ipeak_sat = ipeak

        return pd.DataFrame({'sat': [isat],
                             'nbef_sat': [nbef_sat],
                             'deltaT_sat': [deltaT_sat],
                             'deltaT_befsat': [deltaT_befsat],
                             'ipeak': [ipeak],
                             'ipeak_sat': [ipeak_sat],
                             'fractwi': [frac_twi],
                             'twiphase': [twiphase],
                             'satphase': [satphase],
                             'nlcsat': [nlcsat]})

    def satinfo(self, x):

        sat = np.sum(x['sat'])/len(x)
        ttime = np.median(x['time'])
        twivisits = 0
        twiphase = -100.
        idx = x[self.exptimeCol] < 14.
        if len(x[idx]) > 0:
            twivisits = len(x[idx])
            twiphase = np.median(x[idx]['phase'])

        return pd.DataFrame({'sat': [sat],
                             'time': [ttime],
                             'twivisits': [twivisits],
                             'twiphase': [twiphase]}).reset_index()

    def plotLC(self, lcf, fitted_model, errors):

        import matplotlib.pyplot as plt
        sncosmo.plot_lc(lcf, model=fitted_model, errors=errors)
        plt.show()

    def plotSat(self, tab):
        """
        Method to plot some metric results

        Parameters
        ----------------
        tab: numpy array
          data to plot

        """
        if len(tab) == 0:
            return

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=5, figsize=(8, 12))
        plt.subplots_adjust(hspace=0)
        self.plotSingle(ax[0], tab, 'z', 'probasat',
                        'probasat_err', '$z$', 'Proba sat')
        self.plotSingle(ax[1], tab, 'z', 'nbef_sat',
                        '', '$z$', 'nLC before sat')
        self.plotSingle(ax[2], tab, 'z', 'deltaT_befsat', '',
                        '$z$', '$\Delta$t$^{bef sat}$ [day]')
        self.plotSingle(ax[3], tab, 'z', 'deltaT_sat', '',
                        '$z$', '$\Delta$t$^{sat}$ [day]')
        self.plotSingle(ax[4], tab, 'z', 'effipeak', 'effipeak_err',
                        '$z$', 'effipeak')

        plt.show()

    def plotSingle(self, ax, tab, varx, vary, erry='', legx='', legy='', ls='solid', color='k', label=None):
        """
        Method to plot

        Parameters
        ---------------
        ax: matplotlib axis
        tab: numpy array
           data to plot
        varx: str
          x-axis var
        vary: str
          y-axis var
        erry: str, opt
          y-axis var error(default: '')
        legx: str, opt
          x-axis label(default: '')
        legy: str, opt
          y-axis label(default: '')
        ls: str, opt
          linestyle(default: solid)
        color: str, opt
          line color (default: k)
        label: str, opt
          label for plot (default: None)
        """

        if erry is not '':
            ax.errorbar(tab[varx], tab[vary],
                        yerr=tab[erry], color=color, marker='o', label=label)
        else:
            ax.plot(tab[varx], tab[vary], color=color,
                    marker='o', ls=ls, label=label)

        ax.set_xlabel(legx)
        ax.set_ylabel(legy, color=color)
        if label:
            ax.legend()

    def plotLC_sncosmo(self, table, time_display):
        """ Light curve plot using sncosmo methods

        Parameters
        ---------------
        table: astropy table
         table with LS informations (flux, ...)
       time_display: float
         duration of the window display
        """

        import pylab as plt
        import sncosmo
        from sn_tools.sn_telescope import Telescope
        from astropy import units as u

        telescope = Telescope(airmass=1.2)
        prefix = 'LSST::'

        for band in 'grizy':
            name_filter = prefix+band
            if telescope.airmass > 0:
                bandpass = sncosmo.Bandpass(
                    telescope.atmosphere[band].wavelen,
                    telescope.atmosphere[band].sb,
                    name=name_filter,
                    wave_unit=u.nm)
            else:
                bandpass = sncosmo.Bandpass(
                    telescope.system[band].wavelen,
                    telescope.system[band].sb,
                    name=name_filter,
                    wave_unit=u.nm)
            # print('registering',name_filter)
            sncosmo.registry.register(bandpass, force=True)

        z = table.meta['z']
        if 'x1' in table.meta.keys():
            x1 = table.meta['x1']
            color = table.meta['color']
            x0 = table.meta['x0']
        else:
            x1 = 0.
            color = 0.
            x0 = 0.
        daymax = table.meta['daymax']

        model = sncosmo.Model('salt2')
        model.set(z=z,
                  c=color,
                  t0=daymax,
                  # x0=x0,
                  x1=x1)

        # display only 1 sigma LC points
        table = table[table['flux']/table['fluxerr'] >= 1.]

        sncosmo.plot_lc(data=table)

        """
        plt.draw()
        plt.pause(time_display)
        plt.close()
        """

    def duration_z(self, grp):
        """
        Method to estimate the season length vs redshift
        This is necessary to take into account boundary effects
        when estimating the number of SN that can be detected

        daymin, daymax = min and max MJD of a season
        T0_min(z) = daymin-(1+z)*min_rf_phase_qual
        T0_max(z) = daymax-(1+z)*max_rf_phase_qual
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
        Method to estimate T0(daymax) values for simulation.

        Parameters
        --------------
        grp: group(pandas df sense)
         group of data to process with the following cols:
           T0_min: T0 min value(per season)
           T0_max: T0 max value(per season)

        Returns
        ----------
        pandas df with daymax, min_rf_phase, max_rf_phase values

        """

        T0_max = grp['T0_max'].values
        T0_min = grp['T0_min'].values
        num = (T0_max-T0_min)/self.daymaxStep
        if T0_max-T0_min > 10:
            df = pd.DataFrame(np.linspace(
                T0_min, T0_max, int(num)), columns=['daymax'])
        else:
            df = pd.DataFrame([-1], columns=['daymax'])

        df['minRFphase'] = self.min_rf_phase
        df['maxRFphase'] = self.max_rf_phase

        return df

    def seasonInfo(self, grp):
        """
        Method to estimate seasonal info(cadence, season length, ...)

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
                filtcombi += '{}*{}/'.format(row['Nvisits'], row['filter'])

            df['filters_night'] = filtcombi
            """
            """
            # old code with bandstat
            for val in self.bandstat:
                # print(val, grpb[self.filterCol].str.count(val).sum())
                idx = grpb[self.filterCol] == val
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

    @ verbose_this('Simulation SN')
    @ time_this('Simulation SN')
    def gen_LC(self, obs, ebvofMW, gen_par, **kwargs):
        """
        Method to simulate LC and supernovae

        Parameters
        ---------------
        obs: numpy array
          array of observations(from scheduler)
        ebvofMW: float
           e(B-V) of MW for dust effects
        gen_par: numpy array
          array of parameters for simulation


        """

        time_ref = time.time()
        # LC estimation

        lc_tot = pd.DataFrame()
        for key, vals in self.lcFast.items():
            time_refs = time.time()
            gen_par_cp = np.copy(gen_par)
            lc = vals(obs, ebvofMW, gen_par_cp, bands='grizy')
            if self.verbose:
                print('End of simulation', key, time.time()-time_refs)

            lc_tot = pd.concat([lc_tot, lc], sort=False)

        if self.verbose:
            print('End of simulation after concat',
                  key, time.time()-time_refs)
        return lc_tot
