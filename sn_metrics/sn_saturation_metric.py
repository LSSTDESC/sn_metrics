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
                 verbose=False, timer=False, plotmetric=False, snr_min=5.,
                 lightOutput=False, ebvofMW=-1., fracpixel=None, fullwell=90000., saturationLevel=0.99,
                 obsstat=False, **kwargs):

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
        self.daymaxStep = 3.  # daymax step
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

        # Get ebvofMW here
        ebvofMW = self.ebvofMW
        # get pixel id
        pixRA = np.unique(dataSlice['pixRA'])[0]
        pixDec = np.unique(dataSlice['pixDec'])[0]
        healpixID = int(np.unique(dataSlice['healpixID'])[0])

        self.pixInfo = {}
        self.pixInfo['healpixID'] = healpixID
        self.pixInfo['pixRA'] = pixRA
        self.pixInfo['pixDec'] = pixDec

        if ebvofMW < 0.:
            RA = np.mean(dataSlice[self.RACol])
            Dec = np.mean(dataSlice[self.DecCol])
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
                      'deltaT_sat', 'deltaT_befsat', 'nbef_sat', 'effipeak', 'effipeak_err']
            """
            if self.obsstat:
                # toshow += ['N_filters_night']
                for b in self.bandstat:
                    toshow += ['N_{}'.format(b)]
            """
            print(vara_totdf[toshow])

        return vara_totdf

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

        goodfilters = np.in1d(
            dataSlice[self.filterCol], np.array(['g', 'r', 'i']))
        dataSlice = dataSlice[goodfilters]

        goodseasons = np.in1d(
            dataSlice['season'], np.array(seasons))
        dataSlice = dataSlice[goodseasons]

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
            return None, None
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
            lc['flux_e'] = lc['flux_e_sec'] * \
                self.pixel_max(lc[self.seeingCol]) * \
                lc['visitExposureTime']/lc['numExposures']
            """
            if self.plotmetric:
                self.plotLC(obs, lc.to_records(index=False), T0_min, T0_max)
            """
            res = lc.groupby(['daymax', 'z']).apply(
                lambda x: self.calcLC(x)).reset_index()
            print(res[['z', 'sat', 'ipeak', 'daymax']])
            resb = res.groupby(['z']).apply(
                lambda x: self.proba(x)).reset_index()

            print(
                resb[['z', 'probasat', 'probasat_err', 'effipeak', 'effipeak_err']])

            resb['season'] = seasons[0]
            if self.plotmetric:
                self.plotSat(resb.to_records(index=False))

            if self.verbose:
                print('processed', time.time()-time_ref)
            return resb
        return None

    def proba(self, grp):

        nevts = len(grp)
        nsat = np.sum(grp['sat'])
        npeak = np.sum(grp['ipeak'])

        dictout = {}

        dictout['probasat'] = [nsat/nevts]
        dictout['probasat_err'] = [np.sqrt(nsat*(1.-nsat/nevts)/nevts)]
        dictout['effipeak'] = [npeak/nevts]
        dictout['effipeak_err'] = [np.sqrt(npeak*(1.-npeak/nevts)/nevts)]

        for vv in ['deltaT_sat', 'deltaT_befsat', 'nbef_sat']:
            io = grp[vv] < 500.
            dictout[vv] = [np.median(grp[io][vv])]

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

    def plotLC(self, obs, lc, T0_min, T0_max):

        import matplotlib.pyplot as plt
        from astropy.table import Table

        zref = 0.02
        idx = np.abs(lc['z']-zref) < 1.e-5
        lc = lc[idx]
        lc.sort(order='daymax')
        daymaxs = np.unique(lc['daymax'])
        ndaymax = 0
        npeak = 0
        isat = 0
        r_nbef = []
        r_deltaT_befsat = []

        r = []
        for iday, daymax in enumerate(daymaxs):
            idx = np.abs(lc['daymax']-daymax) < 1.e-5
            idx &= lc['snr_m5'] >= self.snr_min
            sel = lc[idx]
            print('hhh', type(sel))

            lctab = Table(sel)
            lctab.meta['z'] = zref
            lctab.meta['daymax'] = daymax

            fig, ax = plt.subplots(nrows=5, figsize=(
                8, 12), constrained_layout=True)
            fig.suptitle('healpixID: {}'.format(
                self.pixInfo['healpixID']), fontsize='medium')
            self.plotObs(ax[0], obs, daymax, T0_min, T0_max, whatx=self.mjdCol, whaty=self.m5Col,
                         xlabel='MJD [day]', ylabel='5$\sigma$ depth [mag]')
            ndaymax += 1

            print('here', len(sel), daymax)
            if len(sel) > 0:
                self.plotLC_sncosmo(lctab, 20)
                axtitle = '$z$={} - T$_0$={}'.format(zref, np.round(daymax, 1))
                self.plotLC_T0(ax[1], sel, daymax, axtitle=axtitle)
                npeakobs, isatobs, nbef_sat, deltaT_befsat, deltaT_sat = self.statShape(
                    sel, daymax)
                npeak += npeakobs
                isat += isatobs
                if nbef_sat < 100.:
                    r_nbef.append(nbef_sat)
                if deltaT_befsat < 100.:
                    r_deltaT_befsat.append(deltaT_befsat)

            print('nbef', r_nbef, r_deltaT_befsat)
            r.append((npeak/ndaymax, isat/ndaymax, np.median(r_nbef),
                      np.median(r_deltaT_befsat), daymax))
            res = np.rec.fromrecords(
                r, names=['effipeak', 'probasat', 'nbef_sat', 'deltaT_befsat', 'daymax'])
            # ax[2].plot(res['daymax'], res['effipeak'], color='k', marker='o')
            self.plotSingle(ax[2], res, 'daymax', 'effipeak',
                            '', 'T$_0$ [day]', '$\epsilon_{peak}$')
            self.plotSingle(ax[2].twinx(), res, 'daymax', 'probasat',
                            '', 'T$_0$ [day]', 'Saturation proba.', color='r')
            self.plotSingle(ax[3], res, 'daymax', 'nbef_sat',
                            '', 'T$_0$ [day]', 'N$_{LC}$ bef. sat.')
            self.plotSingle(ax[4], res, 'daymax', 'deltaT_befsat',
                            '', 'T$_0$ [day]', '$\Delta$t  bef. sat. [day]')

            plt.close()
            """
            figname = 'figures/{}_{}.jpg'.format(
            self.pixInfo['healpixID'], iday)
            plt.savefig(figname)
            plt.close()
            """
            """
            plt.draw()
            plt.pause(2)
            plt.close()
            """
            # plt.show()

    def plotObs(self, ax, obs, daymax, T0_min, T0_max, whatx, whaty, xlabel, ylabel):
        """
        Method to plot observations

        Parameters
        ---------------
        ax: matplotlib axis
          axis for the plot
        obs: numpy array
            observations to plot
        daymax: float
           T0 LC
        T0_min: float
          min T0 value
        T0_max: float
          max T0 value
        whatx: str
           x-axis variable
        whaty: str
           y-axis variable
        xlabel: str
          x-axis label
        ylabel: str
          y-axis label
        """

        for b in np.unique(obs[self.filterCol]):
            idx = obs[self.filterCol] == b
            sel = obs[idx]
            ax.plot(sel[whatx], sel[whaty],
                    color=filtercolors[b[-1]], marker='o', label='{} band'.format(b[-1]), ls='None')

        ax.plot([daymax]*2, [np.min(obs[whaty]),
                             np.max(obs[whaty])], ls='dashed', color='k')

        ax.plot([T0_min]*2, [np.min(obs[whaty]),
                             np.max(obs[whaty])], ls='solid', color='r')

        ax.plot([T0_max]*2, [np.min(obs[whaty]),
                             np.max(obs[whaty])], ls='solid', color='r')

        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def plotLC_T0(self, ax, sel, daymax, whatx='time', whaty='flux_e', xlabel='MJD [day]', ylabel='max flux pixel [e/s]', axtitle=''):
        """
        Method to plot a light curve corresponding to T0

        Parameters
        ---------------
        ax: matplotlib axis
          axis for the plot
        sel: array
          LC to plot
        daymax: float
           T0 LC
        whatx: str, opt
          x-variable to plot(default: time)
        whaty: str, opt
          y-variable to plot(default: flux_e)
        legx: str, opt
          x-axis label(default: 'MJD [day]')
        legy: str, opt
          y-axis label(default: 'max flux pixel[e/s])
        axtitle: str, opt
          axis title(default: '')
        """
        idx = sel['snr_m5'] >= self.snr_min
        sel = sel[idx]
        for band in np.unique(sel['band']):
            ib = sel['band'] == band
            selb = sel[ib]
            ax.plot(selb[whatx], selb[whaty],
                    '{}o'.format(filtercolors[band[-1]]), label='{} band'.format(band[-1]))
        # ax.legend()
        tmin, tmax = np.min(sel[whatx]), np.max(sel[whatx])
        fluxmin, fluxmax = np.min(sel[whaty]), np.max(sel[whaty])
        ax.plot([tmin, tmax], [self.fullwell]*2, color='k')
        ax.plot([daymax]*2, [fluxmin, fluxmax], color='k', ls='solid')
        ax.plot([daymax-5.]*2,
                [fluxmin, fluxmax], color='k', ls='dashed')
        ax.plot([daymax+5.]*2,
                [fluxmin, fluxmax], color='k', ls='dashed')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(axtitle, loc='right', color='b')

    def statShape(self, sel, daymax):
        """
        Method to estimate a set of features related to the shape of LC

        Parameters
        ---------------
        sel: array
          lc to process
        daymay: float
          T0 LC

        Returns
        -----------
        npeak: int
          1 if number of LC points in a 5 days window around peak is larger that 3, 0 otherwise
        nlc_bef_sat: int
          number of LC points before sat

        """
        print('hello', type(sel))
        sel.sort(order='time')
        idx = sel['snr_m5'] >= self.snr_min
        sel = sel[idx]

        # get mjdmin
        mjd_min = np.min(sel['time'])

        # get non sat points
        idnosat = sel['flux_e'] <= self.fullwell
        sel_nosat = sel[idnosat]

        npeak = 0
        # get peak measurement
        idpeak = np.abs(sel_nosat['time']-daymax) <= 5
        n_aroundpeak = len(sel_nosat[idpeak])
        if n_aroundpeak >= 3:
            npeak = 1

        seldf = pd.DataFrame(np.copy(sel))
        seldf['sat'] = 0
        idsat = seldf['flux_e'] > self.fullwell
        seldf.loc[idsat, 'sat'] = 1

        print(seldf[['night', 'flux_e', 'sat', 'time']])
        selsat = seldf.groupby(['night']).apply(lambda x: pd.DataFrame({'sat': [np.sum(x['sat'])/len(x)],
                                                                        'time': [np.median(x['time'])]})).reset_index()
        print(selsat)

        isat = 0
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
            if len(selnosat) > 0:
                nbef_sat = len(np.unique(selnosat['night']))
                deltaT_befsat = np.max(selnosat['time'])-mjd_min
            else:
                nbef_sat = 0
                deltaT_befsat = 0

        print(isat, nbef_sat, deltaT_befsat, deltaT_sat)
        return npeak, isat, nbef_sat, np.round(deltaT_befsat, 2), np.round(deltaT_sat, 2)

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

        T0 = grp.name[0]
        idx = grp['snr_m5'] >= self.snr_min
        sel = grp[idx]

        # sort by mjd
        sel = sel.sort_values(by=['time'])

        # get mjdmin
        mjd_min = np.min(sel['time'])

        # get saturated points (if any)
        sel['sat'] = 0
        idx = sel['flux_e'] >= self.fullwell
        sel.loc[idx, 'sat'] = 1

        ipeak = 0
        lcnosat = sel[sel['sat'] == 0]
        idxt = np.abs(lcnosat['time']-T0) <= 5.
        np_near_max = len(lcnosat[idxt])
        if np_near_max >= 3:
            ipeak = 1

        selsat = sel.groupby(['night']).apply(lambda x: pd.DataFrame({'sat': [np.sum(x['sat'])/len(x)],
                                                                      'time': [np.median(x['time'])]})).reset_index()

        isat = 0
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

        return pd.DataFrame({'sat': [isat],
                             'nbef_sat': [nbef_sat],
                             'deltaT_sat': [deltaT_sat],
                             'deltaT_befsat': [deltaT_befsat],
                             'ipeak': [ipeak]})

    def plotSat(self, tab):
        """
        Method to plot some metric results

        Parameters
        ----------------
        tab: numpy array
          data to plot

        """
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

    @verbose_this('Simulation SN')
    @time_this('Simulation SN')
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
