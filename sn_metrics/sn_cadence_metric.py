import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from sn_stackers.coadd_stacker import CoaddStacker
import pandas as pd
import time
import healpy as hp
from sn_tools.sn_obs import getPix
import numpy.lib.recfunctions as rf


class SNCadenceMetric(BaseMetric):
    """
    Measure mean m5, cadence per band and per season for SN depth measurements.
    SN depth estimations are done in sn_plotter/sn_cadencePlotters.py

    Parameters
    --------------
    metricName : str, opt
      metric name
      Default : SNCadenceMetric
    mjdCol : str, opt
      mjd column name
      Default : observationStartMJD,
    RACol : str,opt
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
       list of seasons to process
       Default : -1 (all seasons)

    """

    def __init__(self, metricName='SNCadenceMetric',
                 mjdCol='observationStartMJD', RACol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', coadd=True, season=-1, nside=64, verbose=False,
                 uniqueBlocks=False, config=None, **kwargs):

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
        self.verbose = verbose

        cols = [self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol]
        self.stacker = None
        if coadd:
            cols += ['coadd']
            self.stacker = CoaddStacker(mjdCol=self.mjdCol, RACol=self.RACol, DecCol=self.DecCol, m5Col=self.m5Col, nightCol=self.nightCol,
                                        filterCol=self.filterCol, numExposuresCol=self.nexpCol, visitTimeCol=self.vistimeCol, visitExposureTimeCol='visitExposureTime')

        super(SNCadenceMetric, self).__init__(
            col=cols, metricDtype='object', metricName=metricName, **kwargs)

        self.filterNames = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        # self.config = config

        # sn parameters
        # sn_parameters = config['SN parameters']

        self.season = season
        # self.field_type = config['Observations']['fieldtype']
        # self.season = config['Observations']['season']
        # self.season = season
        # area = 9.6  # survey_area in sqdeg - 9.6 by default for DD
        # if self.field_type == 'WFD':
        # in that case the survey area is the healpix area
        self.area = hp.nside2pixarea(nside, degrees=True)
        self.nside = nside

    def run(self, dataSlice, slicePoint=None):
        """
        Runs the metric for each dataSlice

        Parameters
        ---------------
        dataSlice: simulation data
        slicePoint:  slicePoint (default None)


        Returns
        -----------
        record array with the following fields:
        fieldRA: RA of the field considered (float)
        fieldDec: Dec of the field considered (float)
        season:  season num (float)
        band:  band (str)
        m5_mean: mean five sigma depth (over the season) (float)
        cadence_mean: mean cadence (over the season) (float)
        Nobs: number of observation (int)
        healpixID: healpix Id (int)
        pixRA: RA of the pixel (float)
        pixDec: Dec of the pixel (float)
        deltaT_max: max internight gap (float)
        frac_dT_5: fraction of gaps higher than 5 days (float)
        frac_dT_10: fraction of gaps higher than 10 days (float)
        frac_dT_15: fraction of gaps higher than 15 days (float)
        frac_dT_20: fraction of gaps higher than 20 days (float)
        season_length: length of the season (float)
        numExposures: median number of exposures per season
        visitExposureTime: median exposure time per season
        """

        # add pixRA, pixDec and healpixID if not already included
        if 'pixRA' not in dataSlice.dtype.names:
            healpixID, pixRA, pixDec = getPix(self.nside,
                                              np.mean(
                                                  np.copy(dataSlice[self.RACol])),
                                              np.mean(np.copy(dataSlice[self.DecCol])))

            dataSlice = rf.append_fields(dataSlice, 'healpixID', [
                                         healpixID]*len(dataSlice))
            dataSlice = rf.append_fields(
                dataSlice, 'pixRA', [pixRA]*len(dataSlice))
            dataSlice = rf.append_fields(
                dataSlice, 'pixDec', [pixDec]*len(dataSlice))

        if self.stacker is not None:
            dataSlice = self.stacker._run(dataSlice)

        # dataSlice['healpixID'] = dataSlice['healpixID'].astype(int)

        time_ref = time.time()
        obsdf = pd.DataFrame(np.copy(dataSlice))

        obsdf['healpixID'] = obsdf['healpixID'].astype(int)

        res_filter = obsdf.groupby(['healpixID', self.seasonCol, self.filterCol]).apply(
            lambda x: self.anaSeason(x)).reset_index()

        # remove the u-band

        obsdf_no_u = obsdf.loc[lambda df: df[self.filterCol] != 'u', :].copy()

        if not obsdf_no_u.empty:

            res_all_but_u = obsdf_no_u.groupby(['healpixID', self.seasonCol]).apply(
                lambda x: self.anaSeason(x)).reset_index()

            res_all_but_u.loc[:, self.filterCol] = 'all'

            res = pd.concat([res_filter, res_all_but_u], sort=False)
        else:
            res = pd.DataFrame(res_filter)

        # res = res[res.columns.difference([self.filterCol,'level_4','level_5'])]
        # res = res.rename(columns={'{}_cad'.format(self.filterCol): self.filterCol})
        res = res[res.columns.difference(['level_2', 'level_3'])]
        res.loc[:, 'pixArea'] = self.area
        res.loc[:, 'nside'] = int(self.nside)

        res['nside'] = res['nside'].astype(int)

        # return the results as a numpy record array

        if self.verbose:
            print('processed', res)
        return res.to_records(index=False)

    def seq(self, dataSlice):
        """
        Estimate a filter sequence

        Parameters
        --------------

        dataSlice : panda df
         pandas df with a column self.filterCol

        Returns
        ----------
        pandas df with the filter sequence

        """

        seq_band = "".join(
            [val for val in np.unique(dataSlice[self.filterCol])])

        return pd.DataFrame({'seq': seq_band})

    def anaSeason(self, dataSlice):
        """
        Season analysis

        Parameters
        --------------
        dataSlice : pandas df
          data to analyze (scheduler simulations)

        Returns
        ----------
        pandas df estimated from sliceCalc

        """
        time_ref = time.time()
        dataSlice.sort_values(by=[self.mjdCol])

        return self.sliceCalc(dataSlice)

        """
        print('try to define sequences per night')

        # dataSlice['seq'] = dataSlice.groupby([self.nightCol])[self.filterCol].sum()
        print(dataSlice.columns)
        ll = dataSlice.groupby([self.nightCol])[self.filterCol].sum()

        res = pd.DataFrame()
        for seq in dataSlice['seq'].unique():
            index = dataSlice['seq'] == seq

            res = pd.concat([res, self.sliceCalc(dataSlice[index])])

        return res

        """

    def sliceCalc(self, dataSlice):
        """
        Method estimating some features (cadence, gaps, ...) for a set of data

        Parameters
        --------------
        dataSlice : pandas df
          data to process

        Returns
        ----------
        pandas df with the following results (columns):
        cadence_mean: mean cadence
        m5_mean: mean fiveSigmaDepth
        m5_median: median fiveSigmaDepth
        Nobs: number of observations
        season_length: season length
        gap_max: max gap
        pixRA: median pixRA
        pixDec: median pixDec
        gap_5,10,15,20: fraction of gaps bigger than 5,10,15, 20 days
        numExposures: median number of exposures
        visitExposureTime: median visit exposure time

        """

        # slicetime = dataSlice[self.mjdCol]-dataSlice[self.mjdCol].min()
        """
        r = []
        """
        # get the band
        band = "".join([val for val in np.unique(dataSlice[self.filterCol])])
        band = band.ljust(6)

        # this is to estimate the fraction of internight gaps higher than xx days
        internights = [5, 10, 15, 20]
        # myarr = np.array([[5],[10],[15],[20]], dtype= [('gap','f8')])
        myarr = np.array([5, 10, 15, 20], dtype=[('gap', 'f8')])
        deltaT_res = {}

        m5_mean = 0.
        m5_median = 0.
        gap_max = 0.
        stat = np.zeros((len(internights), 1))

        # for threshold in internights:
        #    deltaT_res['gap_{}'.format(threshold)] = 0.
        cadence = 0.
        season_length = 0.
        if len(dataSlice) > 0:
            m5_mean = np.mean(np.copy(dataSlice[self.m5Col]))
            m5_median = np.median(np.copy(dataSlice[self.m5Col]))
            deltaT = dataSlice[self.mjdCol].diff()

        if len(dataSlice) >= 3:
            season_length = np.max(
                dataSlice[self.mjdCol])-np.min(dataSlice[self.mjdCol])
            gap_max = np.max(deltaT)
            """
            for threshold in internights:
                idx = deltaT>= threshold
                deltaT_res['gap_{}'.format(threshold)] = len(
                    deltaT[idx])/(len(deltaT)-1)
            """
            # use broadcasting to estimate this - faster
            arr = np.array(deltaT)
            arr = arr[~np.isnan(arr)]
            diff = arr-myarr['gap'][:, np.newaxis]
            flag = diff >= 0
            flag_idx = np.argwhere(flag)
            madiff = np.ma.array(diff, mask=~flag)
            stat = np.ma.count(
                madiff, axis=1)/(np.ma.count(madiff, axis=1)+np.ma.count_masked(madiff, axis=1))

        if len(dataSlice) >= 5:
            """
            bins = np.arange(np.floor(dataSlice[self.mjdCol].min()), np.ceil(
                dataSlice[self.mjdCol].max()), 1.)
            c, b = np.histogram(dataSlice[self.mjdCol], bins=bins)
            if len(c)>= 2:
                cadence = 1. / c.mean()
            """
            # cadence = np.mean(slicetime)
            cadence = deltaT.median()

        resudf = pd.DataFrame({'cadence_mean': [cadence],
                               'm5_mean': [m5_mean],
                               'm5_median': [m5_median],
                               'Nobs': [len(dataSlice)],
                               'season_length': season_length,
                               'gap_max': gap_max,
                               'pixRA': np.median(dataSlice['pixRA']),
                               'pixDec': np.median(dataSlice['pixDec'])})

        for io in range(len(internights)):
            resudf.loc[:, 'gap_{}'.format(internights[io])] = stat[io]

        # for val in ['gap_{}'.format(thresh) for thresh in internights]:
        #    resudf.loc[:,val]= deltaT_res[val]

        # get the median number of obs per night

        medians = dataSlice.groupby(
            'night')[['numExposures', 'visitExposureTime']].median()

        for val in ['numExposures', 'visitExposureTime']:
            resudf.loc[:, val] = medians[val].median()

        # replace the band - necessary when processing all the filters
        # resudf = resudf[resudf.columns.difference([self.filterCol])]

        resudf.loc[:, '{}_cad'.format(self.filterCol)] = band

        return resudf
