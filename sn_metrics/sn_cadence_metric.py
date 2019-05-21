import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from sn_stackers.coadd_stacker import CoaddStacker

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
       list of seasons to process
       Default : -1 (all seasons)

    """

    def __init__(self, metricName='SNCadenceMetric',
                 mjdCol='observationStartMJD', RaCol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', coadd=True, season=-1,
                 uniqueBlocks=False, config=None, **kwargs):

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
        super(SNCadenceMetric, self).__init__(
            col=cols, metricDtype='object', metricName=metricName, **kwargs)

        self.filterNames = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        #self.config = config

        # sn parameters
        #sn_parameters = config['SN parameters']

        self.season = season
        #self.field_type = config['Observations']['fieldtype']
        #self.season = config['Observations']['season']
        #self.season = season
        #area = 9.6  # survey_area in sqdeg - 9.6 by default for DD
        #if self.field_type == 'WFD':
            # in that case the survey area is the healpix area
            #area = hp.nside2pixarea(
                #config['Pixelisation']['nside'], degrees=True)

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
        healpixID: healpix ID (int)
        pixRA: Ra of the pixel (float)
        pixDec: Dec of the pixel (float)
        deltaT_max: max internight gap (float)
        frac_dT_5: fraction of gaps higher than 5 days (float)
        frac_dT_10: fraction of gaps higher than 10 days (float)
        frac_dT_15: fraction of gaps higher than 15 days (float)
        frac_dT_20: fraction of gaps higher than 20 days (float)
        season_length: length of the season (float)
        """

        # Cut down to only include filters in correct wave range.
        goodFilters = np.in1d(dataSlice[self.filterCol], self.filterNames)
        dataSlice = dataSlice[goodFilters]
        if dataSlice.size == 0:
            return None
        dataSlice.sort(order=self.mjdCol)

        time = dataSlice[self.mjdCol]-dataSlice[self.mjdCol].min()
        r = []
        fieldRA = np.mean(dataSlice[self.RaCol])
        fieldDec = np.mean(dataSlice[self.DecCol])
        band = "".join([val for val in np.unique(dataSlice[self.filterCol])])
        band = band.ljust(6)
        

        seasons = self.season
        internights = [5,10,15,20]
        names = []
        if self.season == -1:
            seasons = np.unique(dataSlice[self.seasonCol])
        for season in seasons:
            ro = []
            deltaT_res = {}
            idx = dataSlice[self.seasonCol] == season
            sel = dataSlice[idx]
            sel.sort(order= self.mjdCol)
            if 'healpixID' in sel.dtype.names:
                healpixID = np.unique(sel['healpixID'])[0]
                pixRa = np.unique(sel['pixRa'])[0]
                pixDec = np.unique(sel['pixDec'])[0]
            else:
                healpixID = -1
                pixRa = -1
                pixDec = -1
            m5_mean = 0.
            deltaT_res['max']= 0.
            for threshold in internights:
                deltaT_res[threshold] = 0.
            cadence = 0.
            season_length = 0.
            if len(sel) > 0:
                m5_mean = np.mean(sel[self.m5Col])
                deltaT = sel[self.mjdCol][1:]-sel[self.mjdCol][:-1]
                
            if len(sel) >=3:
                season_length = np.max(sel[self.mjdCol])-np.min(sel[self.mjdCol])
                deltaT_res['max']= np.max(deltaT)
                for threshold in internights:
                    idx = deltaT >= threshold
                    deltaT_res[threshold] = len(deltaT[idx])/len(deltaT)
                
            if len(sel) >= 5:
                bins = np.arange(np.floor(sel[self.mjdCol].min()), np.ceil(
                    sel[self.mjdCol].max()), 1.)
                c, b = np.histogram(sel[self.mjdCol], bins=bins)
                if len(c)>= 2:
                    cadence = 1. / c.mean()
    
            ro = [fieldRA, fieldDec, season, band,
                      m5_mean, cadence, len(sel), healpixID, pixRa, pixDec]
            names =['fieldRA', 'fieldDec', 'season', 'band', 'm5_mean', 'cadence_mean','Nobs','healpixID','pixRa','pixDec']

            ro += [deltaT_res['max']]
            names += ['dT_max']
            for val in internights:
                ro += [deltaT_res[val]]
                names += ['frac_dT_{}'.format(val)]
            ro += [season_length]
            names += ['season_length']
            r.append(tuple(ro))

        
        res = np.rec.fromrecords(r, names=names)

        return res
