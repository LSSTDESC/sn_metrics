import numpy as np
from lsst.sims.maf.metrics import BaseMetric
import itertools
from sn_tools.sn_obs import LSSTPointing
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon

class SNGlobalMetric(BaseMetric):
    def __init__(self, metricName='SNGlobalMetric',
                 mjdCol='observationStartMJD', RaCol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime',
                 uniqueBlocks=False, **kwargs):

        """
        Estimate global properties (per night) of a given cadence

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

        """
        self.mjdCol = mjdCol
        self.filterCol = filterCol
        self.RaCol = RaCol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.nightCol = nightCol
        self.obsidCol = obsidCol
        self.nexpCol = nexpCol
        self.vistimeCol = vistimeCol



        cols = [self.nightCol,self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol]
       
        super(SNGlobalMetric, self).__init__(
            col=cols, metricDtype='object', metricName=metricName, **kwargs)

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
        night: night number (int)
        nfc: number of filter changes (int)
        obs_area: observed area (float) in deg2
        obs_area_u: u-band observed area (float) in deg2
        obs_area_g: g-band observed area (float) in deg2
        obs_area_r: r-band observed area (float) in deg2
        obs_area_i: i-band observed area (float) in deg2
        obs_area_z: z-band observed area (float) in deg2
        obs_area_y: y-band observed area (float) in deg2

        """

        #sort data
        dataSlice.sort(order=self.mjdCol)

        #get the night number
        night = np.unique(dataSlice['night'])[0]

        #estimate the number of filter changes this night

        nfc = len(list(itertools.groupby(dataSlice['filter'])))-1
    
        
        #estimate the area observed that night (LSST focal plane)
        # in total and per filter
        
        obs_area = {}

        obs_area['all'] = self.area(dataSlice)

        for band in 'ugrizy':
            idx = dataSlice[self.filterCol] == band
            sel = dataSlice[idx]
            obs_area[band] = 0.0
            if len(sel) > 0:
                obs_area[band] = self.area(sel)

        r = []
        names = []
        r = [night,nfc,obs_area['all']]
        names = ['night','nfc','obs_area']
        r += [obs_area[band] for band in 'ugrizy']
        names += ['obs_area_{}'.format(band) for band in 'ugrizy']
        
        
        res = np.rec.fromrecords([tuple(r)], names=names)

        return res

        
    def area(self,obs):
        polylist = []
        for val in obs:
            polylist.append(LSSTPointing(val[self.RaCol],val[self.DecCol]))

        return unary_union(MultiPolygon(polylist)).area
        
