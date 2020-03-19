import numpy as np
from lsst.sims.maf.metrics import BaseMetric
import itertools
from sn_tools.sn_obs import LSSTPointing
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
import numpy.lib.recfunctions as rf


class SNGlobalMetric(BaseMetric):
    def __init__(self, metricName='SNGlobalMetric',
                 mjdCol='observationStartMJD', RACol='fieldRA', DecCol='fieldDec',
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
        RACol : str,opt
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
        self.RACol = RACol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.nightCol = nightCol
        self.obsidCol = obsidCol
        self.nexpCol = nexpCol
        self.vistimeCol = vistimeCol

        cols = [self.nightCol, self.filterCol, self.mjdCol, self.obsidCol,
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
        nvisits: total number of visits
        nvisits_u: u-band total number of visits
        nvisits_g: g-band total number of visits
        nvisits_r: r-band total number of visits
        nvisits_i: i-band total number of visits 
        nvisits_z: z-band total number of visits
        nvisits_y: y-band total number of visits
        """

        # sort data
        dataSlice.sort(order=self.mjdCol)

        # get the night number
        night = np.unique(dataSlice['night'])[0]

        dataSlice, nddf = self.fieldType(dataSlice)

        # estimate the number of filter changes this night

        nfc = len(list(itertools.groupby(dataSlice[self.filterCol])))-1
        iwfd = dataSlice['fieldType'] == 'WFD'
        selwfd = dataSlice[iwfd]
        nfc_noddf = len(list(itertools.groupby(selwfd[self.filterCol])))-1

        # estimate the area observed that night (LSST focal plane)
        # in total and per filter

        obs_area = {}
        nvisits = {}
        obs_area['all'] = self.area(dataSlice)
        nvisits['all'] = len(dataSlice)
        for band in 'ugrizy':
            idx = dataSlice[self.filterCol] == band
            sel = dataSlice[idx]
            obs_area[band] = 0.0
            nvisits[band] = 0
            if len(sel) > 0:
                obs_area[band] = self.area(sel)
                nvisits[band] = len(sel)

        r = []
        names = []
        r = [night, nfc, nfc_noddf, obs_area['all'], nvisits['all'], nddf]
        names = ['night', 'nfc', 'nfc_noddf', 'obs_area', 'nvisits', 'nddf']
        r += [obs_area[band] for band in 'ugrizy']
        names += ['obs_area_{}'.format(band) for band in 'ugrizy']
        r += [nvisits[band] for band in 'ugrizy']
        names += ['nvisits_{}'.format(band) for band in 'ugrizy']

        # get Moon vals
        for val in ['moonRA', 'moonDec', 'moonAlt', 'moonAz', 'moonDistance', 'moonPhase']:
            r += [np.median(dataSlice[val])]
            names += ['med_{}'.format(val)]

        res = np.rec.fromrecords([tuple(r)], names=names)

        return res

    def area(self, obs):
        """
        Method to estimate the area surveyed per night

        Parameters
        ---------------
        obs: numpy array
          array of observations

        Returns
        -----------
        union of polygon used to scan the area that night

        """
        polylist = []
        for val in obs:
            polylist.append(LSSTPointing(val[self.RACol], val[self.DecCol]))

        return unary_union(MultiPolygon(polylist)).area

    def fieldType(self, obs):
        """
        Method to estimate the type of field on the basis of observations

        Parameters
        ---------------
        obs: numpy array
          array of observations

        Returns
        ----------
        obs: numpy array
         original array with fieldtype
        nddf: int
          number of ddf found

        """

        rDDF = []
        for ra, dec in np.unique(obs[[self.RACol, self.DecCol]]):
            idx = np.abs(obs[self.RACol]-ra) < 1.e-5
            idx &= np.abs(obs[self.DecCol]-dec) < 1.e-5
            sel = obs[idx]
            if len(sel) >= 10:
                rDDF.append((ra, dec))

        nddf = len(rDDF)
        rtype = np.array(['WFD']*len(obs))
        if len(rDDF) > 0:
            RADecDDF = np.rec.fromrecords(
                rDDF, names=[self.RACol, self.DecCol])
            for (ra, dec) in RADecDDF[[self.RACol, self.DecCol]]:
                idx = np.argwhere(
                    (np.abs(obs[self.RACol]-ra) < 1.e-5) & (np.abs(obs[self.DecCol]-dec) < 1.e-5))
                rtype[idx] = 'DD'

        obs = rf.append_fields(obs, 'fieldType', rtype)
        return obs, nddf
