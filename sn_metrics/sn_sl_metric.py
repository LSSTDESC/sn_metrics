import numpy as np
from rubin_sim.maf.metrics import BaseMetric
import healpy as hp
from sn_tools.sn_stacker import CoaddStacker
import time
import pandas as pd
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery


class SLSNMetric(BaseMetric):
    def __init__(self, metricName='SLSNMetric',
                 mjdCol='observationStartMJD', RACol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', m5Col='fiveSigmaDepth', season=[-1],
                 nside=64, coadd=False, verbose=False,
                 uniqueBlocks=False, **kwargs):
        """
        Strong Lensed SN metric

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
        season: int (list) or -1, opt
         season to process (default: -1: all seasons)
        nside: int, opt
         healpix parameter nside (default: 64)


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
        self.seasonCol = 'season'
        self.m5Col = m5Col
        self.nside = nside
        self.verbose = verbose

        cols = [self.nightCol, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol]

        super(SLSNMetric, self).__init__(
            col=cols, metricDtype='object', metricName=metricName, **kwargs)

        # get area of the pixels
        self.area = hp.nside2pixarea(self.nside, degrees=True)
        self.season = season
        self.bands = 'ugrizy'

        # stacker
        if coadd:
            self.stacker = CoaddStacker(mjdCol=self.mjdCol, RACol=self.RACol, DecCol=self.DecCol, m5Col=self.m5Col, nightCol=self.nightCol,
                                        filterCol=self.filterCol, numExposuresCol=self.nexpCol, visitTimeCol=self.vistimeCol, visitExposureTimeCol='visitExposureTime')

    def run(self, dataSlice, slicePoint=None):
        """
        Runs the metric for each dataSlice

        Parameters
        ---------------
        dataSlice: simulation data
        slicePoint:  slicePoint (default None)


        Returns
        -----------
        pandas df with the following cols:
        healpixId, pixRA, pixDec: pixel ID, RA and Dec
        season_length: cumulative season length
        gap_median: median gap (all filters)
        gap_max: max gap (all filters)
        gap_median_u, g, r,i,z: median gap for filters u,g,r,i,z
        gap_max_u, g, r,i,z: max gap for filters u,g,r,i,z
        area: area of observation (pixel area here)
        """

        # sort data
        dataSlice.sort(order=self.mjdCol)
        if len(dataSlice) == 0:
            return None

        time_refb = time.time()

        if self.stacker is not None:
            dataSlice = self.stacker._run(dataSlice)
        """    
        print('stacker',time.time()-time_refb)
        """

        # get coordinates for this pixel
        """
        RA = np.mean(dataSlice[self.RACol])
        Dec = np.mean(dataSlice[self.DecCol])
            

        
        print('oooo',RA,Dec)
        table = hp.ang2vec([RA], [Dec], lonlat=True)
        
        healpixs = hp.vec2pix(self.nside, table[:, 0], table[:, 1], table[:, 2], nest=True)
        coord = hp.pix2ang(self.nside, healpixs, nest=True, lonlat=True)
        
        healpixId, pixRA, pixDec = healpixs[0], coord[0][0],coord[1][0]
        """

        seasons = self.season

        if self.season == [-1]:
            seasons = np.unique(dataSlice[self.seasonCol])

        # get infos on seasons
        info_season = self.seasonInfo(dataSlice, seasons)
        if info_season is None:
            return None

        # print(test)
        # print(info_season.dtype)
        # print(info_season)

        healpixId = int(np.unique(dataSlice['healpixID'])[0])
        pixRA = np.unique(dataSlice['pixRA'])[0]
        pixDec = np.unique(dataSlice['pixDec'])[0]
        # get ebvofMW for this pixel
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

        if ebvofMW >= 0.25:
            return None

        r = [healpixId, pixRA, pixDec]
        names = ['healpixId', 'pixRA', 'pixDec']
        # get the cumulative season length

        season_length = np.sum(info_season['season_length'])
        r.append(season_length)
        names.append('season_length')
        # get gaps
        r.append(np.mean(info_season['gap_median']))
        names.append('gap_median')
        r.append(np.mean(info_season['gap_max']))
        names.append('gap_max')

        for band in self.bands:
            r.append(np.mean(info_season['gap_median_{}'.format(band)]))
            r.append(np.mean(info_season['gap_max_{}'.format(band)]))
            names.append('gap_median_{}'.format(band))
            names.append('gap_max_{}'.format(band))

        r.append(self.area)
        names.append('area')

        res = np.rec.fromrecords([r], names=names)

        # print('done',time.time()-time_refb)

        if self.verbose:
            print('processed', res)
        return pd.DataFrame(res)

    def seasonInfo(self, dataSlice, seasons):
        """
        Get info on seasons for each dataSlice
        Parameters
        -----
        dataSlice : array
          array of observations
        Returns
        -----
        recordarray with the following fields:
        season, cadence, season_length, MJDmin, MJDmax,Nvisits
        gap_median, gap_max for each band
        """

        rv = []
        time_ref = time.time()
        for season in seasons:
            time_refb = time.time()
            idx = (dataSlice[self.seasonCol] == season)
            slice_sel = dataSlice[idx]

            if len(slice_sel) < 5:
                continue
            slice_sel.sort(order=self.mjdCol)
            mjds_season = slice_sel[self.mjdCol]
            cadence = mjds_season[1:]-mjds_season[:-1]
            mjd_min = np.min(mjds_season)
            mjd_max = np.max(mjds_season)
            season_length = mjd_max-mjd_min
            Nvisits = len(slice_sel)
            median_gap = np.median(cadence)
            mean_gap = np.mean(cadence)
            max_gap = np.max(cadence)

            rg = [float(season), np.mean(cadence), season_length,
                  mjd_min, mjd_max, Nvisits, median_gap, mean_gap, max_gap]

            # night gaps per band
            for band in self.bands:
                idb = slice_sel['filter'] == band
                selb = slice_sel[idb]
                if len(selb) >= 2:
                    gaps = selb[self.mjdCol][1:]-selb[self.mjdCol][:-1]
                    # print('alors',band,gaps,np.median(gaps),np.max(gaps))
                    rg += [np.median(gaps), np.mean(gaps), np.max(gaps)]

                else:
                    rg += [0.0, 0.0, 0.0]

            rv.append(tuple(rg))

        info_season = None
        names = ['season', 'cadence', 'season_length',
                 'MJD_min', 'MJD_max', 'Nvisits']
        names += ['gap_median', 'gap_mean', 'gap_max']
        for band in self.bands:
            names += ['gap_median_{}'.format(band), 'gap_mean_{}'.format(
                band), 'gap_max_{}'.format(band)]

        if len(rv) > 0:
            info_season = np.rec.fromrecords(
                rv, names=names)

        # print('finished',time.time()-time_ref)
        return info_season
