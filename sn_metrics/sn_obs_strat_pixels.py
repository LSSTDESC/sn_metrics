#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:46:37 2024

@author: philippe.gris@clermont.in2p3.fr
"""
from sn_tools.sn_io import checkDir
import pandas as pd
import numpy as np
import os
from sn_tools.sn_obs import season


class SNObsStratPixel:
    def __init__(self, outDir, prodID, filterCol='filter',
                 expTimeCol='visitExposureTime',
                 mjdCol='observationStartMJD',
                 m5Col='fiveSigmaDepth',
                 timescale='year'):
        """
        class to estimate OS parameters per pixel

        Parameters
        ----------
        outDir : str
            output directory.
        prodID : str
            prod id (output file name).
        filterCol : str, optional
            filter column name. The default is 'filter'.
        expTimeCol : str, optional
            exposure time column. The default is 'visitExposureTime'.
        mjdCol : str, optional
            mjd column name. The default is 'observationStartMJD'.
        m5Col : str, optional
            m5 column name. The default is 'fiveSigmaDepth'.
        timescale: str, optional.
            Time scale to use (year/season). The default is 'year'.

        Returns
        -------
        None.

        """
        self.name = 'sn_obs_strat_pixels'

        checkDir(outDir)

        self.outName = '{}/{}.hdf5'.format(outDir, prodID)
        if os.path.isfile(self.outName):
            os.system('rm {}'.format(self.outName))

        self.filterCol = filterCol
        self.expTimeCol = expTimeCol
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.timescale = timescale

        self.bands = 'ugrizy'

        self.outdf = pd.DataFrame()

    def run(self, dataSlice,  imulti=0):
        """
        Run method called by Process

        Parameters
        ----------
        dataSlice : numpy array
            Data to process.
        imulti : int, optional
            Flag. The default is 0.

        Returns
        -------
        None.

        """

        fieldName = np.unique(dataSlice['field'])[0]
        # grab seasons
        if self.timescale == 'season':
            obs = pd.DataFrame.from_records(season(dataSlice, season_gap=80.))
        else:
            obs = pd.DataFrame.from_records(dataSlice)
            obs['year'] = (obs['observationStartMJD'] -
                           obs['lsst_start'])/365.+1
            obs['year'] = obs['year'].astype(int)

        df_out_a = obs.groupby(['healpixID', 'pixRA', 'pixDec']).apply(
            lambda x: self.get_info(x)).reset_index()
        df_out_a[self.timescale] = -1

        df_out_a = df_out_a.drop(['level_3'], axis=1)
        df_out_b = obs.groupby(['healpixID', 'pixRA', 'pixDec',
                                self.timescale]).apply(
            lambda x: self.get_info(x)).reset_index()
        df_out_b = df_out_b.drop(['level_4'], axis=1)
        ddf = pd.concat((df_out_a, df_out_b), ignore_index=True)
        ddf['field'] = fieldName

        self.outdf = pd.concat((self.outdf, ddf))

        if len(self.outdf) >= 1000:
            self.dump()
            self.outdf = pd.DataFrame()

    def get_info(self, grp):
        """
        Method to get infos

        Parameters
        ----------
        grp : pandas df
            data to process.

        Returns
        -------
        ddf : pandas df
            processed data.

        """

        # sort data
        grp = grp.sort_values(by=[self.mjdCol])

        # total number of visits
        nvisits = len(grp)
        ddf = pd.DataFrame([nvisits], columns=['nvisits'])

        # total exposure time

        ddf['exposuretime'] = grp[self.expTimeCol].sum()

        # season length
        ddf['season_length'] = grp[self.mjdCol].max()-grp[self.mjdCol].min()

        # cadence
        cad = -1.0
        nnights = len(grp['night'].unique())
        if nnights >= 3:
            rr = grp.groupby(['night'])[self.mjdCol].mean().reset_index()
            rr = rr.sort_values(by=['night'])
            cad = rr[self.mjdCol].diff().mean()

        ddf['cadence'] = cad

        # per band
        for b in self.bands:
            idx = grp[self.filterCol] == b
            sel = grp[idx]
            cad = -1.
            nvisits = 0
            exptime = 0.
            m5 = -999.0
            if len(sel) > 0:
                nvisits = len(sel)
                exptime = sel[self.expTimeCol].sum()
                # coadded m5
                m5 = 1.25*np.log10(np.sum(10**(0.8*sel[self.m5Col])))

                nnights = len(sel['night'].unique())
                if nnights >= 3:
                    rr = sel.groupby(['night'])[
                        self.mjdCol].mean().reset_index()
                    rr = rr.sort_values(by=['night'])
                    cad = rr[self.mjdCol].diff().mean()

            ddf['m5_{}'.format(b)] = m5
            ddf['nvisits_{}'.format(b)] = nvisits
            ddf['expTime_{}'.format(b)] = exptime
            ddf['cadence_{}'.format(b)] = cad

        """
        print('**************************')
        print(ddf.dtypes)
        """
        return ddf

    def dump(self):
        """
        Method to dump pandas df in file

        Returns
        -------
        None.

        """
        from time import sleep
        import random

        self.outdf['healpixID'] = self.outdf['healpixID'].astype(int)
        excpt = True

        while (excpt):
            try:
                """
                vv = random.randint(1, 101)
                print('waiting', 0.001*vv)
                sleep(0.001*vv)
                """
                self.outdf.to_hdf(self.outName, key='SN', append=True)
                excpt = False
            except (RuntimeError, TypeError, NameError) as err:
                print('err', err)
                excpt = True

    def finish(self):
        """
        Finish: dump residual events.

        Returns
        -------
        None.

        """

        if len(self.outdf) > 0:
            self.dump()
