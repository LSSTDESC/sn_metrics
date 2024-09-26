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
                 mjdCol='observationStartMJD'):
        """
        class to estimate OS parameters

        Parameters
        ----------
        outDir : str
            output directory path.
        prodID : str
            production id (output file name).

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
        # grab seasons
        obs = pd.DataFrame.from_records(season(dataSlice))

        df_out_a = obs.groupby(['healpixID', 'pixRA', 'pixDec']).apply(
            lambda x: self.get_info(x)).reset_index()
        df_out_a['season'] = -1

        df_out_a = df_out_a.drop(['level_3'], axis=1)
        df_out_b = obs.groupby(['healpixID', 'pixRA', 'pixDec', 'season']).apply(
            lambda x: self.get_info(x)).reset_index()
        df_out_b = df_out_b.drop(['level_4'], axis=1)
        ddf = pd.concat((df_out_a, df_out_b), ignore_index=True)

        self.outdf = pd.concat((self.outdf, ddf))

        if len(self.outdf) >= 1000:
            self.dump()
            self.outdf = pd.DataFrame()

    def get_info(self, grp):

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
        if len(grp) >= 3:
            cad = grp[self.mjdCol].diff().median()

        ddf['cadence'] = cad

        # per band
        for b in self.bands:
            idx = grp[self.filterCol] == b
            sel = grp[idx]
            ddf['nvisits_{}'.format(b)] = len(sel)
            ddf['expTime_{}'.format(b)] = sel[self.expTimeCol].sum()
            cad = -1.0
            if len(sel) >= 3:
                cad = sel[self.mjdCol].diff().median()
            ddf['cadence_{}'.format(b)] = cad

        return ddf

    def dump(self):
        """
        Method to dump pandas df in file

        Returns
        -------
        None.

        """

        self.outdf['healpixID'] = self.outdf['healpixID'].astype(int)
        self.outdf.to_hdf(self.outName, key='SN', append=True)

    def finish(self):
        """
        Finish: dump residual events.

        Returns
        -------
        None.

        """

        if len(self.outdf) > 0:
            self.dump()
