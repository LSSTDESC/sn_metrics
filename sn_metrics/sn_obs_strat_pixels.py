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


class SNObsStratPixel:
    def __init__(self, outDir, prodID):
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

        nvisits = len(dataSlice)
        ddf = pd.DataFrame([nvisits], columns=['nvisits'])
        for vv in ['healpixID', 'pixRA', 'pixDec']:
            ddf[vv] = np.unique(dataSlice[vv])

        self.outdf = pd.concat((self.outdf, ddf))

        if len(self.outdf) > 1000:
            self.dump()
            self.outdf = pd.DataFrame()

    def dump(self):
        """
        Method to dump pandas df in file

        Returns
        -------
        None.

        """

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
