#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:46:37 2024

@author: philippe.gris@clermont.in2p3.fr
"""


class SNObsStratPixel:
    def __init__(self, outDir, outName):

        self.name = 'sn_obs_strat_pixels'
        print('instance of', self.name)

    def run(self, dataSlice,  imulti=0):

        print('running', len(dataSlice),
              dataSlice[['healpixID', 'pixRA', 'pixDec']])
