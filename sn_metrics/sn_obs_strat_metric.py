#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:46:37 2024

@author: philippe.gris@clermont.in2p3.fr
"""


class SNObsStratMetric:
    def __init__(self):

        self.name = 'sn_obs_strat_metric'
        print('instance of', self.name)

    def run(self, dataSlice,  imulti=0):

        print('running', len(dataSlice))
