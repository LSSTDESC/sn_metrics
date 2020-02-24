import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from sn_stackers.coadd_stacker import CoaddStacker
import healpy as hp
import numpy.lib.recfunctions as rf
import multiprocessing
import yaml
from scipy import interpolate
import os
from sn_tools.sn_calcFast import LCfast, covColor
from sn_tools.sn_telescope import Telescope
from astropy.table import Table, vstack, Column
import time
import pandas as pd
from scipy.interpolate import interp1d
from sn_tools.sn_rate import SN_Rate
from scipy.interpolate import RegularGridInterpolator


class SNNSNMetric(BaseMetric):
    """
    Measure zlim of type Ia supernovae.

    Parameters
    --------------
    lc_reference : dict
       SN reference data (LC)(key: (x1,color); vals: sn_tools.sn_utils.GetReference
    metricName : str, opt
      metric name (default : SNSNRMetric)
    mjdCol : str, opt
      mjd column name (default : observationStartMJD)
    RaCol : str,opt
      Right Ascension column name (default : fieldRA)
    DecCol : str,opt
      Declinaison column name (default : fieldDec)
    filterCol : str,opt
       filter column name (default: filter)
    m5Col : str, opt
       five-sigma depth column name (default : fiveSigmaDepth)
    exptimeCol : str,opt
       exposure time column name (default : visitExposureTime)
    nightCol : str,opt
       night column name (default : night)
    obsidCol : str,opt
      observation id column name (default : observationId)
    nexpCol : str,opt
      number of exposure column name (default : numExposures)
     vistimeCol : str,opt
        visit time column name (default : visitTime)
    season : list,opt
       list of seasons to process (float)(default: -1 = all seasons)
    coadd : bool,opt
       coaddition per night (and per band) (default : True)
    zmin : float,opt
       min redshift for the study (default: 0.0)
    zmax : float,opt
       max redshift for the study (default: 1.2)
    pixArea: float, opt
       pixel area (default: 9.6)
    outputType: str, opt
      output type requested (defauls: zlims)
    verbose: bool,opt
      verbose mode (default: False)
    ploteffi: bool, opt
      display efficiencies during processing (default:False)
    proxy_level: int, opt
     proxy level for the processing (default: 0)
    N_bef: int, opt
      number of LC points LC before T0 (default:5)
    N_aft: int, opt
      number of LC points after T0 (default: 10)
     snr_min: float, opt
       minimal SNR of LC points (default: 5.0)
     N_phase_min: int, opt
       number of LC points with phase<= -5(default:1)
    N_phase_max: int, opt
      number of LC points with phase>= 20 (default: 1)
    x1_color_dist: ,opt
     (x1,color) distribution (default: None)
    lightOutput: bool, opt
      output level of information (light or more) (default:True)
    T0s: str,opt
       T0 values for the processing (default: all)

    """

    def __init__(self, lc_reference,
                 metricName='SNNSNMetric',
                 mjdCol='observationStartMJD', RaCol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', season=-1, coadd=True, zmin=0.0, zmax=1.2,
                 pixArea=9.6, outputType='zlims', verbose=False, ploteffi=False, proxy_level=0,
                 N_bef=5, N_aft=10, snr_min=5., N_phase_min=1, N_phase_max=1,
                 x1_color_dist=None, lightOutput=True, T0s='all', **kwargs):

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
        self.pixArea = pixArea
        self.ploteffi = ploteffi
        self.x1_color_dist = x1_color_dist
        self.T0s = T0s

        cols = [self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol]

        self.stacker = None
        if coadd:
            cols += ['coadd']
            self.stacker = CoaddStacker(mjdCol=self.mjdCol, RaCol=self.RaCol, DecCol=self.DecCol, m5Col=self.m5Col, nightCol=self.nightCol,
                                        filterCol=self.filterCol, numExposuresCol=self.nexpCol, visitTimeCol=self.vistimeCol, visitExposureTimeCol='visitExposureTime')
        super(SNNSNMetric, self).__init__(
            col=cols, metricDtype='object', metricName=metricName, **kwargs)

        self.season = season

        telescope = Telescope(airmass=1.2)

        # LC selection parameters
        self.N_bef = N_bef  # nb points before peak
        self.N_aft = N_aft  # nb points after peak
        self.snr_min = snr_min  # SNR cut for points before/after peak
        self.N_phase_min = N_phase_min  # nb of point with phase <=-5
        self.N_phase_max = N_phase_max  # nb of points with phase >=20

        self.lcFast = {}

        # loading reference LC files
        for key, vals in lc_reference.items():
            self.lcFast[key] = LCfast(vals, key[0], key[1], telescope,
                                      self.mjdCol, self.RaCol, self.DecCol,
                                      self.filterCol, self.exptimeCol,
                                      self.m5Col, self.seasonCol,
                                      self.snr_min, lightOutput=lightOutput)

        # loading parameters
        self.zmin = zmin #zmin for the study
        self.zmax = zmax # zmax for the study
        self.zStep = 0.05  # zstep
        self.daymaxStep = 1.  # daymax step
        self.min_rf_phase = -20. # min ref phase for LC points selection
        self.max_rf_phase = 40. # max ref phase for LC points selection

        self.min_rf_phase_qual = -15. # min ref phase for bounds effects
        self.max_rf_phase_qual = 25. # max ref phase for bounds effects

        # snrate
        self.rateSN = SN_Rate(
            min_rf_phase=self.min_rf_phase_qual, max_rf_phase=self.max_rf_phase_qual)

        # verbose mode - useful for debug and code performance estimation
        self.verbose = verbose
        # self.verbose = True
        
        # status of the pixel after processing
        self.status = dict(
            zip(['ok', 'effi', 'season_length', 'nosn', 'simu_parameters'], [1, -1, -2, -3, -4]))

        # supernovae parameters
        self.params = ['x0', 'x1', 'daymax', 'color']

        # output type and proxy level
        self.outputType = outputType  # this is to choose the output: lc, sn, effi, zlims
        self.proxy_level = proxy_level  # proxy level chosen by the user: 0, 1, 2

    def run(self, dataSlice,  slicePoint=None):
        """
        Run method of the metric

        Parameters
        --------------
        dataSlice: numpy array
          data to process (scheduler simulations)
        slicePoint: bool, opt
          (default: None)

        Returns
        ----------
        Depend on self.outputType:
          self.outputType == 'zlims':  numpy array 
      
          self.outputType == 'sn':  numpy array 
          
          self.outputType == 'lc':  numpy array
          
          self.outputType == 'effi':  numpy array
        """

        # time 0 for performance estimation purpose
        time_ref = time.time()

        # get the seasons
        seasons = self.season

        # if seasons = -1: process the seasons seen in data
        if self.season == -1:
            seasons = np.unique(dataSlice[self.seasonCol])

        #seasons = [[1,2,3],[4,5,6],[7,8,9,10]]
        #seasons = [[i] for i in range(1,11)]

        #effi_totdf = pd.DataFrame()
        #zlim_nsndf = pd.DataFrame()

        # get redshift range for processing
        zRange = list(np.arange(self.zmin, self.zmax, self.zStep))
        if zRange[0] < 1.e-6:
            zRange[0] = 0.01

        self.zRange = zRange
        
        #season infos
        dfa = pd.DataFrame(np.copy(dataSlice))
        season_info = dfa.groupby(['season']).apply(lambda x : self.seasonInfo(x)).reset_index()
        
        # select seasons of at least 30 days
        idx = season_info['season_length'] >= 30
        season_info = season_info[idx]
        # get season length depending on the redshift
        dur_z = season_info.groupby(['season']).apply(lambda x: self.duration_z(x)).reset_index()
        if dur_z.empty:
            return None
        # get simulation parameters
        gen_par = dur_z.groupby(['z','season']).apply(lambda x: self.calcDaymax(x)).reset_index()

        # prepare pandas DataFrames for output 
        vara_totdf = pd.DataFrame()
        varb_totdf = pd.DataFrame()
        
        # loop on seasons
        for seas in seasons:
            """
            effi_seasondf, zlimsdf = self.run_seasons(dataSlice,[seas])
            #print(seas,effi_seasondf)
            zlim_nsndf = pd.concat([zlim_nsndf, zlimsdf], sort=False)
            effi_totdf = pd.concat([effi_totdf, effi_seasondf], sort=False)
            """
            # get seasons processing
            vara_df, varb_df = self.run_seasons(dataSlice, [seas],gen_par,dur_z)
            # print(seas,effi_seasondf)
            vara_totdf = pd.concat([vara_totdf, vara_df], sort=False)
            varb_totdf = pd.concat([varb_totdf, varb_df], sort=False)

        # estimate time of processing
        if self.verbose:
            print('finally - eop', time.time()-time_ref)
            print(varb_totdf)

        # return the output as chosen by the user (outputType)
        if self.outputType == 'lc':
            return varb_totdf.to_records()

        if self.outputType == 'sn':
            return vara_totdf.to_records()

        if self.outputType == 'effi':
            return vara_totdf.to_records()

        return varb_totdf.to_records()

    def run_seasons(self, dataSlice, seasons, gen_par,dura_z):
        """
        Method to run on seasons

        Parameters
        --------------
        dataSlice: numpy array, opt
          data to process (scheduler simulations)
        seasons: list(int)
          list of seasons to process

        Returns
        ---------
        effi_seasondf: pandas df
          efficiency curves
        zlimsdf: pandas df
          redshift limits and number of supernovae
        """

        time_ref = time.time()

        # get pixel id
        pixRa = np.unique(dataSlice['pixRa'])[0]
        pixDec = np.unique(dataSlice['pixDec'])[0]
        healpixID = int(np.unique(dataSlice['healpixID'])[0])

        if self.verbose:
            print('#### Processing season', seasons, healpixID)
            
        groupnames = ['pixRa', 'pixDec', 'healpixID', 'season', 'x1', 'color']

        gen_p = gen_par[gen_par['season'].isin(seasons)]
        if gen_p.empty:
            if self.verbose:
                print('No generator parameter found')
            return None, None
        dur_z = dura_z[dura_z['season'].isin(seasons)]
        obs = pd.DataFrame(np.copy(dataSlice))
        obs = obs[obs['season'].isin(seasons)]
   
            
        if self.verbose:
            time_refb = time.time()

        # coaddition per night and per band (if requested by the user)
        if self.stacker is not None:
            obs = self.stacker._run(obs.to_records(index=False))
            if self.verbose:
                print('after stacking', seasons, time.time()-time_refb)

        # simulate supernovae and lc
        if self.verbose:
            print('simulation SN', len(gen_p))
            time_refb = time.time()

        sn, lc = self.gen_LC_SN(obs, gen_p.to_records(index=False))

        if self.outputType == 'lc' or self.outputType == 'sn':
            return sn, lc
        if self.verbose:
            print('after simulation', seasons, time.time()-time_refb)

        if sn.empty:
            # no LC could be simulated -> fill output with errors
            zlimsdf = self.errordf(
                pixRa, pixDec, healpixID, season, self.status['nosn'])
            effi_seasondf = self.erroreffi(
                pixRa, pixDec, healpixID, season)
        else:
            # LC could be simulated -> estimate efficiencies
            if self.verbose:
                print('estimating efficiencies')
                time_refb = time.time()
            # estimate efficiencies
            effi_seasondf = self.effidf(sn)

            if self.verbose:
                print('after effis', time.time()-time_refb)
                print('estimating zlimits')
                time_refb = time.time()

            # estimate zlims
            zlimsdf = effi_seasondf.groupby(groupnames).apply(
                lambda x: self.zlimdf(x, dur_z)).reset_index(level=list(range(len(groupnames))))

            if self.verbose:
                print('zlims', zlimsdf)
                print('after zlims', time.time()-time_refb)
                print('Estimating number of supernovae')
                time_refb = time.time()
            # estimate number of supernovae - medium one

            zlimsdf['nsn_med'] = zlimsdf.apply(lambda x: self.nsn_typedf(
                x, 0.0, 0.0, effi_seasondf, dur_z), axis=1)

            if self.verbose:
                print('Number of supernovae')
                print(zlimsdf)

            if self.proxy_level >0:
                zlimsdf['nsn'] = -1
                return effi_seasondf, zlimsdf

            # estimate number of supernovae - total - proxy_level = 0

            zlimsdf['nsn'] = self.nsn_tot(effi_seasondf, zlimsdf, dur_z)

            if self.verbose:
                print('final result ', zlimsdf)
                print('after nsn', time.time()-time_refb)

        if self.verbose:
            print('#### SEASON processed', time.time()-time_ref,
                  seasons)

        return effi_seasondf, zlimsdf

    def duration_z(self, grp):
        """
        Method to estimate the season length vs redshift
        This is necessary to take into account boundary effects
        when estimating the number of SN that can be detected

        daymin, daymax = min and max MJD of a season
        T0_min(z) =  daymin-(1+z)*min_rf_phase_qual
        T0_max(z) =  daymax-(1+z)*max_rf_phase_qual
        season_length(z) = T0_max(z)-T0_min(z) 

        Parameters
        --------------
        grp: pandas df group
          data to process: season infos

        Returns
        ----------
        pandas df with season_length, z, T0_min and T0_max cols

        """
        
        daymin = grp['MJD_min'].values
        daymax = grp['MJD_max'].values
        dur_z = pd.DataFrame(self.zRange,columns=['z'])
        dur_z['T0_min'] = daymin-(1.+dur_z['z'])*self.min_rf_phase_qual
        dur_z['T0_max'] = daymax-(1.+dur_z['z'])*self.max_rf_phase_qual
        dur_z['season_length'] = dur_z['T0_max']-dur_z['T0_min']

        return dur_z

    def calcDaymax(self, grp):
        """
        Method to estimate T0 (daymax) values for simulation.

        Parameters
        --------------
        grp: group (pandas df sense)
         group of data to process with the following cols:
           T0_min: T0 min value (per season)
           T0_max: T0 max value (per season)

        Returns
        ----------
        pandas df with daymax, min_rf_phase, max_rf_phase values

        """
        
        T0_max = grp['T0_max'].values
        T0_min= grp['T0_min'].values
        num = (T0_max-T0_min)/self.daymaxStep
        if T0_max-T0_min > 10:
            df = pd.DataFrame(np.linspace(T0_min,T0_max,int(num)),columns=['daymax'])
        else:
            df = pd.DataFrame([-1],columns=['daymax'])
            
        df['min_rf_phase'] =self.min_rf_phase_qual
        df['max_rf_phase'] = self.max_rf_phase_qual
        
        return df
        
    def errordf(self, pixRa, pixDec, healpixID, season, errortype):
        """
        Method to return error df related to zlims values
        
        Parameters
        --------------
        pixRa: float
          pixel Ra
        pixDec: float
          pixel Dec
        healpixID: int
          healpix ID
        season: int
          season 
        errortype: str
          type of error

        """
        
        return pd.DataFrame({'pixRa': [np.round(pixRa, 4)],
                             'pixDec': [np.round(pixDec, 4)],
                             'healpixID': [healpixID],
                             'season': [int(season)],
                             'x1': [-1.0],
                             'color': [-1.0],
                             'zlim': [-1.0],
                             'nsn_med': [-1.0],
                             'nsn': [-1.0],
                             'status': [int(errortype)]})

    def erroreffi(self, pixRa, pixDec, healpixID, season):
        """
        Method to return error df related to efficiencies
        
        Parameters
        --------------
        pixRa: float
          pixel Ra
        pixDec: float
          pixel Dec
        healpixID: int
          healpix ID
        season: int
          season 
        errortype: str
          type of error

        """
        return pd.DataFrame({'pixRa': [np.round(pixRa)],
                             'pixDec': [np.round(pixDec)],
                             'healpixID': [healpixID],
                             'season': [int(season)],
                             'x1': [-1.0],
                             'color': [-1.0],
                             'z': [-1.0],
                             'effi': [-1.0],
                             'effi_err': [-1.0]})

    def effidf(self, sn_tot,color_cut=0.04):
        """
        Method estimating efficiency vs z for a sigma_color cut

        Parameters
        ---------------
        sn_tot: 

        Returns
        ----------

        """

        
        sndf = pd.DataFrame(sn_tot)

        listNames = ['season', 'pixRa', 'pixDec', 'healpixID', 'x1', 'color']
        groups = sndf.groupby(listNames)

        effi = groups['Cov_colorcolor', 'z'].apply(
            lambda x: self.effiObsdf(x,color_cut)).reset_index(level=list(range(len(listNames))))


        # this is to plot efficiencies and also sigma_color vs z
        if self.ploteffi:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots()
            figb, axb = plt.subplots()

            self.plot(ax,effi,'effi','effi_err','Observing Efficiencies')
            self.plot(axb,sndf,'Cov_colorcolor',None,'$\sigma_{color}^2$')
            # get efficiencies vs z
           
            plt.show()

        return effi

    def plot(self, ax, effi,vary,erry=None,legy=''):
        """
        Simple method to plot vs z

        Parameters
        --------------
        ax: 
          axis where to plot
        effi: pandas df
          data to plot
        vary: str
          variable (column of effi) to plot
        erry: str, opt
          error on y-axis (default: None)
        legy: str, opt
          y-axis legend (default: '')

        """
        grb = effi.groupby(['x1', 'color'])
        yerr = None
        for key, grp in grb:
            x1 = grp['x1'].unique()[0]
            color = grp['color'].unique()[0]
            if erry is not None:
                yerr=grp[erry]
            ax.errorbar(grp['z'], grp[vary], yerr=yerr,
                        marker='o', label='(x1,color)=({},{})'.format(x1, color),lineStyle='None')
            
        ftsize = 15
        ax.set_xlabel('z', fontsize=ftsize)
        ax.set_ylabel(legy, fontsize=ftsize)
        ax.xaxis.set_tick_params(labelsize=ftsize)
        ax.yaxis.set_tick_params(labelsize=ftsize)
        ax.legend(fontsize=ftsize)
        
    
    def zlimdf(self, grp, duration_z):

        zlimit = 0.0
        status = self.status['effi']

        # z range for the study
        zplot = list(np.arange(self.zmin, self.zmax, 0.001))

        #print(grp['z'], grp['effi'])

        if len(grp['z']) <= 3:
            return pd.DataFrame({'zlim': [zlimit],
                                 'status': [int(status)]})
        # interpolate efficiencies vs z
        effiInterp = interp1d(
            grp['z'], grp['effi'], kind='linear', bounds_error=False, fill_value=0.)

        # get rate
        season = np.median(grp['season'])
        idx = duration_z['season'] == season
        seas_duration_z = duration_z[idx]

        durinterp_z = interp1d(
            seas_duration_z['z'], seas_duration_z['season_length'], bounds_error=False, fill_value=0.)

        zz, rate, err_rate, nsn, err_nsn = self.rateSN(zmin=self.zmin,
                                                       zmax=self.zmax,
                                                       duration_z=durinterp_z,
                                                       survey_area=self.pixArea)
        rateInterp = interp1d(zz, nsn, kind='linear',
                              bounds_error=False, fill_value=0)

        # estimate the cumulated number of SN vs z
        nsn_cum = np.cumsum(effiInterp(zplot)*rateInterp(zplot))

        if nsn_cum[-1] >= 1.e-5:
            nsn_cum_norm = nsn_cum/nsn_cum[-1]  # normalize
            zlim = interp1d(nsn_cum_norm, zplot)
            zlimit = np.asscalar(zlim(0.95))
            status = self.status['ok']

            if self.ploteffi:
                import matplotlib.pylab as plt
                fig, ax = plt.subplots()
                x1 = grp['x1'].unique()[0]
                color = grp['color'].unique()[0]

                ax.plot(zplot, nsn_cum_norm,
                        label='(x1,color)=({},{})'.format(x1, color))

                ftsize = 15
                ax.set_ylabel('NSN ($z<$)', fontsize=ftsize)
                ax.set_xlabel('z', fontsize=ftsize)
                ax.xaxis.set_tick_params(labelsize=ftsize)
                ax.yaxis.set_tick_params(labelsize=ftsize)
                ax.set_xlim((0.0, 1.2))
                ax.set_ylim((0.0, 1.05))
                ax.plot([0., 1.2], [0.95, 0.95], ls='--', color='k')
                plt.legend(fontsize=ftsize)
                plt.show()

        return pd.DataFrame({'zlim': [zlimit],
                             'status': [int(status)]})

    def nsn_typedf(self, grp, x1, color, effi_tot, duration_z, search=True):

        # get rate
        season = np.median(grp['season'])
        idx = duration_z['season'] == season
        seas_duration_z = duration_z[idx]

        durinterp_z = interp1d(
            seas_duration_z['z'], seas_duration_z['season_length'], bounds_error=False, fill_value=0.)

        """
        zz, rate, err_rate, nsn, err_nsn = self.rateSN(zmin=self.zmin,
                                                       zmax=self.zmax,
                                                       duration_z=durinterp_z,
                                                       survey_area=self.pixArea)
        rateInterp = interp1d(zz, nsn, kind='linear',
                              bounds_error=False, fill_value=0)
        """
        if search:
            effisel = effi_tot.loc[lambda dfa: (
                dfa['x1'] == x1) & (dfa['color'] == color), :]
        else:
            effisel = effi_tot

        nsn = self.nsn(effisel, grp['zlim'], durinterp_z)

        return nsn

    def nsn_typedf_new(self, grp, duration_z, zlims):

        # get rate

        # durinterp_z = interp1d(
        #    duration_z['z'], duration_z['season_length'], bounds_error=False, fill_value=0.)
        """
        nsn = []
        for zlim in zlims:
            nsn.append(self.nsn(grp, zlim, durinterp_z))
        """
        nsn = zlims.apply(lambda x: self.nsn_typedf(
            x, 0., 0., grp, duration_z, search=False), axis=1)
        return nsn*np.mean(grp['weight'])

    def nsn_tot(self, effi, zlim, duration_z):

        # the first thing is to interpolate efficiencies to have a regular grid
        zvals = np.arange(0.0, 1.2, 0.05)

        idx = np.abs(effi['x1']) < 1.e-5
        idx &= np.abs(effi['color']) < 1.e-5
        if len(effi[idx]['z']) < 3 or np.mean(effi[idx]['effi']) < 1.e-5:
            return -1.0
        effi_grp = effi.groupby(['x1', 'color'])['x1', 'color', 'effi', 'effi_err', 'z'].apply(
            lambda x: self.effi_interp(x, zvals)).reset_index().to_records(index=False)

        # print(zlim)

        # Now construct the griddata

        """
        x1_vals = effi_grp['x1'].unique()
        color_vals = effi_grp['color'].unique()
        z_vals = effi_grp['z'].unique()
        """
        x1_vals = np.unique(effi_grp['x1'])
        color_vals = np.unique(effi_grp['color'])
        z_vals = np.unique(effi_grp['z'])

        n_x1 = len(x1_vals)
        n_color = len(color_vals)
        n_z = len(z_vals)

        index = np.lexsort((effi_grp['x1'], effi_grp['color'], effi_grp['z']))
        effi_resh = np.reshape(effi_grp[index]['effi'], (n_x1, n_color, n_z))
        effi_err_resh = np.reshape(
            effi_grp[index]['effi_err'], (n_x1, n_color, n_z))

        effi_grid = RegularGridInterpolator(
            (x1_vals, color_vals, z_vals), effi_resh, method='linear', bounds_error=False, fill_value=0.)

        effi_err_grid = RegularGridInterpolator(
            (x1_vals, color_vals, z_vals), effi_err_resh, method='linear', bounds_error=False, fill_value=0.)

        nsnTot = None
        ip = -1
        weight_sum = 0.

        idx = np.abs(self.x1_color_dist['x1']) <= 2
        idx &= np.abs(self.x1_color_dist['color']) <= 0.2

        time_ref = time.time()
        x1_tile = np.repeat(self.x1_color_dist[idx]['x1'], len(zvals))
        color_tile = np.repeat(self.x1_color_dist[idx]['color'], len(zvals))
        z_tile = np.tile(zvals, len(self.x1_color_dist[idx]))
        weight_tile = np.repeat(
            self.x1_color_dist[idx]['weight_tot'], len(zvals))

        df_test = pd.DataFrame()

        df_test.loc[:, 'effi'] = effi_grid((x1_tile, color_tile, z_tile))
        df_test.loc[:, 'effi_err'] = effi_err_grid(
            (x1_tile, color_tile, z_tile))
        df_test.loc[:, 'x1'] = np.round(x1_tile, 2)
        df_test.loc[:, 'color'] = np.round(color_tile, 2)
        df_test.loc[:, 'z'] = z_tile
        df_test.loc[:, 'weight'] = np.round(weight_tile, 2)
        season = np.median(zlim['season'])
        idxb = duration_z['season'] == season

        #print('there man',zlim['zlim'],duration_z[idxb])
        nsn_tot = df_test.groupby(['x1', 'color']).apply(lambda x: self.nsn_typedf_new(
            x, duration_z[idxb], zlim))

        # print(nsn_tot.sum(axis=0),time.time()-time_ref,type(nsn_tot))
        return nsn_tot.sum(axis=0)
        """
        print(x1_tile)
        print(color_tile)
        print(z_tile)
        print(effi_grid((x1_tile,color_tile,z_tile)))
        #print(test)
        """
        time_ref = time.time()

        for vals in self.x1_color_dist[idx]:

            x1 = vals['x1']
            color = vals['color']
            weight = vals['weight_tot']
            # if np.abs(x1)<=2 and np.abs(color)<=0.2:
            ip += 1
            weight_sum += weight
            # print('trying',x1,color,weight)
            df_x1c = pd.DataFrame()
            df_x1c.loc[:, 'effi'] = effi_grid(
                ([x1]*len(zvals), [color]*len(zvals), zvals))
            df_x1c.loc[:, 'effi_err'] = effi_err_grid(
                ([x1]*len(zvals), [color]*len(zvals), zvals))
            df_x1c.loc[:, 'x1'] = x1
            df_x1c.loc[:, 'color'] = color
            df_x1c.loc[:, 'z'] = zvals

            """
            print('here man',x1,color)
            import matplotlib.pylab as plt
            plt.plot(df_x1c['z'],df_x1c['effi'],'ko',mfc='None')
            sel = df_test.loc[lambda df: (np.abs(df.x1-x1)<1.e-5)&(np.abs(df.color-color)<1.e-5), :]
            plt.plot(sel['z'],sel['effi'],'r*')

            plt.show()
            """
            #nsn = self.nsn_typedf(zlim,x1,color,df_x1c,duration_z)

            nsn = zlim.apply(lambda x: self.nsn_typedf(
                x, x1, color, df_x1c, duration_z, search=False), axis=1)

            if nsnTot is None:
                nsnTot = nsn*weight
            else:
                nsnTot = np.sum([nsnTot, nsn*weight], axis=0)
        print('after calc', nsnTot, time.time()-time_ref)
        print(test)
        return nsnTot

        """
        import matplotlib.pylab as plt
        plt.plot(effi_grp['z'],effi_grp['effi'],'ko',mfc='None')
        plt.plot(effi['z'],effi['effi'],'r*',mfc='None')
        plt.show()
        """

    def effi_interp(self, grp, zvals):

        interp = interp1d(grp['z'], grp['effi'],
                          bounds_error=False, fill_value=0.)
        interp_err = interp1d(grp['z'], grp['effi_err'],
                              bounds_error=False, fill_value=0.)

        """
        resdf = pd.DataFrame()

        resdf.loc[:,'z'] = zvals
        resdf.loc[:,'effi'] = interp(zvals)
        resdf.loc[:,'effi_err'] = interp_err(zvals)
        resdf.loc[:'x1'] = grp['x1'].unique()
        resdf.loc[:'color'] = grp['color'].unique()
        """
        """
        return pd.DataFrame({'x1': [grp['x1'].unique()]*len(zvals),
                             'color': [grp['color'].unique()]*len(zvals),
                             'effi': interp(zvals),
                             'effi_err': interp_err(zvals),
                             'z': zvals})
        """
        return pd.DataFrame({'effi': interp(zvals),
                             'effi_err': interp_err(zvals),
                             'z': zvals})

        # return resdf

    """
    def nsndf(self, effi, zlim, rateInterp):

        # zrange for interpolation
        zplot = list(np.arange(self.zmin, self.zmax, 0.001))
        # interpolate efficiencies vs z
        effiInterp = interp1d(
            effi['z'], effi['effi'], kind='linear', bounds_error=False, fill_value=0.)
        # estimate the cumulated number of SN vs z
        nsn_cum = np.cumsum(effiInterp(zplot)*rateInterp(zplot))
        nsn_interp = interp1d(zplot, nsn_cum)

        return nsn_interp(zlim)
    """

    def nsn(self, effi, zlim, duration_z):

        if zlim < 1.e-3:
            return -1.0

        dz = 0.001
        zplot = list(np.arange(self.zmin, self.zmax, dz))
        # interpolate efficiencies vs z
        effiInterp = interp1d(
            effi['z'], effi['effi'], kind='linear', bounds_error=False, fill_value=0.)
        # estimate the cumulated number of SN vs z
        zz, rate, err_rate, nsn, err_nsn = self.rateSN(zmin=self.zmin,
                                                       zmax=self.zmax,
                                                       dz=dz,
                                                       duration_z=duration_z,
                                                       survey_area=self.pixArea)

        nsn_cum = np.cumsum(effiInterp(zplot)*nsn)
        #nsn_cum = np.cumsum(rateInterp(zplot))*dz
        nsn_interp = interp1d(zplot, nsn_cum)

        """
        import matplotlib.pyplot as plt

        plt.plot(zplot,nsn_cum)
        plt.show()
        """
        return np.asscalar(nsn_interp(zlim))

        # print([-1.]*len(zlim),type(zlim))
        #nsn_res = np.zeros(shape=(len(zlim),))
        nsn_res = pd.DataFrame({'test', tuple([-1.]*len(zlim))})

        idxa = zlim >= 1.e-3
        idxb = np.argwhere(zlim < 1.e-3)
        """
        zlim_bad = zlim[idx] 
        zlim_good = zlim[~idx]
        """

        zplot = list(np.arange(self.zmin, self.zmax, 0.001))
        # interpolate efficiencies vs z
        effiInterp = interp1d(
            effi['z'], effi['effi'], kind='linear', bounds_error=False, fill_value=0.)
        # estimate the cumulated number of SN vs z
        nsn_cum = np.cumsum(effiInterp(zplot)*rateInterp(zplot))
        nsn_interp = interp1d(zplot, nsn_cum)

        test = nsn_interp(zlim[idxa])
        print(idxa, zlim[idxa], test.shape, nsn_res.shape)
        print(idxb)
        nsn_res.iloc[idxa] = nsn_interp(zlim[idxa])
        #nsn_res[idxb] = -1.

        print(nsn_res)
        return nsn_res

    def seasonInfo(self,grp):
        """
        Method to estimate seasonal info (cadence, season length, ...)

        Parameters
        --------------
        grp: pandas df group

        Returns
        ---------
        pandas df with the cfollowing cols:
        
        """
        df = pd.DataFrame([len(grp)],columns=['Nvisits'])
        df['MJD_min'] = grp[self.mjdCol].min()
        df['MJD_max'] =grp[self.mjdCol].max()
        df['season_length'] = df['MJD_max']-df['MJD_min']
        df['cadence'] = 0.
        if len(grp) > 5:
            df['cadence'] = grp[self.mjdCol].diff().mean()
        
        return df
        
    
    def seasonInfo_old(self, dataSlice, seasons):
        """
        Get info on seasons for each dataSlice
        Parameters
        -----
        dataSlice : array
          array of observations
        Returns
        -----
        recordarray with the following fields:
        season, cadence, season_length, MJDmin, MJDmax
        """

        rv = []
        for season in seasons:
            idx = (dataSlice[self.seasonCol] == season)
            slice_sel = dataSlice[idx]
            slice_sel.sort(order=self.mjdCol)
            mjds_season = slice_sel[self.mjdCol]
            mjd_min = np.min(mjds_season)
            mjd_max = np.max(mjds_season)
            Nvisits = len(slice_sel)
            season_length = mjd_max-mjd_min
            if len(slice_sel) < 5:
                cadence = 0
            else:
                cadence = np.mean(mjds_season[1:]-mjds_season[:-1])
            rv.append((season, cadence, season_length,
                       mjd_min, mjd_max, Nvisits))

        info_season = None
        if len(rv) > 0:
            info_season = np.rec.fromrecords(
                rv, names=['season', 'cadence', 'season_length', 'MJD_min', 'MJD_max', 'Nvisits'])

        return info_season

    def process(self, tab):

        if self.verbose:
            print('processing', len(tab))
            time_refp = time.time()

        nproc = 1
        #groups = tab.group_by(['season', 'z', 'pixRa', 'pixDec', 'healpixID'])
        #groups = tab.group_by(['season', 'z', 'healpixID'])
        #groups = tab.group_by(['healpixID'])
        #re = tab.apply(lambda grp : CalcSN_df_def(grp,snr_min=self.snr_min))

        """
        # Add columns requested for SN selection

        tab.loc[:, 'N_aft'] = (np.sign(tab['phase']) == 1) & (tab['snr_m5'] >= self.snr_min)
        tab.loc[:, 'N_bef'] = (np.sign(tab['phase']) == -1) & (tab['snr_m5'] >= self.snr_min)
   
        tab.loc[:, 'N_phmin'] = (tab['phase'] <= -5.)
        tab.loc[:, 'N_phmax'] = (tab['phase'] >= 20)
    
        # transform boolean to int because of some problems in the sum()
    
        for colname in ['N_aft', 'N_bef', 'N_phmin', 'N_phmax']:
            tab[colname] = tab[colname].astype(int)
        """
        # now groupby
        tab = tab.round({'pixRa': 4, 'pixDec': 4, 'daymax': 3,
                         'z': 3, 'x1': 2, 'color': 2})
        groups = tab.groupby(
            ['pixRa', 'pixDec', 'daymax', 'season', 'z', 'healpixID', 'x1', 'color'])
        if self.verbose:
            print('groups', groups.ngroups)

        if nproc == 1:
            # restab = CalcSN_df(groups.groups[:], N_bef=self.N_bef, N_aft=self.N_aft,
            #                   snr_min=self.snr_min, N_phase_min=self.N_phase_min, N_phase_max=self.N_phase_max).sn
            #restab = groups.apply(lambda grp : CalcSN_df_def(grp,snr_min=self.snr_min)).reset_index()

            tosum = []
            for ia, vala in enumerate(self.params):
                for jb, valb in enumerate(self.params):
                    if jb >= ia:
                        tosum.append('F_'+vala+valb)
            tosum += ['N_aft', 'N_bef', 'N_phmin', 'N_phmax']
            # apply the sum on the group
            # print(groups[tosum].sum())
            sums = groups[tosum].sum().reset_index()

            if self.verbose:
                print('sums', time.time()-time_refp)
            """
            sums.loc[:,'cov_ColorColor'] = covColor(sums)

            print(sums)
            """
            # select LC according to the number of points bef/aft peak
            idx = sums['N_aft'] >= self.N_aft
            idx &= sums['N_bef'] >= self.N_bef
            idx &= sums['N_phmin'] >= self.N_phase_min
            idx &= sums['N_phmax'] >= self.N_phase_max

            if self.verbose:
                print('after selection', time.time()-time_refp)

            finalsn = pd.DataFrame()
            goodsn = pd.DataFrame(sums.loc[idx])
            if len(goodsn) > 0:
                goodsn.loc[:, 'Cov_colorcolor'] = covColor(goodsn)
                finalsn = pd.concat([finalsn, goodsn], sort=False)
            if self.verbose:
                print('after color', time.time()-time_refp)

            badsn = pd.DataFrame(sums.loc[~idx])

            if len(badsn) > 0:
                badsn.loc[:, 'Cov_colorcolor'] = 100.
                finalsn = pd.concat([finalsn, badsn], sort=False)

            if self.verbose:
                print('end of processing', time.time()-time_refp)
            return finalsn

        indices = groups.groups.indices
        ngroups = len(indices)-1
        delta = ngroups
        if nproc > 1:
            delta = int(delta/(nproc))

        if self.verbose:
            print('multiprocessing delta', delta, ngroups)
        batch = range(0, ngroups, delta)

        if ngroups not in batch:
            batch = np.append(batch, ngroups)

        batch = batch.tolist()
        if batch[-1]-batch[-2] <= 2:
            batch.remove(batch[-2])

        if self.verbose:
            print('multiprocessing batch', batch)
        result_queue = multiprocessing.Queue()
        restot = Table()
        restot = None
        for j in range(len(batch)-1):
            # for j in range(9,10):
            ida = batch[j]
            idb = batch[j+1]
            p = multiprocessing.Process(name='Subprocess-'+str(j), target=self.lcLoop, args=(
                groups, batch[j], batch[j+1], j, result_queue))
            p.start()

        resultdict = {}
        for i in range(len(batch)-1):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        for key, vals in resultdict.items():
            if restot is None:
                restot = vals
            else:
                restot = np.concatenate((restot, vals))

        return restot
        # save output in npy file
        # np.save('{}/SN_{}.npy'.format(self.outputDir,self.procId),restot)

    def lcLoop(self, group, ida, idb, j=0, output_q=None):

        # resfi = Table()
        resfi = CalcSN_df(group.groups[ida:idb]).sn
        """
        for ind in range(ida,idb,1):
            grp = group.groups[ind]
            res = calcSN(grp)
            resfi = vstack([resfi,res])
        """
        # resfi = calcSN(group)

        if output_q is not None:
            return output_q.put({j: resfi})
        else:
            return resfi

    def effiObsdf(self, data, color_cut=0.04):
        """
        Method to estimate observing efficiencies for supernovae

        Parameters
        --------------
        data: pandas df - grp
          data to process

        Returns
        ----------
        pandas df with the following cols:
          - cols used to make the group
          - effi, effi_err: observing efficiency and associated error

        """

        # reference df to estimate efficiencies
        df = data.loc[lambda dfa:  np.sqrt(dfa['Cov_colorcolor']) < 100000., :]

        # selection on sigma_c<= 0.04
        df_sel = df.loc[lambda dfa:  np.sqrt(dfa['Cov_colorcolor']) <= color_cut, :]

        # make groups (with z)
        group = df.groupby('z')
        group_sel = df_sel.groupby('z')
        
        # Take the ratio to get efficiencies
        rb = (group_sel.size()/group.size())
        err = np.sqrt(rb*(1.-rb)/group.size())

        rb = rb.array
        err = err.array

        rb[np.isnan(rb)] = 0.
        err[np.isnan(err)] = 0.

        return pd.DataFrame({group.keys: list(group.groups.keys()),
                             'effi': rb,
                             'effi_err': err})

    def gen_LC_SN(self, obs, gen_par):
        """
        Method to simulate LC and supernovae

        Parameters
        ---------------
        obs: numpy array
          array of observations (from scheduler)
        gen_par: numpy array
          array of parameters for simulation


        """
        
        time_ref = time.time()
        # LC estimation

        sn_tot = pd.DataFrame()
        lc_tot = pd.DataFrame()
        for key, vals in self.lcFast.items():
            time_refs = time.time()
            gen_par_cp = np.copy(gen_par)
            if key == (-2.0, 0.2):
                idx = gen_par_cp['z'] < 0.9
                gen_par_cp = gen_par_cp[idx]
            lc = vals(obs, -1, gen_par_cp, bands='grizy')
            if self.ploteffi:
                self.plotLC(lc)
            if self.outputType == 'lc':
                lc_tot = pd.concat([lc_tot, lc], sort=False)
            if self.verbose:
                print('End of simulation', key, time.time()-time_refs)

            # estimate SN
            
            
            sn = pd.DataFrame()
            if len(lc) > 0:
                #sn = self.process(Table.from_pandas(lc))
                sn = self.process(pd.DataFrame(lc))

            if self.verbose:
                print('End of supernova', time.time()-time_refs)

            """
            if sn is not None:
                if sn_tot is None:
                    sn_tot = sn
                else:
                    sn_tot = np.concatenate((sn_tot, sn))
            """
            if not sn.empty:
                sn_tot = pd.concat([sn_tot, pd.DataFrame(sn)], sort=False)

        if self.verbose:
            print('End of supernova - all', time.time()-time_ref)

        return sn_tot, lc_tot

    def plotLC(self, lc, zref=0.01):
        
        import matplotlib.pyplot as plt

        lc = lc.round({'daymax':6})
        sel = lc[np.abs(lc['z']-zref)<1.e-5]
        #sel = sel[sel['band']=='LSST::g']

        #print(sel['daymax'].unique())
        fig, ax = plt.subplots(ncols=2,nrows=3)
        pos = dict(zip('ugrizy',[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]))

        fig.suptitle('(x1,color)=({},{})'.format(sel['x1'].unique(),sel['color'].unique()))
        for band in sel['band'].unique():
            idxb = sel['band'] == band
            selb = sel[idxb]
            ix = pos[band.split(':')[-1]][0]
            iy = pos[band.split(':')[-1]][1]
            for daymax in selb['daymax'].unique():
                selc = selb[selb['daymax']==daymax]
                ax[ix][iy].plot(selc['time'],selc['flux_e_sec'])
            
        plt.show()
        

    
# oldies


""""
      def effi(self, sn_tot):

        effi_tot = Table()
        x1_color = np.unique(sn_tot[['x1', 'color']])

        if self.ploteffi:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots()

        # get efficiencies vs z
        for key in x1_color:
            idx = np.abs(sn_tot['x1']-key[0]) < 1.e-5
            idx &= np.abs(sn_tot['color']-key[1]) < 1.e-5
            effi = self.effiObs(sn_tot[idx])
            effi_tot = vstack([effi_tot, effi])
            if self.ploteffi:
                ax.errorbar(effi['z'], effi['effi'], yerr=effi['effi_err'],
                            marker='o', label='(x1,color)=({},{})'.format(key[0], key[1]))

        if self.ploteffi:
            ftsize = 15
            ax.set_xlabel('z', fontsize=ftsize)
            ax.set_ylabel('Observation efficiency', fontsize=ftsize)
            ax.xaxis.set_tick_params(labelsize=ftsize)
            ax.yaxis.set_tick_params(labelsize=ftsize)
            plt.legend(fontsize=ftsize)
            plt.show()

        return effi_tot

  def zlim(self, effi_tot, rateInterp, season, zlims):

        x1_color = np.unique(effi_tot[['x1', 'color']])
        zplot = list(np.arange(self.zmin, self.zmax, 0.001))

        if self.ploteffi:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots()

        r = []
        names = []
        for key in x1_color:
            idx = np.abs(effi_tot['x1']-key[0]) < 1.e-5
            idx &= np.abs(effi_tot['color']-key[1]) < 1.e-5
            effi = effi_tot[idx]
            idb = np.abs(zlims['x1']-key[0]) < 1.e-5
            idb &= np.abs(zlims['color']-key[1]) < 1.e-5
            idarg = np.argwhere(idb)
            snName = self.nameSN[(np.round(key[0], 1), np.round(key[1], 1))]
            # interpolate efficiencies vs z
            effiInterp = interp1d(
                effi['z'], effi['effi'], kind='linear', bounds_error=False, fill_value=0.)

            # estimate the cumulated number of SN vs z
            nsn_cum = np.cumsum(effiInterp(zplot)*rateInterp(zplot))
            if nsn_cum[-1] <= 1.e-5:
                r.append(0.0)
                r.append(self.status['effi'])
                zlims['zlim_{}'.format(snName)] = 0.0
                zlims['status_{}'.format(snName)] = self.status['effi']
            else:
                nsn_cum_norm = nsn_cum/nsn_cum[-1]  # normalize

                if self.ploteffi:
                    ax.plot(zplot, nsn_cum_norm,
                            label='(x1,color)=({},{})'.format(key[0], key[1]))

                zlim = interp1d(nsn_cum_norm, zplot)
                # nsn = interp1d(zplot,nsn_cum)
                r.append(np.asscalar(zlim(0.95)))
                r.append(self.status['ok'])
                zlims['zlim_{}'.format(snName)] = np.asscalar(zlim(0.95))
                zlims['status_{}'.format(snName)] = self.status['ok']
            names.append('zlim_{}'.format(snName))
            names.append('status_{}'.format(snName))

        if self.verbose:
            print('zlims', r, names)

        res = Table(rows=[r], names=names)
        if self.ploteffi:
            ftsize = 15
            ax.set_ylabel('NSN ($z<$)', fontsize=ftsize)
            ax.set_xlabel('z', fontsize=ftsize)
            ax.xaxis.set_tick_params(labelsize=ftsize)
            ax.yaxis.set_tick_params(labelsize=ftsize)
            ax.set_xlim((0.0, 1.2))
            ax.set_ylim((0.0, 1.05))
            ax.plot([0., 1.2], [0.95, 0.95], ls='--', color='k')
            plt.legend(fontsize=ftsize)
            plt.show()
        return np.copy(zlims)

  def nsn_type(self, x1, color, effi_tot, zlims, rateInterp):

        x1 = 0.0
        color = 0.0

        idx = np.abs(effi_tot['x1']-x1) < 1.e-5
        idx &= np.abs(effi_tot['color']-color) < 1.e-5
        effi = effi_tot[idx]

        nsn = {}
        sn_types = ['faint', 'medium']

        # res = np.copy(zlims)

        res = Table(zlims)
        for typ in sn_types:
            zlim = zlims['zlim_{}'.format(typ)][0]
            if zlim <= 1.e-5:
                nsn = 0.
            else:
                nsn = self.nsn(effi, zlim, rateInterp)
            # res = rf.append_fields(res,'nsn_z{}'.format(typ),[nsn], usemask=False)
            res.add_column(Column([nsn], name='nsn_z{}'.format(typ)))

        return res

    def fillOutput(self, pixRa, pixDec, healpixID):
        rtot = [pixRa, pixDec, healpixID, 0., 0, 0., 0.]
        namestot = ['pixRa', 'pixDec', 'healpixID', 'zlim_faint',
                    'status_faint', 'zlim_medium', 'status_medium', 'x1', 'color', 'season']

        ro = []
        for key, val in self.lcFast.items():
            ro.append(rtot+[key[0], key[1]])
        return ro, namestot

    def effiObs(self, data):

        pixRa = np.unique(data['pixRa'])[0]
        pixDec = np.unique(data['pixDec'])[0]
        healpixID = int(np.unique(data['healpixID'])[0])
        season = np.unique(data['season'])[0]
        x1 = np.unique(data['x1'])[0]
        color = np.unique(data['color'])[0]

        idxa = np.sqrt(data['Cov_colorcolor']) < 100000.

        sela = data[idxa]
        # df = sela.to_pandas()
        df = pd.DataFrame(np.copy(sela))

        idx = np.sqrt(sela['Cov_colorcolor']) <= 0.04

        # df_sel = sela[idx].to_pandas()
        df_sel = pd.DataFrame(np.copy(sela[idx]))

        group = df.groupby('z')

        group_sel = df_sel.groupby('z')

        rb = (group_sel.size()/group.size())
        err = np.sqrt(rb*(1.-rb)/group.size())

        rb = rb.array
        err = err.array

        rb[np.isnan(rb)] = 0.
        err[np.isnan(err)] = 0.

        res = Table()

        res.add_column(Column(list(group.groups.keys()), name=group.keys))
        res.add_column(Column(rb, name='effi'))
        res.add_column(Column(err, name='effi_err'))
        res.add_column(Column([healpixID]*len(res), 'healpixID'))
        res.add_column(Column([pixRa]*len(res), 'pixRa'))
        res.add_column(Column([pixDec]*len(res), 'pixDec'))
        res.add_column(Column([season]*len(res), 'season'))
        res.add_column(Column([x1]*len(res), 'x1'))
        res.add_column(Column([color]*len(res), 'color'))

        return res
"""
