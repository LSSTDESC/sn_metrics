import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from sn_stackers.coadd_stacker import CoaddStacker
import healpy as hp
import numpy.lib.recfunctions as rf
import multiprocessing
from sn_tools.sn_cadence_tools import GenerateFakeObservations
from sn_tools.sn_obs import getPix
import yaml
from scipy import interpolate
import os
import pandas as pd
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery

class SNSNRMetric(BaseMetric):
    """
    Measure SN-Signal-to-Noise Ratio as a function of time.
    Extract the detection rate from these measurements

    Parameters
    --------------
    lim_sn : str
       SN reference data (LC)
    names_ref : str
       names of the simulator used to generate reference files
    metricName : str, opt
      metric name
      Default : SNSNRMetric
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
       list of seasons to process (float)
       Default : -1. (all seasons)
    shift : float,opt
       T0 of the SN to consider = MJD-shift
       Default: 10.
    z : float,opt
       redshift for the study
       Default : 0.01


    """

    def __init__(self, lim_sn, names_ref, fake_file,
                 metricName='SNSNRMetric',
                 mjdCol='observationStartMJD', RACol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', seeingaCol='seeingFwhmEff',
                 seeingbCol='seeingFwhmGeom',
                 # airmassCol='airmass',skyCol='sky', moonCol='moonPhase'
                 season=[-1], shift=10., coadd=True, z=0.01,
                 display=False, nside=64, band='r', verbose=False, **kwargs):

        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.RACol = RACol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.seasonCol = 'season'
        self.nightCol = nightCol
        self.obsidCol = obsidCol
        self.nexpCol = nexpCol
        self.vistimeCol = vistimeCol
        self.seeingaCol = seeingaCol
        self.seeingbCol = seeingbCol
        self.nside = nside
        self.band = band
        self.verbose = verbose

        cols = [self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol, self.seeingaCol, self.seeingbCol]
        self.stacker = None
        if coadd:
            cols += ['coadd']
            self.stacker = CoaddStacker(mjdCol=self.mjdCol, RACol=self.RACol, DecCol=self.DecCol, m5Col=self.m5Col, nightCol=self.nightCol,
                                        filterCol=self.filterCol, numExposuresCol=self.nexpCol, visitTimeCol=self.vistimeCol, visitExposureTimeCol='visitExposureTime', seeingaCol=self.seeingaCol, seeingbCol=self.seeingbCol)

        super(SNSNRMetric, self).__init__(
            col=cols, metricDtype='object', metricName=metricName, **kwargs)

        self.filterNames = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        # self.config = config
        self.blue_cutoff = 300.
        self.red_cutoff = 800.
        self.min_rf_phase = -20.
        self.max_rf_phase = 60.
        self.z = z
        self.shift = shift
        self.season = season
        self.fakeFile = fake_file
        # sn parameters
        # sn_parameters = config['SN parameters']

        # SN DayMax: current date - shift days
        # self.shift = sn_parameters['shift']
        # self.field_type = config['Observations']['fieldtype']
        # self.season = config['Observations']['season']
        # self.season = season
        # area = 9.6  # survey_area in sqdeg - 9.6 by default for DD
        # Load the reference Li file

        # self.Li = np.load(config['Reference File'])

        self.lim_sn = lim_sn
        self.names_ref = names_ref

        self.display = display

    def run(self, dataSlice,  slicePoint=None):
        """
        Run method for the metric


        Returns
        ----------

        pandas df with the following cols:
        fieldRA,fieldDec: RA and Dec of the field
        pixRA,pixDec, healpixID: pixel RA, Dec, ID
        season: season
        band: filter
        frac_obs_SNCosmo: frac of events with SNR>=SNR_ref
        """

        """
        goodFilters = np.in1d(dataSlice[self.filterCol], self.filterNames)
        dataSlice = dataSlice[goodFilters]
        if dataSlice.size == 0:
            return None
        dataSlice.sort(order=self.mjdCol)

        time = dataSlice[self.mjdCol]-dataSlice[self.mjdCol].min()
        """

        # select data corresponding to the band
        idx = dataSlice[self.filterCol] == self.band

        dataSlice = dataSlice[idx]

        
        # get pixRA and pixDec if necessary
        if 'pixRA' not in dataSlice.dtype.names:
            healpixID, pixRA, pixDec = getPix(dataSlicef.nside,
                                              np.mean(dataSlice[dataSlicef.RACol]),
                                              np.mean(dataSlice[dataSlicef.DecCol]))

            dataSlice = rf.append_fields(dataSlice, 'healpixID', [
                healpixID]*len(dataSlice))
            dataSlice = rf.append_fields(
                dataSlice, 'pixRA', [pixRA]*len(dataSlice))
            dataSlice = rf.append_fields(
                dataSlice, 'pixDec', [pixDec]*len(dataSlice))

        # get ebvofMW for this pixel
        pixRA = np.unique(dataSlice['pixRA']).item()
        pixDec = np.unique(dataSlice['pixDec']).item()
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


        # stack the data if requested
        if self.stacker is not None:
            dataSlice = self.stacker._run(dataSlice)

        seasons = self.season

        if self.season == [-1]:
            seasons = np.unique(dataSlice[self.seasonCol])

        # get infos on seasons
        self.info_season = self.seasonInfo(dataSlice, seasons)
        if self.info_season is None:
            return None

        seasons = self.info_season['season']

        snr_obs = self.snrSeason(dataSlice, seasons)  # SNR for observations
        snr_fakes = self.snrFakes(dataSlice, seasons)  # SNR for fakes
        detect_frac = self.detectingFraction(
            snr_obs, snr_fakes)  # Detection fraction

        snr_obs = np.asarray(snr_obs)
        snr_fakes = np.asarray(snr_fakes)
        detect_frac = np.asarray(detect_frac)

        # return {'snr_obs': snr_obs, 'snr_fakes': snr_fakes, 'detec_frac': detect_frac}
        if self.verbose:
            print('processed', detect_frac)

        return pd.DataFrame(detect_frac)

    def snrSeason(self, dataSlice, seasons, j=-1, output_q=None):
        """
        Estimate SNR for a dataSlice

        Parameters
        --------------

        dataSlice: array
          Array of observations
        seasons: list
          list of seasons to consider
        j: int, opt
          index for multiprocessing
          Default: -1
        output_q: multiprocessing queue, opt
          queue for multiprocessing
          Default: none

        Returns
        ----------

        array with the following fields(all are of f8 type, except band which is of U1)
        SNR_name_ref:  Signal-To-Noise Ratio estimator
        season: season
        cadence: cadence of the season
        season_length: length of the season
        MJD_min: min MJD of the season
        DayMax: SN max luminosity MJD(aka T0)
        MJD:
        m5_eff: mean m5 of obs passing the min_phase, max_phase cut
        fieldRA: mean field RA
        fieldDec: mean field Dec
        pixRA: mean field pixRA
        pixDec: mean field pixDec
        healpixID: field healpix Id
        band:  band
        m5: mean m5(over the season)
        Nvisits: median number of visits(per observation)(over the season)
        ExposureTime: median exposure time(per observation)(over the season)
        """

        # check whether we do have data for the considered season for this dataSlice
        sel = None
        for season in seasons:
            idx = dataSlice[self.seasonCol] == season
            if sel is None:
                sel = dataSlice[idx]
            else:
                sel = np.concatenate((sel, dataSlice[idx]))

        if len(sel) == 0:
            return None

  

        # Get few infos: RA, Dec, Nvisits, m5, exptime
        fieldRA = np.mean(sel[self.RACol])
        fieldDec = np.mean(sel[self.DecCol])
        pixRA = np.mean(sel['pixRA'])
        pixDec = np.mean(sel['pixDec'])
        healpixID = int(np.unique(sel['healpixID'])[0])

        Nvisits = np.median(sel[self.nexpCol]/2.)  # one visit = 2 exposures
        m5 = np.mean(sel[self.m5Col])
        exptime = np.median(sel[self.exptimeCol])
        sel.sort(order=self.mjdCol)
        mjds = sel[self.mjdCol]
        band = np.unique(sel[self.filterCol])[0]

        # Define MJDs to consider for metric estimation
        # basically: step of one day between MJDmin and MJDmax
        dates = None
        for val in self.info_season:
            if dates is None:
                dates = np.arange(val['MJD_min'], val['MJD_max']+1., 1.)
            else:
                dates = np.concatenate(
                    (dates, np.arange(val['MJD_min'], val['MJD_max']+1., 1.)))

        # SN  DayMax: dates-shift where shift is chosen in the input yaml file
        T0_lc = dates-self.shift

        # for these DayMax, estimate the phases of LC points corresponding to the current dataSlice MJDs
        diff_time = dates[:, np.newaxis]-mjds
        time_for_lc = -T0_lc[:, None]+mjds

        phase = time_for_lc/(1.+self.z)  # phases of LC points
        # flag: select LC points only in between min_rf_phase and max_phase
        phase_max = self.shift/(1.+self.z)
        flag = (phase >= self.min_rf_phase) & (phase <= phase_max)

        # tile m5, MJDs, and seasons to estimate all fluxes and SNR at once
        m5_vals = np.tile(sel[self.m5Col], (len(time_for_lc), 1))
        mjd_vals = np.tile(sel[self.mjdCol], (len(time_for_lc), 1))
        season_vals = np.tile(sel[self.seasonCol], (len(time_for_lc), 1))

        # estimate fluxes and snr in SNR function
        fluxes_tot, snr = self.SNR(
            time_for_lc, m5_vals, flag, season_vals, T0_lc)

        # now save the results in a record array
        snr_nomask = np.ma.copy(snr)
        _, idx = np.unique(snr['season'], return_inverse=True)
        infos = self.info_season[idx]
        vars_info = ['cadence', 'season_length', 'MJD_min']
        snr = rf.append_fields(
            snr, vars_info, [infos[name] for name in vars_info])
        snr = rf.append_fields(snr, 'DayMax', T0_lc)
        snr = rf.append_fields(snr, 'MJD', dates)
        snr = rf.append_fields(snr, 'm5_eff', np.mean(
            np.ma.array(m5_vals, mask=~flag), axis=1))
        global_info = [(fieldRA, fieldDec, pixRA, pixDec, healpixID, band, m5,
                        Nvisits, exptime)]*len(snr)
        names = ['fieldRA', 'fieldDec',
                 'pixRA', 'pixDec',
                 'healpixID', 'band',
                 'm5', 'Nvisits',
                 'ExposureTime']
        global_info = np.rec.fromrecords(global_info, names=names)
        snr = rf.append_fields(
            snr, names, [global_info[name] for name in names])

       # Display LC and SNR at the same time
        if self.display:
            self.plot(fluxes_tot, mjd_vals, flag, snr, T0_lc, dates)

        if output_q is not None:
            output_q.put({j: snr})
        else:
            return snr

    def seasonInfo(self, dataSlice, seasons):
        """
        Get info on seasons for each dataSlice

        Parameters
        --------------

        dataSlice: array
          array of observations

        Returns
        ------------
        recordarray with the following fields:
        season, cadence, season_length, MJDmin, MJDmax
        """

        rv = []
        for season in seasons:
            idx = (dataSlice[self.seasonCol] == season)
            slice_sel = dataSlice[idx]
            if len(slice_sel) < 5:
                continue
            slice_sel.sort(order=self.mjdCol)
            mjds_season = slice_sel[self.mjdCol]
            cadence = np.mean(mjds_season[1:]-mjds_season[:-1])
            mjd_min = np.min(mjds_season)
            mjd_max = np.max(mjds_season)
            season_length = mjd_max-mjd_min
            # Nvisits = np.median(slice_sel[self.nexpCol])
            Nvisits = len(slice_sel)
            rv.append((float(season), cadence, season_length,
                       mjd_min, mjd_max, Nvisits))

        info_season = None
        if len(rv) > 0:
            info_season = np.rec.fromrecords(
                rv, names=['season', 'cadence', 'season_length', 'MJD_min', 'MJD_max', 'Nvisits'])

        return info_season

    def SNR(self, time_lc, m5_vals, flag, season_vals, T0_lc):
        """
        Estimate SNR vs time

        Parameters
        -------------

        time_lc: numpy array(float)
           deltaT = mjd-time_T0
        m5_vals: array(float)
           five-sigma depth values
        flag: array(bool)
          flag to be applied(example: selection from phase cut)
        season_vals: array(float)
          season values
        T0_lc: array(float)
           array of T0 for supernovae

        Returns
        ----------
        fluxes_tot: list(float)
         list of(interpolated) fluxes
        snr_tab: array with the following fields:
          snr_name_ref(float): Signal-to-Noise values
          season(float): season num.
        """

        seasons = np.ma.array(season_vals, mask=~flag)

        fluxes_tot = {}
        snr_tab = None

        for ib, name in enumerate(self.names_ref):
            fluxes = self.lim_sn.fluxes[ib](np.copy(time_lc))
            if name not in fluxes_tot.keys():
                fluxes_tot[name] = fluxes
            else:
                fluxes_tot[name] = np.concatenate((fluxes_tot[name], fluxes))

            flux_5sigma = self.lim_sn.mag_to_flux[ib](np.copy(m5_vals))
            snr = fluxes**2/flux_5sigma**2
            snr_season = 5.*np.sqrt(np.sum(snr*flag, axis=1))
            if snr_tab is None:
                snr_tab = np.asarray(np.copy(snr_season), dtype=[
                                     ('SNR_'+name, 'f8')])
            else:
                snr_tab = rf.append_fields(
                    snr_tab, 'SNR_'+name, np.copy(snr_season))
        snr_tab = rf.append_fields(snr_tab, 'season', np.mean(seasons, axis=1))

        # check if any masked value remaining
        # this would corrspond to case where no obs point has been selected
        # ie no points with phase in [phase_min,phase_max]
        # this happens when internight gaps are large (typically larger than shift)
        idmask = np.where(snr_tab.mask)
        if len(idmask) > 0:
            tofill = np.copy(snr_tab['season'])
            season_recover = self.getSeason(
                T0_lc[np.where(snr_tab.mask)])
            tofill[idmask] = season_recover
            snr_tab = np.ma.filled(snr_tab, fill_value=tofill)

        return fluxes_tot, snr_tab

    def getSeason(self, T0):
        """
        Estimate the seasons corresponding to T0 values

        Parameters
        ---------------
        T0: array(float)
           set of T0 values

        Returns
        -----------
        array(float) of corresponding seasons
        """

        diff_min = T0[:, None]-self.info_season['MJD_min']
        diff_max = -T0[:, None]+self.info_season['MJD_max']
        seasons = np.tile(self.info_season['season'], (len(diff_min), 1))
        flag = (diff_min >= 0) & (diff_max >= 0)
        seasons = np.ma.array(seasons, mask=~flag)

        return np.mean(seasons, axis=1)

    def snrFakes(self, dataSlice, seasons):
        """
        Estimate SNR for fake observations
        in the same way as for observations(using snrSeason)

        Parameters
        --------------

        dataSlice: array
           array of observations

        Returns
        ----------
        snr_tab: array with the following fields:
          snr_name_ref(float): Signal-to-Noise values
          season(float): season num.
        """

        # generate fake observations
        fake_obs = None
        for season in seasons:

            idx = (dataSlice[self.seasonCol] == season)
            band = np.unique(dataSlice[idx][self.filterCol])[0]
            fakes = self.genFakes(dataSlice[idx], band, season)

            if fake_obs is None:
                fake_obs = fakes
            else:
                fake_obs = np.concatenate((fake_obs, fakes))

        # estimate SNR vs MJD

        snr_fakes = self.snrSeason(
            fake_obs[fake_obs['filter'] == band], seasons=seasons)

        return snr_fakes

    def genFakes(self, slice_sel, band, season):
        """
        Generate fake observations
        according to observing values extracted from simulations

        Parameters
        --------------
        slice_sel: array
          array of observations
        band: str
          band to consider

        Returns
        ----------
        fake_obs_season: array
          array of observations with the following fields
          observationStartMJD(float)
          fieldRA(float)
          fieldDec(float)
          pixRA(float)
          pixDec(float)
          healpixID(int)
          filter(U1)
          fiveSigmaDepth(float)
          numExposures(float)
          visitExposureTime(float)
          season(int)
        """
        slice_sel.sort(order=self.mjdCol)
        fieldRA = np.mean(slice_sel[self.RACol])
        fieldDec = np.mean(slice_sel[self.DecCol])
        pixRA, pixDec, healpixID = 0., 0., 0

        if 'pixRA' in slice_sel.dtype.names:
            pixRA = np.mean(slice_sel['pixRA'])
            pixDec = np.mean(slice_sel['pixDec'])
            healpixID = int(np.unique(slice_sel['healpixID'])[0])
        mjds_season = slice_sel[self.mjdCol]
        cadence = np.mean(mjds_season[1:]-mjds_season[:-1])
        mjd_min = np.min(mjds_season)
        mjd_max = np.max(mjds_season)
        season_length = mjd_max-mjd_min
        Nvisits = np.median(slice_sel[self.nexpCol])
        m5 = np.median(slice_sel[self.m5Col])
        Tvisit = 30.

        f = open(self.fakeFile, 'r')
        config_fake = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
        config_fake['RA'] = fieldRA
        config_fake['Dec'] = fieldDec
        config_fake['bands'] = [band]
        config_fake['Cadence'] = [cadence]
        config_fake['MJD_min'] = [mjd_min]
        config_fake['season_length'] = season_length
        config_fake['Nvisits'] = [Nvisits]
        m5_nocoadd = m5-1.25*np.log10(float(Nvisits)*Tvisit/30.)
        config_fake['m5'] = [m5_nocoadd]
        config_fake['seasons'] = [season]
        config_fake['seeingEff'] = [0.87]
        config_fake['seeingGeom'] = [0.87]

        fake_obs_season = GenerateFakeObservations(config_fake).Observations

        fake_obs_season = rf.append_fields(fake_obs_season, 'pixRA', [
                                           pixRA]*len(fake_obs_season))
        fake_obs_season = rf.append_fields(fake_obs_season, 'pixDec', [
                                           pixDec]*len(fake_obs_season))
        fake_obs_season = rf.append_fields(fake_obs_season, 'healpixID', [
                                           healpixID]*len(fake_obs_season))

        return fake_obs_season

    def plot(self, fluxes, mjd, flag, snr, T0_lc, dates):
        """
        Plotting method to illustrate the method used to estimate the metric.
        Two plots(one for scheduler data, the other for fake data)are displayed(in "real time")
        and are composed of two plots:
            - top plot: SN flux(in pe/sec) as a function of mjd
            - bottom: SNR as a function of mjd

        Parameters
        --------------
        fluxes:
          SN fluxes
        mjd: float
         mjd chosen for the display
        flag: array(bool)
         flag to be applied(example: selection from phase cut)
        snr: array(float)
         snr values
        T0_lc: float
          max luminosity time for SN
        dates:

        """

        dirSave = 'Plots'
        if not os.path.isdir(dirSave):
            os.makedirs(dirSave)
        import pylab as plt
        plt.ion()
        fig, ax = plt.subplots(ncols=1, nrows=2)
        fig.canvas.draw()

        colors = ['b', 'r']
        myls = ['-', '--']
        mfc = ['b', 'None']
        tot_label = []
        fontsize = 12
        mjd_ma = np.ma.array(mjd, mask=~flag)

        fluxes_ma = {}
        for key, val in fluxes.items():
            fluxes_ma[key] = np.ma.array(val, mask=~flag)
        key = list(fluxes.keys())[0]
        jmax = len(fluxes_ma[key])
        tot_label = []
        tot_label_snr = []
        min_flux = []
        max_flux = []
        for j in range(jmax):

            for ib, name in enumerate(fluxes_ma.keys()):
                tot_label.append(ax[0].errorbar(
                    mjd_ma[j], fluxes_ma[name][j], marker='s', color=colors[ib], ls=myls[ib], label=name))

                tot_label_snr.append(ax[1].errorbar(
                    snr['MJD'][:j], snr['SNR_'+name][:j], color=colors[ib], label=name))
                fluxx = fluxes_ma[name][j]
                fluxx = fluxx[~fluxx.mask]
                if len(fluxx) >= 2:
                    min_flux.append(np.min(fluxx))
                    max_flux.append(np.max(fluxx))
                else:
                    min_flux.append(0.)
                    max_flux.append(200.)

            min_fluxes = np.min(min_flux)
            max_fluxes = np.max(max_flux)

            tot_label.append(ax[0].errorbar([T0_lc[j], T0_lc[j]], [
                             min_fluxes, max_fluxes], color='k', label='DayMax'))
            tot_label.append(ax[0].errorbar([dates[j], dates[j]], [
                             min_fluxes, max_fluxes], color='k', ls='--', label='Current MJD'))
            fig.canvas.flush_events()
            plt.savefig('{}/{}_{}.png'.format(dirSave, 'snr', 1000 + j))
            if j != jmax-1:
                ax[0].clear()
                tot_label = []
                tot_label_snr = []

        labs = [l.get_label() for l in tot_label]
        ax[0].legend(tot_label, labs, ncol=1, loc='best',
                     prop={'size': fontsize}, frameon=False)
        ax[0].set_ylabel('Flux [e.sec$^{-1}$]', fontsize=fontsize)

        ax[1].set_xlabel('MJD', fontsize=fontsize)
        ax[1].set_ylabel('SNR', fontsize=fontsize)
        ax[1].legend()
        labs = [l.get_label() for l in tot_label_snr]
        ax[1].legend(tot_label_snr, labs, ncol=1, loc='best',
                     prop={'size': fontsize}, frameon=False)
        for i in range(2):
            ax[i].tick_params(axis='x', labelsize=fontsize)
            ax[i].tick_params(axis='y', labelsize=fontsize)

    def detectingFraction(self, snr_obs, snr_fakes):
        """
        Estimate the time fraction(per season) for which
        snr_obs > snr_fakes = detection rate
        For regular cadences one should get a result close to 1

        Parameters
        ---------------
        snr_obs: array
         array estimated using snrSeason(observations)
         snr_fakes: array
           array estimated using snrSeason(fakes)

        Returns
        ----------
        record array with the following fields:
          fieldRA(float)
          fieldDec(float)
          season(float)
         band (str)
         frac_obs_name_ref (float)
        """

        ra = np.mean(snr_obs['fieldRA'])
        dec = np.mean(snr_obs['fieldDec'])
        pixRA = np.mean(snr_obs['pixRA'])
        pixDec = np.mean(snr_obs['pixDec'])
        healpixID = int(np.unique(snr_obs['healpixID'])[0])

        band = np.unique(snr_obs['band'])[0]

        rtot = []

        for season in np.unique(snr_obs['season']):
            idx = snr_obs['season'] == season
            sel_obs = snr_obs[idx]
            idxb = snr_fakes['season'] == season
            sel_fakes = snr_fakes[idxb]
            sel_obs.sort(order='MJD')
            sel_fakes.sort(order='MJD')
            r = [ra, dec, pixRA, pixDec, healpixID, season, band]
            names = [self.RACol, self.DecCol, 'pixRA',
                     'pixDec', 'healpixID', 'season', 'band']
            for sim in self.names_ref:
                fakes = interpolate.interp1d(
                    sel_fakes['MJD'], sel_fakes['SNR_'+sim])
                obs = interpolate.interp1d(sel_obs['MJD'], sel_obs['SNR_'+sim])
                mjd_min = np.max(
                    [np.min(sel_obs['MJD']), np.min(sel_fakes['MJD'])])
                mjd_max = np.min(
                    [np.max(sel_obs['MJD']), np.max(sel_fakes['MJD'])])
                mjd = np.arange(mjd_min, mjd_max, 1.)

                diff_res = obs(mjd)-fakes(mjd)
                idx = diff_res >= 0
                r += [len(diff_res[idx])/len(diff_res)]
                names += ['frac_obs_'+sim]
            rtot.append(tuple(r))
        return np.rec.fromrecords(rtot, names=names)
