import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from sn_maf.sn_stackers.coadd_stacker import CoaddStacker
import healpy as hp
import numpy.lib.recfunctions as rf
import multiprocessing
from sn_maf.sn_tools.sn_cadence_tools import Generate_Fake_Observations
import yaml
from scipy import interpolate


class SNSNRMetric(BaseMetric):
    """
    Measure SN-SNR as a function of time.
    """

    def __init__(self, metricName='SNSNRMetric',
                 mjdCol='observationStartMJD', RaCol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', coadd=True, lim_sn=None, names_ref=None, z=0.01,
                 uniqueBlocks=False, config=None, **kwargs):

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

        cols = [self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol]
        if coadd:
            cols += ['coadd']
        super(SNSNRMetric, self).__init__(
            col=cols, metricDtype='object', metricName=metricName, **kwargs)

        self.filterNames = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        self.config = config
        self.blue_cutoff = 300.
        self.red_cutoff = 800.
        self.min_rf_phase = -20.
        self.max_rf_phase = 60.
        self.z = z
        # this is for output
        """
        save_status = config['Output']['save']
        outdir = config['Output']['directory']
        prodid = config['ProductionID']
        """
        # sn parameters
        sn_parameters = config['SN parameters']

        # SN DayMax: current date - shift days
        self.shift = sn_parameters['shift']
        self.field_type = config['Observations']['fieldtype']
        self.season = config['Observations']['season']
        # self.season = season
        area = 9.6  # survey_area in sqdeg - 9.6 by default for DD
        if self.field_type == 'WFD':
            # in that case the survey area is the healpix area
            area = hp.nside2pixarea(
                config['Pixelisation']['nside'], degrees=True)

        # Load the reference Li file

        # self.Li = np.load(config['Reference File'])
        self.lim_sn = lim_sn
        self.names_ref = names_ref

        self.display = config['Display_Processing']

    def run(self, dataSlice,  slicePoint=None):

        goodFilters = np.in1d(dataSlice['filter'], self.filterNames)
        dataSlice = dataSlice[goodFilters]
        if dataSlice.size == 0:
            return None
        dataSlice.sort(order=self.mjdCol)

        time = dataSlice[self.mjdCol]-dataSlice[self.mjdCol].min()

        seasons = [float(seas) for seas in self.season]
        # seasons = self.season
        if self.season == -1:
            seasons = np.unique(dataSlice[self.seasonCol])

        # get infos on seasons
        self.info_season = self.SeasonInfo(dataSlice, seasons)

        snr_obs = self.SNR_Season(dataSlice, seasons)  # SNR for observations
        snr_fakes = self.SNR_Fakes(dataSlice, seasons)  # SNR for fakes
        detect_frac = self.Detecting_Fraction(
            snr_obs, snr_fakes)  # Detection fraction

        snr_obs = np.asarray(snr_obs)
        snr_fakes = np.asarray(snr_fakes)
        detect_frac = np.asarray(detect_frac)

        return {'snr_obs': snr_obs, 'snr_fakes': snr_fakes, 'detec_frac': detect_frac}

    def SNR_Season(self, dataSlice, seasons, j=-1, output_q=None):
        """
        Estimate SNR for all seasons
        Input: dataSlice, seasons
        Output: array with the following fields all are of f8 type, except band which is of U1
        SNR_SNCosmo , SNR_SNSim :  Signal-To-Noise Ratio estimator
        season : season
        cadence: cadence of the season
        season_length: length of the season
        MJD_min: min MJD of the season
        DayMax: SN max luminosity MJD (aka T0)
        MJD:
        m5_eff: mean m5 of obs passing the min_phase, max_phase cut
        fieldRA: mean field RA
        fieldDec: mean field Dec
        band:  band
        m5: mean m5 (over the season)
        Nvisits: median number of visits (per observation) (over the season)
        ExposureTime: median exposure time (per observation) (over the season)
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
        fieldRA = np.mean(sel[self.RaCol])
        fieldDec = np.mean(sel[self.DecCol])
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
        # flag = (phase >= self.min_rf_phase) & (phase <= self.max_rf_phase)
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
        global_info = [(fieldRA, fieldDec, band, m5,
                        Nvisits, exptime)]*len(snr)
        names = ['fieldRA', 'fieldDec', 'band',
                 'm5', 'Nvisits', 'ExposureTime']
        global_info = np.rec.fromrecords(global_info, names=names)
        snr = rf.append_fields(
            snr, names, [global_info[name] for name in names])

       # Display LC and SNR at the same time
        if self.display:
            self.Plot(fluxes_tot, mjd_vals, flag, snr, T0_lc, dates)

        if output_q is not None:
            output_q.put({j: snr})
        else:
            return snr

    def SeasonInfo(self, dataSlice, seasons):
        """
        Get info on seasons for each dataSlice
        input : dataSlice
        output: recordarray with the following:
        season, cadence, season_length, MJDmin, MJDmax
        """

        rv = []
        for season in seasons:
            idx = (dataSlice[self.seasonCol] == season)
            slice_sel = dataSlice[idx]
            slice_sel.sort(order=self.mjdCol)
            mjds_season = slice_sel[self.mjdCol]
            cadence = np.mean(mjds_season[1:]-mjds_season[:-1])
            mjd_min = np.min(mjds_season)
            mjd_max = np.max(mjds_season)
            season_length = mjd_max-mjd_min
            rv.append((float(season), cadence, season_length, mjd_min, mjd_max))

        info_season = np.rec.fromrecords(
            rv, names=['season', 'cadence', 'season_length', 'MJD_min', 'MJD_max'])

        return info_season

    def SNR(self, time_lc, m5_vals, flag, season_vals, T0_lc):
        """
        Estimate SNR vs time
        Input: time(MJDS-T0), m5, seasons of observations
        flag: selection of LC points
        Output:
        estimated fluxes
        array of snr
        """
        seasons = np.ma.array(season_vals, mask=~flag)
        fluxes_tot = {}
        snr_tab = None

        for ib, name in enumerate(self.names_ref):
            fluxes = self.lim_sn.fluxes[ib](time_lc)
            if name not in fluxes_tot.keys():
                fluxes_tot[name] = fluxes
            else:
                fluxes_tot[name] = np.concatenate((fluxes_tot[name], fluxes))

            flux_5sigma = self.lim_sn.mag_to_flux[ib](m5_vals)
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
            season_recover = self.GetSeason(
                T0_lc[np.where(snr_tab.mask)])
            tofill[idmask] = season_recover
            snr_tab = np.ma.filled(snr_tab, fill_value=tofill)

        return fluxes_tot, snr_tab

    def GetSeason(self, T0):
        """
        Estimate the seasons corresponding to T0 values
        Input: set of T0s
        Output: set of correspondinf seasons
        """

        diff_min = T0[:, None]-self.info_season['MJD_min']
        diff_max = -T0[:, None]+self.info_season['MJD_max']
        seasons = np.tile(self.info_season['season'], (len(diff_min), 1))
        flag = (diff_min >= 0) & (diff_max >= 0)
        seasons = np.ma.array(seasons, mask=~flag)

        return np.mean(seasons, axis=1)

    def SNR_Fakes(self, dataSlice, seasons):
        """
        Estimate SNR for fake observations
        in the same way as for observations (using SNR_Season)
        """
        # generate fake observations
        fake_obs = None
        for season in seasons:

            idx = (dataSlice[self.seasonCol] == season)
            band = np.unique(dataSlice[idx][self.filterCol])[0]
            if fake_obs is None:
                fake_obs = self.Gen_Fakes(dataSlice[idx], band, season)
            else:
                fake_obs = np.concatenate(
                    (fake_obs, self.Gen_Fakes(dataSlice[idx], band, season)))

        # estimate SNR vs MJD

        snr_fakes = self.SNR_Season(
            fake_obs[fake_obs['filter'] == band], seasons=seasons)

        return snr_fakes

    def Gen_Fakes(self, slice_sel, band, season):
        """
        Generate fake observations
        according to observing values extracted from simulations
        """
        fieldRA = np.mean(slice_sel[self.RaCol])
        fieldDec = np.mean(slice_sel[self.DecCol])
        mjds_season = slice_sel[self.mjdCol]
        cadence = np.mean(mjds_season[1:]-mjds_season[:-1])
        mjd_min = np.min(mjds_season)
        mjd_max = np.max(mjds_season)
        season_length = mjd_max-mjd_min
        Nvisits = np.median(slice_sel[self.nexpCol])
        m5 = np.median(slice_sel[self.m5Col])
        Tvisit = 30.

        config_fake = yaml.load(open(self.config['Fake_file']))
        config_fake['Ra'] = fieldRA
        config_fake['Dec'] = fieldDec
        config_fake['bands'] = [band]
        config_fake['Cadence'] = [cadence]
        config_fake['MJD_min'] = [mjd_min]
        config_fake['season_length'] = season_length
        config_fake['Nvisits'] = [Nvisits]
        m5_nocoadd = m5-1.25*np.log10(float(Nvisits)*Tvisit/30.)
        config_fake['m5'] = [m5_nocoadd]
        config_fake['seasons'] = [season]
        fake_obs_season = Generate_Fake_Observations(config_fake).Observations

        return fake_obs_season

    def Plot(self, fluxes, mjd, flag, snr, T0_lc, dates):

        dir_save = '/home/philippe/LSST/sn_metric_new/Plots'
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
        # print(mjd_ma)
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
            plt.savefig('{}/{}_{}.png'.format(dir_save, 'snr', 1000 + j))
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

        # print(test)

    def Detecting_Fraction(self, snr_obs, snr_fakes):
        """
        Estimate the time fraction (per season) for which
        snr_obs > snr_fakes
        This would correspond to a fraction of detectability.
        For regular cadences one should get a result close to 1
        """

        ra = np.mean(snr_obs['fieldRA'])
        dec = np.mean(snr_obs['fieldDec'])
        band = np.unique(snr_obs['band'])[0]

        rtot = []

        for season in np.unique(snr_obs['season']):
            idx = snr_obs['season'] == season
            sel_obs = snr_obs[idx]
            idxb = snr_fakes['season'] == season
            sel_fakes = snr_fakes[idxb]

            sel_obs.sort(order='MJD')
            sel_fakes.sort(order='MJD')
            r = [ra, dec, season, band]
            names = [self.RaCol, self.DecCol, 'season', 'band']
            for sim in self.config['names_ref']:
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
