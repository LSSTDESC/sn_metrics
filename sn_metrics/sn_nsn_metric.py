import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from sn_stackers.coadd_stacker import CoaddStacker
import healpy as hp
import numpy.lib.recfunctions as rf
import multiprocessing
import yaml
from scipy import interpolate
import os
from sn_tools.sn_calcFast import LCfast, CalcSN, CalcSN_df
from sn_tools.sn_telescope import Telescope
from astropy.table import Table, vstack, Column
import time
import pandas as pd
from scipy.interpolate import interp1d
from sn_tools.sn_rate import SN_Rate


class SNNSNMetric(BaseMetric):
    """
    Measure zlim of type Ia supernovae.

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
    RaCol : str,opt
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

    def __init__(self, lc_reference,
                 metricName='SNNSNMetric',
                 mjdCol='observationStartMJD', RaCol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', season=-1, coadd=True, zmin=0.0, zmax=1.0,
                 pixArea=9.6, verbose=False, ploteffi=False, **kwargs):

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
        self.lcFast = {}

        for key, vals in lc_reference.items():
            self.lcFast[key] = LCfast(vals, key[0], key[1], telescope,
                                      self.mjdCol, self.RaCol, self.DecCol,
                                      self.filterCol, self.exptimeCol,
                                      self.m5Col, self.seasonCol)
        self.zmin = zmin
        self.zmax = zmax
        self.zStep = 0.05  # zstep
        self.daymaxStep = 3.  # daymax step
        self.min_rf_phase = -20.
        self.max_rf_phase = 60.

        self.min_rf_phase_qual = -10.
        self.max_rf_phase_qual = 20.

        # snrate
        self.rateSN = SN_Rate(
            min_rf_phase=self.min_rf_phase_qual, max_rf_phase=self.max_rf_phase_qual)

        # verbose mode - useful for debug and code performance estimation
        self.verbose = verbose
        #self.verbose = True
        # Two SN considered
        # self.nameSN = dict(zip([(-2.0, 0.2), (0.0, 0.0)], ['faint', 'medium']))

        # status of the pixel after processing
        self.status = dict(
            zip(['ok', 'effi', 'season_length', 'nosn'], [1, -1, -2, -3]))

    def run(self, dataSlice,  slicePoint=None):

        time_ref = time.time()

        seasons = self.season

        if self.season == -1:
            seasons = np.unique(dataSlice[self.seasonCol])

        # get infos on seasons
        self.info_season = self.seasonInfo(dataSlice, seasons)
        if self.info_season is None:
            return None

        zRange = list(np.arange(self.zmin, self.zmax, self.zStep))
        if zRange[0] < 1.e-6:
            zRange[0] = 0.01

        pixRa = np.unique(dataSlice['pixRa'])[0]
        pixDec = np.unique(dataSlice['pixDec'])[0]
        healpixID = np.unique(dataSlice['healpixID'])[0]

        # possible outputs - by default zlim are returned
        # effi_tot = Table()  # efficiencies vs z
        # zlim_nsn = Table()  # zlim

        effi_totdf = pd.DataFrame()
        zlim_nsndf = pd.DataFrame()

        if self.verbose:
            print('Field', pixRa, pixDec, healpixID)
            print('Info seasons', self.info_season)

        # Loop on seasons

        # outvals, namesout = self.fillOutput(pixRa, pixDec, healpixID)

        # print('here', outvals, namesout)
        groupnames = ['pixRa', 'pixDec', 'healpixID', 'season', 'x1', 'color']
        for seas in self.info_season:
            # for each season:
            # generate simulation parameters
            # estimate LC points from fast simu (LCfast)
            # get error on color
            # estimate efficiency curve

            season = seas['season']
            # define output table here
            """
            rp = []

            for ival in range(len(outvals)):
                rp = outvals[ival]+[season]
            # names = namesout+['season']
            zlims = Table(rows=[rp], names=namesout)
            # zlims = pd.DataFrame(rp, columns=namesout)
            """
            if self.verbose:
                print('#### PROCESSING SEASON', season, pixRa, pixDec)

            time_refb = time.time()

            # select obs for this season
            idx = dataSlice['season'] == season
            obs = dataSlice[idx]

            if seas['season_length'] < 50.:
                # season too short -> not considered
                if self.verbose:
                    print('season too short', season, seas['season_length'])
                zlimsdf = self.errordf(
                    pixRa, pixDec, healpixID, season, self.status['season_length'])
                effi_seasondf = self.erroreffi(
                    pixRa, pixDec, healpixID, season)

            else:
                # stack obs (per band and per night) if necessary
                if self.stacker is not None:
                    obs = self.stacker._run(obs)
                    if self.verbose:
                        print('after stacking', season, time.time()-time_refb)

                # Simulation parameters
                gen_par, duration_z = self.simuParameters(season, zRange)

                # nslice = 1
                # nperSlice = int(nz/nslice)

                time_ref = time.time()

                if self.verbose:
                    print('NSN to simulate:', len(
                        gen_par), time.time()-time_ref)

                # get the supernova
                sn = self.run_season_slice(obs, gen_par)

                if sn is None:
                    # no SN generated
                    if self.verbose:
                        print('No supernova generated')

                    zlimsdf = self.errordf(
                        pixRa, pixDec, healpixID, season, self.status['nosn'])
                    effi_seasondf = self.erroreffi(
                        pixRa, pixDec, healpixID, season)
                else:
                    # Estimate efficiencies
                    if self.verbose:
                        print('estimating efficiencies')
                    # effi_season = self.effi(sn)
                    effi_seasondf = self.effidf(sn)
                    # concatenate all efficiencies over seasons
                    # effi_tot = vstack([effi_tot, effi_season])

                    durinterp_z = interp1d(
                        duration_z['z'], duration_z['season_length'], bounds_error=False, fill_value=0.)

                    zz, rate, err_rate, nsn, err_nsn = self.rateSN(zmin=self.zmin,
                                                                   zmax=self.zmax,
                                                                   duration_z=durinterp_z,
                                                                   survey_area=self.pixArea)
                    rateInterp = interp1d(zz, nsn, kind='linear',
                                          bounds_error=False, fill_value=0)

                    # Estimate zlim
                    # zlims = self.zlim(effi_season, rateInterp, season, zlims)

                    if self.verbose:
                        print('estimating zlimits')
                    zlimsdf = effi_seasondf.groupby(groupnames).apply(
                        lambda x: self.zlimdf(x, rateInterp)).reset_index()

                    if self.verbose:
                        print('zlims', zlimsdf)

                    # estimate number of supernovae
                    # znsn = self.nsn_type(
                    #    0.0, 0.0, effi_season, zlims, rateInterp)

                    if self.verbose:
                        print('Estimating number of supernovae')

                    zlimsdf['nsn'] = zlimsdf.apply(lambda x: self.nsn_typedf(
                        x, 0.0, 0.0, effi_seasondf, rateInterp), axis=1)

                    # print(znsn)
                    # print(zlimsdf)
                    if self.verbose:
                        print('final result ', zlimsdf)

                    # stack the results
                    # zlim_nsn = vstack([zlim_nsn, znsn])
            zlim_nsndf = pd.concat([zlimsdf, zlimsdf])
            effi_totdf = pd.concat([effi_totdf, effi_seasondf])
            print(zlimsdf, effi_totdf)
            if self.verbose:
                print('#### SEASON processed', time.time()-time_refb,
                      season, pixRa, pixDec)
        # return np.array(effi_tot)
        # print('hello',zlim_nsn)
        #print('resultat', zlim_nsndf.to_numpy())
        return zlim_nsndf.to_numpy()
        # return np.array(zlim_nsn)

    def simuParameters(self, season, zRange):

        idseas = self.info_season['season'] == season
        daymin = self.info_season[idseas]['MJD_min']
        daymax = self.info_season[idseas]['MJD_max']
        # print('hello',daymin,daymax)
        season_length = daymax-daymin

        # define T0 range for simulation
        daymaxRange = np.arange(daymin, daymax, self.daymaxStep)
        r_durz = []
        nz = int((self.zmax-self.zmin)/self.zStep)

        r = []

        for z in zRange:
            # for z in zRange[i*nperSlice:(i+1)*nperSlice]:

            T0_min = daymin-(1.+z)*self.min_rf_phase_qual
            T0_max = daymax-(1.+z)*self.max_rf_phase_qual
            r_durz.append((z, np.asscalar(T0_max-T0_min)))
            widthWindow = T0_max-T0_min
            if widthWindow < 1.:
                break
            daymaxRange = np.arange(T0_min, T0_max, self.daymaxStep)

            for mydaymax in daymaxRange:
                r.append(
                    (z, mydaymax, self.min_rf_phase, self.max_rf_phase))

        gen_par = np.rec.fromrecords(
            r, names=['z', 'daymax', 'min_rf_phase', 'max_rf_phase'])

        duration_z = np.rec.fromrecords(
            r_durz, names=['z', 'season_length'])
        return gen_par, duration_z

    def errordf(self, pixRa, pixDec, healpixID, season, errortype):

        return pd.DataFrame({'pixRa': [pixRa],
                             'pixDec': [pixDec],
                             'healpixID': [healpixID],
                             'season': [season],
                             'x1': [-1.0],
                             'color': [-1.0],
                             'zlim': [-1.0],
                             'nsn': [-1.0],
                             'status': [errortype]})

    def erroreffi(self, pixRa, pixDec, healpixID, season):

        return pd.DataFrame({'pixRa': [pixRa],
                             'pixDec': [pixDec],
                             'healpixID': [healpixID],
                             'season': [season],
                             'x1': [-1.0],
                             'color': [-1.0],
                             'z': [-1.0],
                             'effi': [-1.0],
                             'effi_err': [-1.0]})

    def effidf(self, sn_tot):

        sndf = pd.DataFrame(sn_tot)

        listNames = ['season', 'pixRa', 'pixDec', 'healpixID', 'x1', 'color']
        groups = sndf.groupby(listNames)

        sndf = pd.DataFrame(sn_tot)

        listNames = ['season', 'pixRa', 'pixDec', 'healpixID', 'x1', 'color']
        groups = sndf.groupby(listNames)

        effi = groups.apply(lambda x: self.effiObsdf(x)).reset_index()

        return effi

    def zlimdf(self, grp, rateInterp):
        # z range for the study
        zplot = list(np.arange(self.zmin, self.zmax, 0.001))

        # interpolate efficiencies vs z
        effiInterp = interp1d(
            grp['z'], grp['effi'], kind='linear', bounds_error=False, fill_value=0.)

        # estimate the cumulated number of SN vs z
        nsn_cum = np.cumsum(effiInterp(zplot)*rateInterp(zplot))

        zlimit = 0.0
        status = self.status['effi']
        if nsn_cum[-1] >= 1.e-5:
            nsn_cum_norm = nsn_cum/nsn_cum[-1]  # normalize
            zlim = interp1d(nsn_cum_norm, zplot)
            zlimit = np.asscalar(zlim(0.95))
            status = self.status['ok']
        return pd.DataFrame({'zlim': [zlimit],
                             'status': [status]})

    def nsn_typedf(self, grp, x1, color, effi_tot, rateInterp):

        effisel = effi_tot.loc[lambda dfa: (
            dfa['x1'] == x1) & (dfa['color'] == color), :]
        nsn = self.nsn(effisel, grp['zlim'], rateInterp)

        return nsn

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

    def nsn(self, effi, zlim, rateInterp):

        zplot = list(np.arange(self.zmin, self.zmax, 0.001))
        # interpolate efficiencies vs z
        effiInterp = interp1d(
            effi['z'], effi['effi'], kind='linear', bounds_error=False, fill_value=0.)
        # estimate the cumulated number of SN vs z
        nsn_cum = np.cumsum(effiInterp(zplot)*rateInterp(zplot))
        nsn_interp = interp1d(zplot, nsn_cum)

        return np.asscalar(nsn_interp(zlim))

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

        nproc = 1
        groups = tab.group_by(['season', 'z', 'pixRa', 'pixDec', 'healpixID'])

        if self.verbose:
            print('groups', len(groups))

        if nproc == 1:
            restab = CalcSN_df(groups.groups[:]).sn
            return restab

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

    def effiObsdf(self, data):

        df = data.loc[lambda dfa:  np.sqrt(dfa['Cov_colorcolor']) < 100000., :]

        df_sel = df.loc[lambda dfa:  np.sqrt(dfa['Cov_colorcolor']) < 0.04, :]

        group = df.groupby('z')

        group_sel = df_sel.groupby('z')

        rb = (group_sel.size()/group.size())
        err = np.sqrt(rb*(1.-rb)/group.size())

        rb = rb.array
        err = err.array

        rb[np.isnan(rb)] = 0.
        err[np.isnan(err)] = 0.

        return pd.DataFrame({group.keys: list(group.groups.keys()),
                             'effi': rb,
                             'effi_err': err})

    def run_season_slice(self, obs, gen_par):

        time_ref = time.time()

        # LC estimation

        sn_tot = None
        for key, vals in self.lcFast.items():
            gen_par_cp = np.copy(gen_par)
            if key == (-2.0, 0.2):
                idx = gen_par_cp['z'] < 0.9
                gen_par_cp = gen_par_cp[idx]
            lc = vals(obs, -1, gen_par_cp, bands='grizy')
            if self.verbose:
                print('End of simulation', time.time()-time_ref)

            # estimate SN

            sn = None
            if len(lc) > 0:
                sn = self.process(Table.from_pandas(lc))
            else:
                print('no LC here', len(obs), np.unique(obs['season']))

            if self.verbose:
                print('End of supernova', time.time()-time_ref)

            if sn is not None:
                if sn_tot is None:
                    sn_tot = sn
                else:
                    sn_tot = np.concatenate((sn_tot, sn))

        return sn_tot

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
