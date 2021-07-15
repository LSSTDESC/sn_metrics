import numpy as np
import matplotlib.pyplot as plt
from sn_tools.sn_calcFast import CovColor
import pandas as pd
from astropy.table import Table, vstack
import os

filtercolors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))
"""
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['figure.titlesize'] = 22
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['font.size'] = 22
"""
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['font.size'] = 15
#plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.family'] = 'Arial'
#plt.rcParams['font.sans-serif'] = ['Helvetica']


class Plot_NSN_metric:
    """
    class to plot nsn metric estimations

    """

    def __init__(self, snrmin, n_bef, n_aft, n_phase_min, n_phase_max, errmodrel, mjdCol, m5Col, filterCol,figdir='figures_nsn', templateLC={},dbName=''):

        self.snrmin = snrmin
        self.n_bef = n_bef
        self.n_aft = n_aft
        self.n_phase_min = n_phase_min
        self.n_phase_max = n_phase_max
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.errmodrel = errmodrel

        self.figdir = '{}/{}'.format(figdir, dbName)
        if not os.path.exists(self.figdir):
            os.makedirs(self.figdir)

        self.nfig = -1

        self.template_LC = {}
        if templateLC:
            self.templateLC = templateLC
            
        
    def plotLoop(self, healpixID, season, obs, lc, gen_par, x1=-2.0, color=0.2):
        """
        Method to loop on LC and plot results

        Parameters
        --------------
        season: int
          season of observation
        obs: array
          observations
        lc: array
          SN light curves
        gen_par: array
          parameter for simulation

        """

        # Loop on z and daymax
        print(lc.columns)
        print(lc)
        idm = np.abs(lc['x1']-x1) < 1.e-5
        idm &= np.abs(lc['color']-color) < 1.e-5
        idm &= lc['snr_m5'] >= self.snrmin
        lc = lc[idm]
        lc['flux_e_sec_err'] = lc['flux_e_sec']/lc['snr_m5']
        #filt_layout = dict(zip('griz', [(1, 0), (1, 1), (2, 0), (2, 1)]))
        filt_layout = dict(zip('griz', [(0, 1), (0, 2), (1, 1), (1, 2)]))
        print(gen_par.dtype)
    
        ifig = -1
        effi_z = None
        for zref in np.unique(lc['z']):
            idxa = np.abs(lc['z']-zref) < 1.e-5
            lca = lc[idxa]
            idg = np.abs(gen_par['z']-zref) < 1.e-5
            gen = gen_par[idg]
            #select template LC here
            lcref = None
            if self.templateLC:
                idtemp = np.abs(self.templateLC[(x1,color)]['z']-zref)<1.e-5
                lcref = self.templateLC[(x1,color)][idtemp]
            T0_min = np.min(gen['daymax'])
            T0_max = np.max(gen['daymax'])
            nlc = 0
            nsel = 0
            ra = []
            daymaxs = np.unique(lca['daymax'])
            for iday, daymax in enumerate(daymaxs):
                rb = []
                ifig += 1
                self.nfig += 1
                nlc += 1
                idxb = np.abs(lca['daymax']-daymax) < 1.e-5
                lcb = lca[idxb]
                fig = plt.figure(figsize=(15, 8),constrained_layout=True)
                fig.suptitle('healpixID: {} - season {}'.format(healpixID, season), fontsize='medium')
                gs = fig.add_gridspec(3,3)
                # plot observations
                self.plotObs(fig.add_subplot(gs[0,0]), obs, daymax, T0_min, T0_max, whatx=self.mjdCol, whaty=self.m5Col,
                             xlabel='MJD [day]', ylabel='5$\sigma$ depth [mag]')
                # plot lc
                for b in 'griz':
                    idx = lcb['band'] == 'LSST::{}'.format(b)
                    selb = lcb[idx]
                    ia = filt_layout[b][0]
                    ib = filt_layout[b][1]
                    lcrefb = None
                    if lcref:
                        idp = lcref['band'] ==  b
                        idp &= lcref['flux_e_sec'] > 0.001
                        lcrefb = lcref[idp]
                    self.plotLC_T0(fig.add_subplot(gs[ia, ib]), selb, b, daymax, lcrefb,whatx='time',
                                   whaty='flux_e_sec', yerr='flux_e_sec_err', xlabel='MJD [day]', ylabel='flux [e/s]', axtitle='')
                # get infos for selection
                selected = self.getSelection(lcb, daymax, zref)
                nsel += selected
                effi = nsel/nlc
                effi_err = np.sqrt(nsel*(1.-effi))/nlc
                ra.append((selected, daymax))
                resa = np.rec.fromrecords(ra, names=['sel', 'daymax'])
                rb.append((effi, effi_err, np.round(zref, 2)))
                resb = np.rec.fromrecords(rb, names=['effi', 'effi_err', 'z'])
                if effi_z is not None:
                    resb = np.concatenate((resb,effi_z))

                if iday == len(daymaxs)-1:
                    effi_z = np.array(resb)
                    
                #print(selected)
                self.plotSingle(fig.add_subplot(
                    gs[1, 0]), resa, varx='daymax', vary='sel', legx='T$_0$ [day]', legy='sel')
                self.plotSingle(fig.add_subplot(
                    gs[2, 0]), resb, varx='z', vary='effi', erry='effi_err', legx='z', legy='$\epsilon$')
                #plt.show()

                
                figname = '{}/healpix{}_{}.png'.format(self.figdir,healpixID, self.nfig)
                fig.savefig(figname)
                plt.close()
                
    def plotObs(self, ax, obs, daymax, T0_min, T0_max, whatx, whaty, xlabel, ylabel):
        """
        Method to plot observations

        Parameters
        ---------------
        ax: matplotlib axis
          axis for the plot
        obs: numpy array
            observations to plot
        daymax: float
           T0 LC
        T0_min: float
          min T0 value
        T0_max: float
          max T0 value
        whatx: str
           x-axis variable
        whaty: str
           y-axis variable
        xlabel: str
          x-axis label
        ylabel: str
          y-axis label
        """

        for b in np.unique(obs[self.filterCol]):
            idx = obs[self.filterCol] == b
            sel = obs[idx]
            ax.plot(sel[whatx], sel[whaty],
                    color=filtercolors[b[-1]], marker='o', label='{}'.format(b[-1]), ls='None')

        ax.plot([daymax]*2, [np.min(obs[whaty]),
                             np.max(obs[whaty])], ls='dashed', color='k')

        ax.plot([T0_min]*2, [np.min(obs[whaty]),
                             np.max(obs[whaty])], ls='solid', color='r')

        ax.plot([T0_max]*2, [np.min(obs[whaty]),
                             np.max(obs[whaty])], ls='solid', color='r')

        ax.legend(loc='upper left', bbox_to_anchor=(0., 1.20), ncol=4, fontsize=12, frameon=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def plotLC_T0(self, ax, sel, band, daymax, lcref=None, whatx='time', whaty='flux_e', yerr=None, xlabel='MJD [day]', ylabel='max flux pixel [e/s]', axtitle=''):
        """
        Method to plot a light curve corresponding to T0

        Parameters
        ---------------
        ax: matplotlib axis
          axis for the plot
        sel: array
          LC to plot
        daymax: float
           T0 LC
        whatx: str, opt
          x-variable to plot(default: time)
        whaty: str, opt
          y-variable to plot(default: flux_e)
        legx: str, opt
          x-axis label(default: 'MJD [day]')
        legy: str, opt
          y-axis label(default: 'max flux pixel[e/s])
        axtitle: str, opt
          axis title(default: '')
        """
        """
        idx = sel['snr_m5'] >= self.snr_min
        sel = sel[idx]
        for band in np.unique(sel['band']):
            ib = sel['band'] == band
            selb = sel[ib]
            ax.plot(selb[whatx], selb[whaty],
                    '{}o'.format(filtercolors[band[-1]]), label='{} band'.format(band[-1]))
        """
        if yerr is not None:
            yerr = sel[yerr]
        ax.errorbar(sel[whatx], sel[whaty], yerr=yerr, color=filtercolors[band[-1]],
                    marker='o', label='{} band'.format(band[-1]),ls='None')

        if lcref:
            ax.plot(lcref[whatx]+daymax,lcref[whaty], color=filtercolors[band[-1]])
        
        # ax.legend()
        tmin, tmax = np.min(sel[whatx]), np.max(sel[whatx])
        fluxmin, fluxmax = np.min(lcref[whaty]), np.max(lcref[whaty])
        ax.plot([daymax]*2, [fluxmin, fluxmax], color='k', ls='solid')
        """
        ax.plot([daymax-5.]*2,
                [fluxmin, fluxmax], color='k', ls='dashed')
        ax.plot([daymax+5.]*2,
                [fluxmin, fluxmax], color='k', ls='dashed')
        """
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(axtitle, loc='right', color='b')

    def getSelection(self, lcm, daymax, zref):
        """"
        Method to extract infos from ls

        Parameters
        ---------------
        lcm: array
          light curve of interest

        """

        idx = lcm['band'] != 'LSST::y'
        idx &= lcm['band'] != 'LSST::u'

        lc = lcm[idx]
        # remove points with too high errormodel

        if self.errmodrel > 0.:
            lc = self.select_error_model(lc)

        # estimate the number of lc points with min and max phase
        idx = lc['phase'] <= -5
        n_phase_min = len(lc[idx])
        idx = lc['phase'] >= 20
        n_phase_max = len(lc[idx])

        # estimate the number of LC epochs before and after daymax
        lc = lc.sort_values(by=['night'])
        idx = lc['phase'] <= 0
        n_bef = len(np.unique(lc[idx]['night']))

        idx = lc['phase'] >= 0
        n_aft = len(np.unique(lc[idx]['night']))

        var_color = CovColor(pd.DataFrame(lc).sum()).Cov_colorcolor

        # print('sel', n_bef, n_aft, n_phase_min,
        #      n_phase_max, np.sqrt(var_color), daymax, zref)
        if n_bef < self.n_bef:
            return 0
        if n_aft < self.n_aft:
            return 0
        if n_phase_min < self.n_phase_min:
            return 0
        if n_phase_max < self.n_phase_max:
            return 0
        if var_color > 0.04**2:
            return 0

        return 1

    def plotSingle(self, ax, tab, varx, vary, erry='', legx='', legy='', ls='solid', color='k', label=None):
        """
        Method to plot

        Parameters
        ---------------
        ax: matplotlib axis
        tab: numpy array
           data to plot
        varx: str
          x-axis var
        vary: str
          y-axis var
        erry: str, opt
          y-axis var error(default: '')
        legx: str, opt
          x-axis label(default: '')
        legy: str, opt
          y-axis label(default: '')
        ls: str, opt
          linestyle(default: solid)
        color: str, opt
          line color (default: k)
        label: str, opt
          label for plot (default: None)
        """

        if erry is not '':
            ax.errorbar(tab[varx], tab[vary],
                        yerr=tab[erry], color=color, marker='o', label=label)
        else:
            ax.plot(tab[varx], tab[vary], color=color,
                    marker='o', ls=ls, label=label)

        ax.set_xlabel(legx)
        ax.set_ylabel(legy, color=color)
        if label:
            ax.legend()

    def select_error_model(self, grp):
        """
        function to select LCs

        Parameters
        ---------------
        grp : pandas df
          lc to consider

        Returns
        ----------
        lc with filtered values (pandas df)

       """

        lc = Table.from_pandas(grp)
        if self.errmodrel < 0.:
            return lc.to_pandas()

        # first: select iyz bands

        bands_to_keep = []

        lc_sel = Table()
        for b in 'izy':
            bands_to_keep.append('LSST::{}'.format(b))
            idx = lc['band'] == 'LSST::{}'.format(b)
            lc_sel = vstack([lc_sel, lc[idx]])

        # now apply selection on g band for z>=0.25
        sel_g = self.sel_band(lc, 'g', 0.25)

        # now apply selection on r band for z>=0.6
        sel_r = self.sel_band(lc, 'r', 0.6)

        lc_sel = vstack([lc_sel, sel_g])
        lc_sel = vstack([lc_sel, sel_r])

        return lc_sel.to_pandas()

    def sel_band(self, tab, b, zref):
        """
        Method to perform selections depending on the band and z

        Parameters
        ---------------
        tab: astropy table
          lc to process
        b: str
          band to consider
        zref: float
           redshift below wiwh the cut wwill be applied

        Returns
        ----------
        selected lc
        """

        idx = tab['band'] == 'LSST::{}'.format(b)
        sel = tab[idx]

        if len(sel) == 0:
            return Table()

        ida = sel['z'] < zref

        idb = sel['z'] >= zref
        idb &= sel['fluxerr_model']/sel['flux'] <= self.errmodrel

        tabres = vstack([sel[idb], sel[ida]])

        return tabres

    def loadTemplate(self, x1, color,templateDir = 'Template_LC'):
        """
        Method to load template LC
        
        Parameters
        ---------------
        x1: float
          x1 SN
        color: float
          SN color

        Returns
        ----------
        pandas df of template LC


        """

        wave_cutoff = 'error_model'

        if not self.error_model:
            wave_cutoff = '{}_{}'.format(self.bluecutoff, self.redcutoff)
        lcname = 'LC_{}_{}_{}_ebvofMW_0.0_vstack.hdf5'.format(
            x1, color, wave_cutoff)
         # Load the file - lc reference
        lcFullName = '{}/{}'.format(templateDir, lcName)
        f = h5py.File(lcFullName, 'r')
        keys = list(f.keys())
        # lc_ref_tot = Table.read(filename, path=keys[0])
        lc_ref_tot = Table.from_pandas(pd.read_hdf(lcFullName))

        return lc_ref_tot
        
def plotNSN_effi(effi, vary, erry=None, legy='', ls='None'):
    """
    Simple method to plot vs z

    Parameters
    --------------
    effi: pandas df
          data to plot
    vary: str
      variable (column of effi) to plot
    erry: str, opt
      error on y-axis (default: None)
     legy: str, opt
       y-axis legend (default: '')

    """
    fig, ax = plt.subplots(figsize=(10, 6))

    grb = effi.groupby(['x1', 'color'])
    yerr = None
    for key, grp in grb:
        x1 = grp['x1'].unique()[0]
        color = grp['color'].unique()[0]
        if erry is not None:
            yerr = grp[erry]
        ax.errorbar(grp['z'], grp[vary], yerr=yerr,
                    marker='o', label='(x1,color)=({},{})'.format(x1, color), lineStyle=ls)

    ax.set_xlabel('z')
    ax.set_ylabel(legy)
    ax.legend()
    ax.grid()
    plt.show()


def plotNSN_cumul(grp, nsn_cum_norm, nsn_cum_norm_err, zplot, zlim_coeff, zlim, zmean, zpeak):
    """
    Method to plot the NSN cumulative vs redshift

    Parameters
    --------------
    grp: pandas group
    data to process

    """

    # First plot: cumulative vs z
    fig, ax = plt.subplots(figsize=(8, 6))
    x1 = grp['x1'].unique()[0]
    color = grp['color'].unique()[0]

    ax.plot(zplot, nsn_cum_norm,
            label='(x1,color)=({},{})'.format(x1, color), color='r')
    ax.fill_between(zplot, nsn_cum_norm-nsn_cum_norm_err,
                    nsn_cum_norm+nsn_cum_norm_err, color='y')

    plotzbar(ax, zlim, zlim_coeff, zmean, zpeak, 0., zlim_coeff)
    """
    ax.plot([zlim]*2, [0, zlim_coeff], ls='solid', color='b')

    zlimstr = '$z_{'+str(zlim_coeff)+'}$'
    ax.text(0.3, 0.6, '{} = {}'.format(zlimstr, np.round(zlim, 2)), color='b')
    """

    ax.set_ylabel('NSN ($z<$)')
    ax.set_xlabel('z')
    ax.set_xlim((0.0, 0.6))
    ax.set_ylim((0.0, 1.05))
    ax.plot([0., 1.2], [zlim_coeff, zlim_coeff],
            ls='--', color='k')
    ax.grid()
    ax.legend(loc='upper left')


def plotNSN_z(grp, zplot, nsn_z, zlim_coeff, zlim, zmean, zpeak):
    """
    Method to plot the NSN cumulative vs z

    Parameters
    --------------
    grp: pandas group
    data to process

    """

    fig, ax = plt.subplots(figsize=(10, 6))
    x1 = grp['x1'].unique()[0]
    color = grp['color'].unique()[0]

    ax.plot(zplot, nsn_z, ls='solid', color='r',
            label='(x1,color)=({},{})'.format(x1, color))
    ax.set_ylim(0., None)
    y_min, y_max = ax.get_ylim()
    plotzbar(ax, zlim, zlim_coeff, zmean, zpeak, y_min, y_max)

    ax.grid()
    ax.set_xlabel('$z$')
    ax.set_ylabel('N$_{SN}$')
    ax.legend(loc='upper right')
    plt.show()


def plotzbar(ax, zlim, zlim_coeff, zmean, zpeak, y_min, y_max):

    ymed = np.mean([y_min, y_max])
    # plot zmean bar
    ax.plot([zmean]*2, [y_min, y_max], ls='solid', color='m')
    zmeanstr = '$z_{mean}$'
    ax.text(0.4, ymed, '{} = {}'.format(zmeanstr, zmean), color='m')

    # plot zpeak bar
    ax.plot([zpeak]*2, [y_min, y_max], ls='solid', color='k')
    zpeakstr = '$z_{peak}$'
    ax.text(0.4, 0.8*ymed, '{} = {}'.format(zpeakstr, zpeak), color='k')

    # plot zlimit bar
    ax.plot([zlim]*2, [y_min, y_max], ls='solid', color='b')
    zlimstr = '$z_{'+str(zlim_coeff)+'}$'
    ax.text(0.4, 0.6*ymed, '{} = {}'.format(zlimstr, np.round(zlim, 2)), color='b')


class Plot_Saturation_Metric:
    """
    class to plot SN saturation metric - live

    Parameters
    ---------------
    healpixID: int
      healpixel ID
    season: int
      season number
    T0_min: float
      min daymax value
    T0_max: float
       max daymax value
    zref: redshift value
    """

    def __init__(self, healpixID, zref, snr_min, mjdCol, m5Col, filterCol, fullwell, saturationLevel,figdir='figures_saturation'):

        self.healpixID = healpixID
        self.zref = zref
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.snr_min = snr_min
        self.fullwell = fullwell
        self.saturationLevel = saturationLevel

        if not os.path.exists(figdir):
            os.mkdir(figdir)
        self.figdir = figdir
        self.nfig = -1
        
    def __call__(self, obs, lc, season, T0_min, T0_max):

        idx = np.abs(lc['z']-self.zref) < 1.e-5
        lc = lc[idx]
        lc.sort(order='daymax')
        daymaxs = np.unique(lc['daymax'])
        ndaymax = 0
        npeak = 0
        isat = 0
        r_nbef = []
        r_deltaT_befsat = []

        r = []
        for iday, daymax in enumerate(daymaxs):
            self.nfig += 1
            idx = np.abs(lc['daymax']-daymax) < 1.e-5
            idx &= lc['snr_m5'] >= self.snr_min
            sel = lc[idx]
            print('hhh', type(sel))

            lctab = Table(sel)
            lctab.meta['z'] = self.zref
            lctab.meta['daymax'] = daymax

            # fig, ax = plt.subplots(nrows=5, figsize=(
            #    8, 12), constrained_layout=False)
            fig = plt.figure(figsize=(15, 10), constrained_layout=True)
            gs = fig.add_gridspec(4, 2)
            fig.suptitle('healpixID: {} - season {}'.format(
                self.healpixID, season), fontsize='medium')

            # plot observation
            self.plotObs(fig.add_subplot(gs[:2, 0]), obs, daymax, T0_min, T0_max, whatx=self.mjdCol, whaty=self.m5Col,
                         xlabel='MJD [day]', ylabel='5$\sigma$ depth [mag]')
            ndaymax += 1

            print('here', len(sel), daymax)
            if len(sel) > 0:
                axtitle = '$z$={} - T$_0$={}'.format(
                    self.zref, np.round(daymax, 1))
                # plot LC
                self.plotLC_T0(fig.add_subplot(
                    gs[2:, 0]), sel, daymax, np.min(obs[self.mjdCol]), np.max(obs[self.mjdCol]), axtitle=axtitle)
                # get stat
                npeakobs, isatobs, nbef_sat, deltaT_befsat, deltaT_sat = self.statShape(
                    sel, daymax)
                npeak += npeakobs
                isat += isatobs
                if nbef_sat < 100.:
                    r_nbef.append(nbef_sat)
                if deltaT_befsat < 100.:
                    r_deltaT_befsat.append(deltaT_befsat)

            print('nbef', r_nbef, r_deltaT_befsat)
            r.append((npeak/ndaymax, isat/ndaymax, np.median(r_nbef),
                      np.median(r_deltaT_befsat), daymax))
            res = np.rec.fromrecords(
                r, names=['effipeak', 'probasat', 'nbef_sat', 'deltaT_befsat', 'daymax'])
            # ax[2].plot(res['daymax'], res['effipeak'], color='k', marker='o')
            # plot stat
            self.plotSingle(fig.add_subplot(gs[0, 1]), res, 'daymax', 'effipeak',
                            '', 'T$_0$ [day]', '$\epsilon_{peak}$', color='b')
            self.plotSingle(fig.add_subplot(gs[1, 1]), res, 'daymax', 'probasat',
                            '', 'T$_0$ [day]', 'Saturation proba.', color='r')
            self.plotSingle(fig.add_subplot(gs[2, 1]), res, 'daymax', 'nbef_sat',
                            '', 'T$_0$ [day]', 'N$_{LC}$ bef. sat.')
            self.plotSingle(fig.add_subplot(gs[3, 1]), res, 'daymax', 'deltaT_befsat',
                            '', 'T$_0$ [day]', '$\Delta$t  bef. sat. [day]')

            # plt.close()

            
            figname = '{}/healpix{}_{}.jpg'.format(self.figdir,
                                                   self.healpixID, self.nfig)
            plt.savefig(figname)
            plt.close()

            """
            plt.draw()
            plt.pause(2)
            plt.close()
            """
            # plt.show()

    def plotObs(self, ax, obs, daymax, T0_min, T0_max, whatx, whaty, xlabel, ylabel):
        """
        Method to plot observations

        Parameters
        ---------------
        ax: matplotlib axis
          axis for the plot
        obs: numpy array
            observations to plot
        daymax: float
           T0 LC
        T0_min: float
          min T0 value
        T0_max: float
          max T0 value
        whatx: str
           x-axis variable
        whaty: str
           y-axis variable
        xlabel: str
          x-axis label
        ylabel: str
          y-axis label
        """

        for b in np.unique(obs[self.filterCol]):
            idx = obs[self.filterCol] == b
            sel = obs[idx]
            ax.plot(sel[whatx], sel[whaty],
                    color=filtercolors[b[-1]], marker='o', label='{} band'.format(b[-1]), ls='None')

        ax.plot([daymax]*2, [np.min(obs[whaty]),
                             np.max(obs[whaty])], ls='dashed', color='k')

        ax.plot([T0_min]*2, [np.min(obs[whaty]),
                             np.max(obs[whaty])], ls='solid', color='r')

        ax.plot([T0_max]*2, [np.min(obs[whaty]),
                             np.max(obs[whaty])], ls='solid', color='r')

        ax.legend(loc='upper left', bbox_to_anchor=(
            0., 1.10), ncol=3, fontsize=15, frameon=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def plotLC_T0(self, ax, sel, daymax, tmin, tmax, whatx='time', whaty='flux_e', xlabel='MJD [day]', ylabel='max flux pixel [e/s]', axtitle=''):
        """
        Method to plot a light curve corresponding to T0

        Parameters
        ---------------
        ax: matplotlib axis
          axis for the plot
        sel: array
          LC to plot
        daymax: float
           T0 LC
        whatx: str, opt
          x-variable to plot(default: time)
        whaty: str, opt
          y-variable to plot(default: flux_e)
        legx: str, opt
          x-axis label(default: 'MJD [day]')
        legy: str, opt
          y-axis label(default: 'max flux pixel[e/s])
        axtitle: str, opt
          axis title(default: '')
        """
        idx = sel['snr_m5'] >= self.snr_min
        sel = sel[idx]
        for band in np.unique(sel['band']):
            ib = sel['band'] == band
            selb = sel[ib]
            ax.plot(selb[whatx], selb[whaty],
                    '{}o'.format(filtercolors[band[-1]]), label='{} band'.format(band[-1]))
        # ax.legend()
        #tmin, tmax = np.min(sel[whatx]), np.max(sel[whatx])
        fluxmin, fluxmax = np.min(sel[whaty]), np.max(sel[whaty])
        ax.plot([tmin, tmax], [self.fullwell]*2, color='k')
        ax.plot([daymax]*2, [fluxmin, fluxmax], color='k', ls='solid')
        ax.plot([daymax-5.]*2,
                [fluxmin, fluxmax], color='k', ls='dashed')
        ax.plot([daymax+5.]*2,
                [fluxmin, fluxmax], color='k', ls='dashed')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(axtitle, loc='right', color='b')

    def statShape(self, sel, daymax):
        """
        Method to estimate a set of features related to the shape of LC

        Parameters
        ---------------
        sel: array
          lc to process
        daymay: float
          T0 LC

        Returns
        -----------
        npeak: int
          1 if number of LC points in a 5 days window around peak is larger that 3, 0 otherwise
        nlc_bef_sat: int
          number of LC points before sat

        """
        print('hello', type(sel))
        sel.sort(order='time')
        idx = sel['snr_m5'] >= self.snr_min
        sel = sel[idx]

        # get mjdmin
        mjd_min = np.min(sel['time'])

        # get non sat points
        idnosat = sel['flux_e'] <= self.fullwell
        sel_nosat = sel[idnosat]

        npeak = 0
        # get peak measurement
        idpeak = np.abs(sel_nosat['time']-daymax) <= 5
        n_aroundpeak = len(sel_nosat[idpeak])
        if n_aroundpeak >= 3:
            npeak = 1

        seldf = pd.DataFrame(np.copy(sel))
        seldf['sat'] = 0
        idsat = seldf['flux_e'] > self.fullwell
        seldf.loc[idsat, 'sat'] = 1

        print(seldf[['night', 'flux_e', 'sat', 'time']])
        selsat = seldf.groupby(['night']).apply(lambda x: pd.DataFrame({'sat': [np.sum(x['sat'])/len(x)],
                                                                        'time': [np.median(x['time'])]})).reset_index()
        print(selsat)

        isat = 0
        nbef_sat = 999
        deltaT_sat = 999.
        deltaT_befsat = 999.

        idx = selsat['sat'] >= self.saturationLevel
        selsatb = selsat[idx]
        if len(selsatb) > 0:
            isat = 1
            time_sat = np.min(selsatb['time'])
            deltaT_sat = time_sat-mjd_min
            ido = sel['time'] < time_sat
            selnosat = sel[ido]
            if len(selnosat) > 0:
                nbef_sat = len(np.unique(selnosat['night']))
                deltaT_befsat = np.max(selnosat['time'])-mjd_min
            else:
                nbef_sat = 0
                deltaT_befsat = 0

        print(isat, nbef_sat, deltaT_befsat, deltaT_sat)
        return npeak, isat, nbef_sat, np.round(deltaT_befsat, 2), np.round(deltaT_sat, 2)

    def plotSingle(self, ax, tab, varx, vary, erry='', legx='', legy='', ls='solid', color='k', label=None):
        """
        Method to plot

        Parameters
        ---------------
        ax: matplotlib axis
        tab: numpy array
           data to plot
        varx: str
          x-axis var
        vary: str
          y-axis var
        erry: str, opt
          y-axis var error(default: '')
        legx: str, opt
          x-axis label(default: '')
        legy: str, opt
          y-axis label(default: '')
        ls: str, opt
          linestyle(default: solid)
        color: str, opt
          line color (default: k)
        label: str, opt
          label for plot (default: None)
        """

        if erry is not '':
            ax.errorbar(tab[varx], tab[vary],
                        yerr=tab[erry], color=color, marker='o', label=label)
        else:
            ax.plot(tab[varx], tab[vary], color=color,
                    marker='o', ls=ls, label=label)

        ax.set_xlabel(legx)
        ax.set_ylabel(legy, color=color)
        if label:
            ax.legend()

    
