import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy.lib.recfunctions as rf
import healpy as hp
import pandas as pd


class Lims:

    """
    class to handle light curve of SN

    Parameters
    ---------------

    Li_files : str
       light curve reference file
    mag_to_flux_files : str
       files of magnitude to flux
    band : str
        band considered
    SNR : float
        Signal-To-Noise Ratio cut
    mag_range : pair(float),opt
        mag range considered
        Default : (23., 27.5)
    dt_range : pair(float)
        difference time range considered (cadence)
        Default : (0.5, 25.)
    """

    def __init__(self, Li_files,
                 mag_to_flux_files,
                 band, SNR,
                 mag_range=(23., 27.5),
                 dt_range=(0.5, 25.)):

        self.band = band
        self.SNR = SNR
        self.lims = []
        self.mag_to_flux = []
        self.mag_range = mag_range
        self.dt_range = dt_range

        for val in Li_files:
            self.lims.append(self.getLims(self.band, np.load(val), SNR))
        for val in mag_to_flux_files:
            self.mag_to_flux.append(np.load(val))
        self.interp()

    def getLims(self, band, tab, SNR):
        """
        Estimations of the limits

        Parameters
        ---------------

        band : str
          band to consider
        tab : numpy array
          table of data
        SNR : float
           Signal-to-Noise Ratio cut

        Returns
        -----------
        dict of limits with redshift and band as keys.
        """

        lims = {}

        for z in np.unique(tab['z']):

            idx = tab['z'] == z
            idx &= (tab['band'] == 'LSST::'+band)
            idx &= (tab['flux_e'] > 0.)
            sel = tab[idx]

            if len(sel) > 0:
                Li2 = np.sqrt(np.sum(sel['flux_e']**2))
                lim = 5. * Li2 / SNR
                if z not in lims.keys():
                    lims[z] = {}
                lims[z][band] = lim

        return lims

    def mesh(self, mag_to_flux):
        """
        Mesh grid to estimate five-sigma depth values (m5) from mags input.

        Parameters
        ---------------

        mag_to_flux : magnitude to flux values


        Returns
        -----------
        m5 values
        time difference dt (cadence)
        metric=sqrt(dt)*F5 where F5 is the 5-sigma flux
        """
        dt = np.linspace(self.dt_range[0], self.dt_range[1], 100)
        m5 = np.linspace(self.mag_range[0], self.mag_range[1], 50)
        ida = mag_to_flux['band'] == self.band
        fa = interpolate.interp1d(
            mag_to_flux[ida]['m5'], mag_to_flux[ida]['flux_e'], bounds_error=False, fill_value=0.0)
        f5 = fa(m5)
        F5, DT = np.meshgrid(f5, dt)
        M5, DT = np.meshgrid(m5, dt)
        metric = np.sqrt(DT) * F5

        return M5, DT, metric

    def interp(self):
        """
        Estimate a grid of interpolated values
        in the plane (m5, cadence, metric)

        Parameters
        ---------------
        None

        """

        M5_all = []
        DT_all = []
        metric_all = []

        for val in self.mag_to_flux:
            M5, DT, metric = self.mesh(val)
            M5_all.append(M5)
            DT_all.append(DT)
            metric_all.append(metric)

        sorted_keys = []
        for i in range(len(self.lims)):
            sorted_keys.append(np.sort([k for k in self.lims[i].keys()])[::-1])
        figa, axa = plt.subplots()
        self.Points_Ref = []
        for kk, lim in enumerate(self.lims):
            fmt = {}
            ll = [lim[zz][self.band] for zz in sorted_keys[kk]]
            cs = axa.contour(M5_all[kk], DT_all[kk], metric_all[kk], ll)

            points_values = None
            for io, col in enumerate(cs.collections):
                if col.get_segments():

                    myarray = col.get_segments()[0]
                    res = np.array(myarray[:, 0], dtype=[('m5', 'f8')])
                    res = rf.append_fields(res, 'cadence', myarray[:, 1])
                    res = rf.append_fields(
                        res, 'z', [sorted_keys[kk][io]]*len(res))
                    if points_values is None:
                        points_values = res
                    else:
                        points_values = np.concatenate((points_values, res))
            self.Points_Ref.append(points_values)

        plt.close(figa)  # do not display

    def interpGriddata(self, index, data, m5_str='m5_mean', cadence_str='cadence_mean'):
        """
        Estimate metric interpolation for data (m5,cadence)

        Parameters
        ---------------

        data : data where interpolation has to be done (m5,cadence)

        Returns
        -----------
        griddata interpolation (m5,cadence,metric)

        """

        ref_points = self.Points_Ref[index]
        res = interpolate.griddata((ref_points['m5'], ref_points['cadence']), ref_points['z'], (
            data[m5_str], data[cadence_str]), method='cubic')
        return res

    def plotCadenceMetric(self, restot,
                          target={'g': (26.91, 3.),  # was 25.37
                                  'r': (26.5, 3.),  # was 26.43
                                  # was 25.37      # could be 25.3 (400-s)
                                  'i': (26.16, 3.),
                                  # was 24.68      # could be 25.1 (1000-s)
                                  'z': (25.56, 3.),
                                  'y': (24.68, 3.)},  # was 24.72
                          dbName='', saveFig=False):
        """ Plot the cadence metric in the plane: median cadence vs m5

        Parameters
        --------------
        restot : array
          array of observations containing at least the following fields:
          m5_mean : mean five-sigma depth value (par season and per band)
          cadence_mean : mean cadence (per season and per band)
        target: dict
          Values corresponding to targets


        Returns
        ---------
        None

        """

        M5_all = []
        DT_all = []
        metric_all = []

        for val in self.mag_to_flux:
            M5, DT, metric = self.mesh(val)
            M5_all.append(M5)
            DT_all.append(DT)
            metric_all.append(metric)

        sorted_keys = []
        for i in range(len(self.lims)):
            sorted_keys.append(np.sort([k for k in self.lims[i].keys()])[::-1])

        plt.figure(figsize=(8, 6))
        plt.imshow(metric, extent=(
            self.mag_range[0], self.mag_range[1], self.dt_range[0], self.dt_range[1]), aspect='auto', alpha=0.25)

        plt.plot(restot['m5_mean'], restot['cadence_mean'], 'r+', alpha=0.9)

        color = ['k', 'b']
        for kk, lim in enumerate(self.lims):
            fmt = {}
            ll = [lim[zz][self.band] for zz in sorted_keys[kk]]
            cs = plt.contour(M5_all[kk], DT_all[kk],
                             metric_all[kk], ll, colors=color[kk])
            strs = ['$z=%3.1f$' % zz for zz in sorted_keys[kk]]
            for l, s in zip(cs.levels, strs):
                fmt[l] = s
            plt.clabel(cs, inline=True, fmt=fmt,
                       fontsize=16, use_clabeltext=True)

        t = target.get(self.band, None)
        if t is not None:
            plt.plot(t[0], t[1],
                     color='r', marker='*',
                     markersize=15)
        plt.xlabel('$m_{5\sigma}$', fontsize=18)
        plt.ylabel(r'Observer frame cadence $^{-1}$ [days]', fontsize=18)
        #plt.title('$%s$' % self.band.split(':')[-1], fontsize=18)
        plt.title('{} band'.format(self.band.split(':')[-1]), fontsize=18)
        plt.xlim(self.mag_range)
        plt.ylim(self.dt_range)
        plt.grid(1)
        if saveFig:
            plt.savefig('{}_{}_cad_vs_m5.png'.format(dbName, self.band))

    def plotHistzlim(self, names_ref, restot, dbName, saveFig):
        """
        Plot histogram of redshift limits

        Parameters
        --------------
        name_ref : list(str)
          name of the simulator used to produce the reference files
        restot : array


        """
        r = []
        fontsize = 15
        colors = dict(zip(range(0, 4), ['r', 'k', 'b', 'g']))

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(self.band + ' band', fontsize=fontsize)
        label = []
        xminv = []
        xmaxv = []

        for j, name in enumerate(names_ref):
            # remove point with zlim<0
            idx = restot['zlim_'+name] > 0.
            sel = restot[idx]
            xminv.append(np.min(sel['zlim_'+name]))
            xmaxv.append(np.max(sel['zlim_'+name]))

        xmin = np.min(xminv)
        xmax = np.max(xmaxv)
        xstep = 0.025
        bins = np.arange(xmin, xmax+xstep, xstep)

        for j, name in enumerate(names_ref):
            label.append(
                name + '  $z_{med}$ = ' + str(np.median(np.round(restot['zlim_'+name], 2))))

            ax.hist(restot['zlim_'+name], range=[xmin, xmax],
                    bins=bins, histtype='step', color=colors[j], linewidth=2)

        ax.set_xlabel('$z_{lim}$', fontsize=fontsize)
        ax.set_ylabel(r'Number of Entries', fontsize=fontsize)
        # ax.set_xticks(np.arange(0.5,0.75,0.05))
        ax.tick_params(labelsize=fontsize)
        ax.grid()
        plt.legend(label, fontsize=fontsize-2., loc='upper left')
        # plt.grid(1)
        if saveFig:
            plt.savefig('{}_{}_histzlim.png'.format(dbName, self.band))


def plotCadence(band, Li_files, mag_to_flux_files, SNR, metricValues, names_ref, mag_range, dt_range, dbName, saveFig=False, display=True, m5_str='m5_mean', cadence_str='cadence_mean'):
    """
    Main cadence plot
    Will display two plots: cadence plot and histogram of redshift limits

    Parameters
    --------------
    band : str
      band considered
    Li_files : str
       light curve reference file
    mag_to_flux_files : str
       files of magnitude to flux
    band : str
        band considered
    SNR : float
        Signal-To-Noise Ratio cut
    metricValues:
        values for the metric
    names_ref : list(str)
        name of the simulator used to generate reference files
    mag_range : pair(float)
        mag range considered
    dt_range : pair(float)
        difference time range considered (cadence)
    dbName : str
        cadence name
    saveFig : bool, opt
      to save the figures on disk or not.
    m5_str : str, opt
      m5 field name to use for computation
    cadence_str : str, opt
      cadence field name to use for computation


    Returns
    ---------
    restot : array with the following fields:
    fieldRA : right ascension (float)
    fieldDec : declination (float)
    season : season num (float)
    band : band (str)
    m5_str : m5 value (float)
    cadence_str : cadence value(float)
    zlim_name_ref : redshift limit for name_ref (float)

    """

    """
    if not isinstance(metricValues, np.ndarray):
        if len(metricValues) >= 1:
            for val in metricValues:
                res = np.concatenate(val)
            res = np.unique(res)
    else:
        res = metricValues
    """
    res = metricValues
    lim_sn = Lims(Li_files, mag_to_flux_files,
                  band, SNR, mag_range=mag_range,
                  dt_range=dt_range)

    if display:
        lim_sn.plotCadenceMetric(res, dbName=dbName, saveFig=saveFig)

    restot = None

    idx = (res[m5_str] >= mag_range[0]) & (
        res[m5_str] <= mag_range[1])
    idx &= (res[cadence_str] >= dt_range[0]) & (
        res[cadence_str] <= dt_range[1])
    res = res[idx]

    if len(res) > 0:
        resu = np.copy(res)
        for io, interp in enumerate(names_ref):
            zlims = lim_sn.interpGriddata(
                io, res, m5_str=m5_str, cadence_str=cadence_str)
            zlims[np.isnan(zlims)] = -1
            resu = rf.append_fields(resu, 'zlim_'+names_ref[io], zlims)
        if restot is None:
            restot = resu
        else:
            restot = np.concatenate((restot, resu))

        if display:
            lim_sn.plotHistzlim(names_ref, restot, dbName, saveFig)

    return restot


def plotMollview(nside, tab, xval, legx, unitx, minx, maxx, band, dbName, saveFig=False, seasons=-1, type='mollview', fieldzoom=None, healpixId='healpixID'):
    """
    Mollweid or Cart view

    Parameters
    --------------

    nside: int
     healpix nside parameter
    tab: array
     array of values
    xval: str
     name of the variable to display
    legx: str
     legend to display
    unitx: str
     unit of the legend
    minx: float
     min value for display
    maxx: float
     max value for display
    band: str
     band considered
    dbName: str
     name of the cadence file
    saveFig: bool, opt
     To save figure or not (default: False)
    seasons: int
     list of seasons to display (-1 = all)
    type: str, opt
     type of display: mollview (default) or cartview
    fieldzoom: bool, opt
     to make a zoom centered on (fieldzoom['RA'],fieldzoom['Dec'])
     (default: None)
    healpixId: str, opt
     string to identify healpixId (default: healpixID)

    Returns
    ---------
    None

    """

    # print(tab.dtype)
    if seasons == -1:
        plotViewIndiv(nside, tab, xval, legx, unitx,
                      minx, maxx, band,
                      dbName, saveFig,
                      season=seasons, type=type, fieldzoom=fieldzoom,
                      healpixId=healpixId)
    else:
        for season in seasons:
            idx = tab['season'] == season
            sel = tab[idx]
            plotViewIndiv(nside, sel, xval, legx, unitx,
                          minx, maxx, band, dbName,
                          saveFig, season=season, type=type,
                          fieldzoom=fieldzoom,
                          healpixId=healpixId)


def plotViewIndiv(nside, tab, xval, legx, unitx, minx, maxx, band, dbName, saveFig, season=-1, type='mollview', fieldzoom=None, healpixId='healpixID'):
    """
    Mollweid or Cart view

    Parameters
    --------------

    nside: int
     healpix nside parameter
    tab: array
     array of values
    xval: str
     name of the variable to display
    legx: str
     legend to display
    unitx: str
     unit of the legend
    minx: float
     min value for display
    maxx: float
     max value for display
    band: str
     band considered
    dbName: str
     name of the cadence file
    saveFig: bool, opt
     To save figure or not (default: False)
    seasons: int
     list of seasons to display (-1 = all)
    type: str, opt
     type of display: mollview (default) or cartview
    fieldzoom: bool, opt
     to make a zoom centered on (fieldzoom['RA'],fieldzoom['Dec'])
     (default: None)    
    healpixId: str, opt
     string to identify healpixId (default: healpixID)

    Returns
    ---------
    None

    """

    fig, ax = plt.subplots(figsize=(8, 6))
    # print(xval,tab[xval])
    med = np.median(tab[xval])
    # print(med)
    if band != '':
        leg = '{} band - {} \n {}: {} {}'.format(
            band, 'season {}'.format(season), legx, np.round(med, 1), unitx)
        if season == -1:
            leg = '{} band - {} \n {}: {} {}'.format(
                band, 'all seasons', legx, np.round(med, 1), unitx)
    else:
        leg = '{} \n {}: {} {}'.format('season {}'.format(
            season), legx, np.round(med, 1), unitx)
        if season == -1:
            leg = '{} \n {}: {} {}'.format(
                'all seasons', legx, np.round(med, 1), unitx)

    npix = hp.nside2npix(nside=nside)
    cmap = plt.cm.jet

    cmap.set_under('w')
    # cmap.set_bad('w')

    hpxmap = np.zeros(npix, dtype=np.float)
    """
    if season > 0.:
        hpxmap[tab['healpixID'].astype(int)] = tab[xval]
    else:
    """
    r = []
    for healpixID in np.unique(tab[healpixId]):
        ii = tab[healpixId] == healpixID
        sel = tab[ii]
        r.append((healpixID, np.median(sel[xval])))
    rt = np.rec.fromrecords(r, names=['healpixID', xval])
    hpxmap[rt['healpixID'].astype(int)] = rt[xval]

    plt.axes(ax)
    if type == 'mollview':
        hp.mollview(hpxmap, min=minx, max=maxx, cmap=cmap,
                    title=leg, nest=True, hold=True)
        hp.graticule()
    if type == 'cartview':
        #fig,ax = plt.subplots()
        lonra = [-180., 180.]
        latra = [-90., 90.]
        if fieldzoom is not None:
            RA = fieldzoom['RA'][0]
            Dec = fieldzoom['Dec'][0]
            lonra = [RA-5., RA+5.]
            latra = [Dec-5., Dec+5.]
        hp.cartview(hpxmap, min=minx, max=maxx, cmap=cmap,
                    title=leg, nest=True, lonra=lonra, latra=latra)

        #test = hp.cartview(hpxmap, return_projected_map=True,cmap=cmap, title=leg, nest=True,hold=True,lonra=lonra,latra=latra)

        #test = hp.cartview(hpxmap, return_projected_map=True,nest=True,lonra=lonra,latra=latra)
        #ax.imshow(test, origin='lower',extent=(lonra[0],lonra[1],latra[0],latra[1]), interpolation = 'none',cmap=cmap,vmin=10.)
        # hp.graticule()
        # plt.xlim([lonra[0],lonra[1]])

    if saveFig:
        plt.savefig('{}_{}_{}_mollview.png'.format(dbName, legx, band))


def plotDD(tab, cadenceName, what, ax, marker, color, mfc):
    """
    plot display for DD

    Parameters
    --------------

    tab: array
     array of values
    cadenceName: str
     name of the cadence considered
    what: str
     name of the variable to display
    ax: 
     axis for the display
    marker: str
     marker type for the display
    color: str
     marker color for the display
    mfc: str
     marker mfc for the display
    """

    # ax.plot(tab['fieldnum'], tab[what],
    #        marker=marker, color=color, linestyle='None', mfc=mfc, label=cadenceName, ms=10)

    """
    if type == 'barh':
        ax.barh(tab['fieldnum'], tab[what])
    """


def plotDDLoop(nside, dbNames, tabtot,
               what, legx,
               markers, colors, mfc,
               adjl, fields, figleg,
               type='plot'):
    """
    Loop of plots for DDF

    Parameters
    --------------

    nside: int
     healpix nside parameter
    dbNames: list(str)
      list of cadences to consider
    tabtot: array
     array of values
    what: str
     name of the variable to display
    legx: str
     legend to display
    markers: list(str)
     marker types for the display
    colors: list(str)
     color marker for the display   
    mfc: list(str)
     marker face color for the display
    adjl: int
     to adjust cadence names on the same length
    fields: array
     fields to display
    figleg: str
     legend of the figure

    Returns
    ---------
    None

    """

    fontsize = 15
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(figleg, fontsize=fontsize)
    fig.subplots_adjust(right=0.8)

    dfa = pd.DataFrame(np.copy(tabtot))

    df = dfa.groupby(['cadence', 'fieldname']).median().reset_index()

    for io, cadenceName in enumerate(dbNames):
        print('hello', cadenceName.ljust(adjl))
        idx = (df['cadence'] == cadenceName.ljust(
            adjl)) & (df['nside'] == nside)
        #idx = (tabtot['cadence'] == cadenceName)
        #idx &= (tabtot['nside'] == nside)

        #plotDD(df[idx],'_'.join(cadenceName.split('_')[:2]), what, ax, markers[io],colors[io],mfc[io])
        tab = df[idx]
        ax.plot(tab['fieldnum'], tab[what],
                marker=markers[io], color=colors[io], linestyle='None', mfc=mfc[io], label=cadenceName, ms=10)
    ax.set_ylabel(legx, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    #fields = getFields(0.)
    plt.xticks(fields['fieldnum'], fields['name'], fontsize=fontsize)
    # plt.legend(fontsize=fontsize)
    ax.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8),
              ncol=1, fancybox=True, shadow=True, fontsize=10)


def plotDDCadence_barh(tabtot,
                     what, legx,bands=['all'],fields=['COSMOS'],scale=1):
    """
    Loop of plots for DDF

    Parameters
    ----------------

    tabtot: array
     array of values
    what: str
     name of the variable to display
    legx: str
     legend to display
    bands: str list, opt
     list of the bands to show (default: ['all'])
    fields: str list, opt
     list of Fields to display (default: ['COSMOS'])
    scale: float, opt
     scale to apply for display results

    Returns
    -----------
    None

    """

    fontsize = 15

    dfa = pd.DataFrame(np.copy(tabtot))

    df = dfa.groupby(['cadence', 'fieldname','filter']).median().reset_index()

    df['cadence'] = df['cadence'].map(lambda x: '_'.join(x.split('_')[:4]))

    for fieldname in fields:
        idf = df['fieldname'] == fieldname
        sela = df[idf]
        for b in bands:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.suptitle('{} - {} band'.format(fieldname,b))
            idfb = sela['filter'] == b
            sel = sela[idfb]
            sel = sel.sort_values(by=what)
            ax.barh(sel['cadence'], sel[what]/scale)
            ax.set_xlabel(r'{}'.format(legx), fontsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize-5.)
        

def plotDDCorrel(tab, cadenceName, whatx, whaty, ax, marker, color, mfc):
    """
    plot display for DD correlation

    Parameters
    ----------------

    tab: array
     array of values
    cadenceName: str
     name of the cadence considered
    whatx: str
     name of the 1rst variable to display
    whaty: str
     name of the 2nd variable to display
    ax: 
     axis for the display
    marker: str
     marker type for the display
    color: str
     marker color for the display
    mfc: str
     marker mfc for the display
    """
    ax.plot(tab[whatx], tab[whaty],
            marker=marker, color=color, linestyle='None', mfc=mfc, label=cadenceName, ms=10)


def plotDDFit(tabtot, whatx, whaty, whatz='', whatzb=''):
    # linear fit
    figb, axb = plt.subplots()

    idx = (tabtot[whatx] > 0.) & (tabtot[whaty] > 0.)
    tabfit = tabtot[idx]
    if whatz == '':
        xfit = tabfit[whatx]
        yfit = tabfit[whaty]
    else:
        yfit = tabfit[whatx]/tabfit[whaty]
        xfit = tabfit[whatz]/tabfit[whatzb]

    axb.plot(xfit, yfit, 'ko')

    fit = np.polyfit(xfit, yfit, 1)
    p = np.poly1d(fit)

    xrange = np.arange(np.min(xfit), np.max(yfit), 0.01)

    axb.plot(xrange, p(xrange), color='r')

    figc, axc = plt.subplots()

    # axc.plot(tabfit[whatx],p(tabfit[whatx])/tabfit[whaty],'ro')
    if whatz == '':
        axc.hist(p(xfit)/yfit, histtype='step')
    else:
        axc.plot(xfit, p(xfit)/yfit, 'bo')
    n, bins = np.histogram(p(xfit)/yfit)
    mids = 0.5*(bins[1:] + bins[:-1])
    mean = np.average(mids, weights=n)

    var = np.average((mids - mean)**2, weights=n)

    print('mean and var', mean, np.sqrt(var))


def plotDDLoopCorrel(nside, dbNames, tabtot,
                     whatx, whaty, legx, legy,
                     markers, colors, mfc,
                     adjl, fields, figleg):
    """
    Loop of plots for DDF correlation

    Parameters
    --------------

    nside: int
     healpix nside parameter
    dbNames: list(str)
      list of cadences to consider
    tabtot: array
     array of values
    whatx: str
     name of the 1rst variable to display
    whaty: str
     name of the 2nd variable to display      
    legx: str
     legend to display (1rst var)
    legy: str
     legend to display (2nd var) 
    markers: list(str)
     marker types for the display
    colors: list(str)
     color marker for the display   
    mfc: list(str)
     marker face color for the display
    adjl: int
     to adjust cadence names on the same length
    fields: array
     fields to display
    figleg: str
     legend of the figure

    Returns
    ---------
    None

    """

    fontsize = 20
    fig, ax = plt.subplots()
    fig.suptitle(figleg, fontsize=fontsize)
    for io, cadenceName in enumerate(dbNames):
        idx = (tabtot['cadence'] == cadenceName.ljust(
            adjl)) & (tabtot['nside'] == nside)

        plotDDCorrel(tabtot[idx], cadenceName, whatx, whaty,
                     ax, markers[io], colors[io], mfc[io])

    ax.set_xlabel(legx, fontsize=fontsize)
    ax.set_ylabel(legy, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    #fields = getFields(0.)
    #plt.xticks(fields['fieldnum'], fields['fieldname'], fontsize=fontsize)
    # plt.legend(fontsize=fontsize)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.8, 1.15),
    #          ncol=1, fancybox=True, shadow=True, fontsize=15)


def plotDDCadence(tab, dbName, what, legx, adjl, fields):
    """
    DDF cadence plot

    Parameters
    --------------

    tab: array
     array of values
    dbNames: list(str)
      list of cadences to consider
    what: str
     name of the variable to display
    legx: str
     legend for the variable to display
    adjl: int
     to adjust cadence names on the same length
    fields: array
     fields to display

    """

    fcolors = 'cgyrm'
    fmarkers = ['o', '*', 's', 'v', '^']
    fmfc = ['None']*len(fcolors)
    fontsize = 15.
    fig, ax = plt.subplots()
    fig.suptitle('{}'.format(dbName))
    fig.subplots_adjust(right=0.85)
    # select data for this cadence
    ia = tab['cadence'] == dbName
    sela = tab[ia]
    # loop on bands and plot
    for io, band in enumerate('grizy'):
        idx = sela['filter'] == band
        selb = sela[idx]
        ax.plot(selb['fieldnum'], selb[what],
                marker=fmarkers[io], color=fcolors[io], linestyle='None', mfc=fmfc[io], label='{}'.format(band), ms=10)

    plt.xticks(fields['fieldnum'], fields['name'], fontsize=fontsize)
    # plt.legend(fontsize=fontsize)
    ax.legend(loc='upper center', bbox_to_anchor=(1.1, 0.7),
              ncol=1, fancybox=True, shadow=True, fontsize=15)

    ax.set_ylabel(legx, fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
