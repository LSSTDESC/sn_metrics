import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy.lib.recfunctions as rf
import healpy as hp


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

    def __init__(self, Li_files, mag_to_flux_files, band, SNR,
                 mag_range=(23., 27.5), dt_range=(0.5, 25.)):

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

            idx = (tab['z'] == z) & (tab['band'] == 'LSST::'+band)
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
            mag_to_flux[ida]['m5'], mag_to_flux[ida]['flux_e'])
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

    def interpGriddata(self, index, data):
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
            data['m5_mean'], data['cadence_mean']), method='cubic')
        return res

    def plotCadenceMetric(self, restot,
                          target={  # 'g': (26.91, 3.), # was 25.37
                              'r': (26.5, 3.),  # was 26.43
                              # was 25.37      # could be 25.3 (400-s)
                              'i': (26.16, 3.),
                              # was 24.68      # could be 25.1 (1000-s)
                              'z': (25.56, 3.),
                              'y': (24.68, 3.)}):  # was 24.72
        """ Plot the cadence metric in the plane: median cadence vs m5

        Parameters
        --------------
        restot : array
          array of observations containing at least the following fields:
          m5_mean : mean five-sigma depth value (par season and per band)
          cadence_mean : mean cadence (per season and per band)

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
        plt.title('$%s$' % self.band.split(':')[-1], fontsize=18)
        plt.xlim(self.mag_range)
        plt.ylim(self.dt_range)
        plt.grid(1)

    def plotHistzlim(self, names_ref, restot):
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
            xminv.append(np.min(restot['zlim_'+name]))
            xmaxv.append(np.max(restot['zlim_'+name]))

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


def plotCadence(band, Li_files, mag_to_flux_files, SNR, metricValues, names_ref, mag_range, dt_range, display=True):
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

    Returns
    ---------
    restot : array with the following fields:
    fieldRA : right ascension (float)
    fieldDec : declination (float)
    season : season num (float)
    band : band (str)
    m5_mean : mean cadence (float)
    cadence_mean : mean cadence (float)
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
                  band, SNR, mag_range=mag_range, dt_range=dt_range)

    if display:
        lim_sn.plotCadenceMetric(res)

    restot = None

    idx = (res['m5_mean'] >= mag_range[0]) & (
        res['m5_mean'] <= mag_range[1])
    idx &= (res['cadence_mean'] >= dt_range[0]) & (
        res['cadence_mean'] <= dt_range[1])
    res = res[idx]

    if len(res) > 0:
        resu = np.copy(res)
        for io, interp in enumerate(names_ref):
            zlims = lim_sn.interpGriddata(io, res)
            zlims[np.isnan(zlims)] = -1
            resu = rf.append_fields(resu, 'zlim_'+names_ref[io], zlims)
        if restot is None:
            restot = resu
        else:
            restot = np.concatenate((restot, resu))

        if display:
            lim_sn.plotHistzlim(names_ref, restot)

    return restot


def plotMollview(nside, tab, xval, legx, unitx, minx, band, seasons=-1):

    print(tab.dtype)
    if seasons == -1:
        plotMollviewIndiv(nside, tab, xval, legx, unitx,
                          minx, band, season=seasons)
    else:
        for season in seasons:
            idx = tab['season'] == season
            sel = tab[idx]
            plotMollviewIndiv(nside, sel, xval, legx, unitx,
                              minx, band, season=seasons)


def plotMollviewIndiv(nside, tab, xval, legx, unitx, minx, band, season=-1):

    med = np.median(tab[xval])
    print(np.min(tab[xval]))
    leg = 'band {} - {} \n {}: {} {}'.format(
        band, 'season {}'.format(season), legx, np.round(med, 1), unitx)
    if season == -1:
        leg = 'band {} - {} \n {}: {} {}'.format(
            band, 'all seasons', legx, np.round(med, 1), unitx)

    npix = hp.nside2npix(nside=nside)
    cmap = plt.cm.jet

    cmap.set_under('w')
    # cmap.set_bad('w')

    hpxmap = np.zeros(npix, dtype=np.float)
    if season > 0.:
        hpxmap[tab['healpixID']] = tab[xval]
    else:
        r = []
        for healpixID in np.unique(tab['healpixID']):
            ii = tab['healpixID'] == healpixID
            sel = tab[ii]
            r.append((healpixID, np.median(sel[xval])))
        rt = np.rec.fromrecords(r, names=['healpixID', xval])
        hpxmap[rt['healpixID']] = rt[xval]
    hp.mollview(hpxmap, min=minx, cmap=cmap, title=leg, nest=True)
    hp.graticule()
