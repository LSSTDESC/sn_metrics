import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy.lib.recfunctions as rf


class Lims:
    """
    Test

    Parameters
    --------------
    ooo

    Returns
    ----------
    bbb

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
            self.lims.append(self.Get_Lims(self.band, np.load(val), SNR))
        for val in mag_to_flux_files:
            self.mag_to_flux.append(np.load(val))
        self.Interp()

    def Get_Lims(self, band, tab, SNR):

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

    def Mesh(self, mag_to_flux):
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

    def Interp(self):
        M5_all = []
        DT_all = []
        metric_all = []

        for val in self.mag_to_flux:
            M5, DT, metric = self.Mesh(val)
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

    def Interp_griddata(self, index, data):

        ref_points = self.Points_Ref[index]
        res = interpolate.griddata((ref_points['m5'], ref_points['cadence']), ref_points['z'], (
            data['m5_mean'], data['cadence_mean']), method='cubic')
        return res

    def Plot_Cadence_Metric(self, restot,
                            target={  # 'g': (26.91, 3.), # was 25.37
                                'r': (26.5, 3.),  # was 26.43
                                # was 25.37      # could be 25.3 (400-s)
                                'i': (26.16, 3.),
                                # was 24.68      # could be 25.1 (1000-s)
                                'z': (25.56, 3.),
                                'y': (24.68, 3.)}):  # was 24.72

        M5_all = []
        DT_all = []
        metric_all = []

        for val in self.mag_to_flux:
            M5, DT, metric = self.Mesh(val)
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

    def Plot_Hist_zlim(self, names_ref, restot):
        r = []
        fontsize = 15
        colors = dict(zip(range(0, 4), ['r', 'k', 'b', 'g']))

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(self.band + ' band', fontsize=fontsize)
        label = []
        xminv = []
        xmaxv = []
        print('alors', names_ref)
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


def Plot_Cadence(band, Li_files, mag_to_flux_files, SNR, metricValues, names_ref, mag_range, dt_range):

    if len(metricValues) > 1:
        res = np.concatenate(metricValues)
        res = np.unique(res)
    else:
        res = metricValues
    lim_sn = Lims(Li_files, mag_to_flux_files,
                  band, SNR, mag_range=mag_range, dt_range=dt_range)

    lim_sn.Plot_Cadence_Metric(res)

    restot = None

    idx = (res['m5_mean'] >= mag_range[0]) & (
        res['m5_mean'] <= mag_range[1])
    idx &= (res['cadence_mean'] >= dt_range[0]) & (
        res['cadence_mean'] <= dt_range[1])
    res = res[idx]
    # print(len(res))
    if len(res) > 0:
        resu = np.copy(res)
        for io, interp in enumerate(names_ref):
            zlims = lim_sn.Interp_griddata(io, res)
            zlims[np.isnan(zlims)] = -1
            print(io, zlims)
            resu = rf.append_fields(resu, 'zlim_'+names_ref[io], zlims)
        if restot is None:
            restot = resu
        else:
            restot = np.concatenate((restot, resu))

        lim_sn.Plot_Hist_zlim(names_ref, restot)
    return restot
