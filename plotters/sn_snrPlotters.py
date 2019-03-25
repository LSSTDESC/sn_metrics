import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import numpy.lib.recfunctions as rf


def SNRPlot(Ra, Dec, season, data, data_fakes, config, metric, z, draw_fakes=True):
    """
    Signal-to-Ratio vs MJD plot
    SNR of  a SN with T0=MJD-10 days
    (x1,color) chosen in the input yaml file
    Fake observations can be superimposed
    One plot per field, per season.
    """

    colors = ['b', 'r']
    fontsize = 15
    bands_ref = 'ugrizy'
    id_band = [0, 1, 2, 3, 4, 5]
    bands_id = dict(zip(bands_ref, id_band))
    id_bands = dict(zip(id_band, bands_ref))
    bands = np.unique(data['band'])
    lista = sorted([bands_id[b] for b in bands])
    bands = [id_bands[jo] for jo in lista]
    n_bands = len(bands)
    # estimate the number of rows and columns depending on the number of bands
    ncols = 1
    nrows = 1
    if n_bands >= 2:
        ncols = 2
        nrows = int(n_bands/2+(n_bands % 2))

    figa, axa = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15, 10))

    figa.suptitle('Ra = '+str(np.round(Ra, 2))+' Dec = '+str(np.round(Dec, 2)) +
                  ' \n '+' Season '+str(int(season))+' - z = '+str(z), fontsize=fontsize)
    for ib, band in enumerate(bands):
        tot_label = []
        idb = data['band'] == band
        sel = data[idb]
        idb = data_fakes['band'] == band
        sel_fakes = data_fakes[idb]
        sel.sort(order='MJD')
        sel_fakes.sort(order='MJD')
        ifig = int(ib/2)
        jfig = int(ib % 2)

        if nrows > 1:
            ax = axa[ifig][jfig]
        else:
            if ncols > 1:
                ax = axa[jfig]
            else:
                ax = axa

        # Draw results
        for io, sim in enumerate(config['names_ref']):
            tot_label.append(ax.errorbar(
                sel['MJD'], sel['SNR_'+sim], ls='-', color=colors[io], label=sim))
            if draw_fakes:
                tot_label.append(ax.errorbar(
                    sel_fakes['MJD'], sel_fakes['SNR_'+sim], ls='--', color=colors[io], label=sim+'_fake'))

        if ifig == nrows-1:
            ax.set_xlabel('MJD [day]', fontsize=fontsize)
        if jfig == 0:
            ax.set_ylabel('Signal-To-Noise ratio', fontsize=fontsize)

        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

        if ifig == 0 and jfig == 0:
            labs = [l.get_label() for l in tot_label]
            ax.legend(tot_label, labs, ncol=1, loc='best',
                      prop={'size': fontsize}, frameon=False)

        ax.text(0.9, 0.9, band, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes,
                fontsize=fontsize)


def DetecFracPlot(data, nside, names_ref):

    data_heal = GetHealpix(data, nside)
    npix = hp.nside2npix(nside)
    # print(data_heal)
    for band, season in np.unique(data_heal[['band', 'season']]):
        idx = (data_heal['band'] == band) & (data_heal['season'] == season)
        sel = data_heal[idx]
        for sim in names_ref:
            fig, ax = plt.subplots()
            hpxmap = np.zeros(npix, dtype=np.float)
            hpxmap[sel['healpixID']] += sel['frac_obs_'+sim]
            cmap = plt.cm.jet
            # cmap.Normalize(clip=True)
            cmap.set_under('w')
            # remove max=200 and norm='hist' to get the DDFs
            median_value = np.median(sel['frac_obs_'+sim])
            plt.axes(ax)
            hp.mollview(hpxmap, min=0, max=1., cmap=cmap,
                        title='{} - season {} \n median: {}'.format(band, int(season), np.round(median_value, 2)), hold=True)

    plt.show()


def GetHealpix(data, nside):

    res = data.copy()
    npix = hp.nside2npix(nside)
    table = hp.ang2vec(data['fieldRA'], data['fieldDec'], lonlat=True)
    healpixs = hp.vec2pix(nside, table[:, 0], table[:, 1], table[:, 2])
    res = rf.append_fields(res, 'healpixID', healpixs)
    return res
