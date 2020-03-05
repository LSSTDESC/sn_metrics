import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import numpy.lib.recfunctions as rf


def SNRPlot(RA, Dec, season, data, data_fakes, config, metric, z, draw_fakes=True):
    """
    Signal-to-Ratio vs MJD plot for one field and one season
    SNR of  a SN with T0=MJD-10 days

    Parameters
    -------------
    RA : float
      right ascension of the field
    Dec : float
       declination of the field
    data : array
        data (obs) with the fields
        SNR_name_ref : SNR (float)
        season : season  num(float)
        cadence: cadence (float)
        season_length: season length (float)
        MJD_min: min MJD (float)
        DayMax: SN T0 (float)
        MJD: MJD (float) 
        m5_eff: effective 5-sigma depth (float)
        fieldRA: right ascension (float)
        fieldDec: declination (float)
        band: filter (str)
        m5: five-sigma depth (float)
        Nvisits: number of visits (int)
        ExposureTime: exposure time (float)
    data_fakes : array 
         fake data (same fields as data)
    config : dict
         configuration parameters
    metric : 
    z : float
       redshift
    draw_fakes : bool,opt
       if True : fake data are superimposed

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

    figa.suptitle('RA = '+str(np.round(RA, 2))+' Dec = '+str(np.round(Dec, 2)) +
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


def detecFracPlot(data, nside, names_ref):
    """
    Plot Mollweid view of detection rates

    Parameters
    --------------
    data : array with the following fields:
    fieldRA : right ascension (float)
    fieldDec : declination (float)
    season : season num (float)
    band : filter (str)
    frac_obs_name_ref : fraction of detection (detection rate) (float)
    name_ref : list(str)
      name of the simulator used to produce the reference files
    """""
    #data_heal = GetHealpix(data, nside)
    npix = hp.nside2npix(nside)
    
    for band, season in np.unique(data[['band', 'season']]):
        idx = (data['band'] == band) & (data['season'] == season)
        sel = data[idx]
        for sim in names_ref:
            fig, ax = plt.subplots()
            hpxmap = np.zeros(npix, dtype=np.float)
            hpxmap[sel['healpixID'].astype(int)] += sel['frac_obs_'+sim]
            cmap = plt.cm.jet
            # cmap.Normalize(clip=True)
            cmap.set_under('w')
            # remove max=200 and norm='hist' to get the DDFs
            median_value = np.median(sel['frac_obs_'+sim])
            #plt.axes(ax)
            plt.sca(ax)
            hp.mollview(hpxmap, min=0, max=1., cmap=cmap,
                        title='{} - season {} \n median: {}'.format(band, int(season), np.round(median_value, 2)), hold=True)


def detecFracHist(data, names_ref,saveFig=False):
    """
    Plot histogram of detection rates 

    Parameters
    --------------
    data : array with the following fields:
    fieldRA : right ascension (float)
    fieldDec : declination (float)
    season : season num (float)
    band : filter (str)
    frac_obs_name_ref : fraction of detection (detection rate) (float)
    name_ref : list(str)
      name of the simulator used to produce the reference files
    """""

    for band, season in np.unique(data[['band', 'season']]):
        idx = (data['band'] == band) & (np.abs(data['season']-season) < 1.e-5)
        detecFracHist_bandseason(data[idx], band, season, names_ref,saveFig=saveFig)


def detecFracHist_bandseason(data, band, season, names_ref,saveFig=False):
    """
    Plot histogram of detection rates per band and per season

    Parameters
    -------------- 
    data : array with the following fields:
    fieldRA : right ascension (float)
    fieldDec : declination (float)
    season : season num (float)
    band : filter (str)
    frac_obs_name_ref : fraction of detection (detection rate) (float)
    name_ref : list(str)
      name of the simulator used to produce the reference files
    """
    r = []
    fontsize = 15
    colors = dict(zip(range(0, 4), ['r', 'k', 'b', 'g']))

    fig, ax = plt.subplots(figsize=(8, 6))
    title = '{} band - season {}'.format(band, season)
    fig.suptitle(title, fontsize=fontsize)
    label = []
    xminv = []
    xmaxv = []

    for j, name in enumerate(names_ref):
        xminv.append(np.min(data['frac_obs_'+name]))
        xmaxv.append(np.max(data['frac_obs_'+name]))

        xmin = np.min(xminv)
        xmax = np.max(xmaxv)
        xstep = 0.025
        bins = np.arange(xmin, xmax+xstep, xstep)

        for j, name in enumerate(names_ref):
            label.append(
                name + '  Detection rate = ' + str(np.median(np.round(data['frac_obs_'+name], 2))))
            ax.hist(data['frac_obs_'+name], range=[xmin, xmax],
                    bins=bins, histtype='step', color=colors[j], linewidth=2)

        ax.set_xlabel('Detection Rate', fontsize=fontsize)
        ax.set_ylabel(r'Number of Entries', fontsize=fontsize)
        # ax.set_xticks(np.arange(0.5,0.75,0.05))
        ax.tick_params(labelsize=fontsize)
        ax.grid()
        plt.legend(label, fontsize=fontsize-2., loc='upper left')
        plt.grid(1)
        if saveFig:
            plt.savefig('test.png')


def GetHealpix(data, nside):
    """
    Get Healpix map of data

    Parameters
    --------------
    data : array with the following fields
    fieldRA : float
      right ascension
    fieldDec : float
      declination

    Returns
    ---------
    array with the following fields:
      fieldRA: right ascension (float)
      fieldDec : declination (float)
      healpixId : healpix id (int)
    """
    res = data.copy()
    npix = hp.nside2npix(nside)
    print(res[['fieldRA', 'fieldDec']])
    table = hp.ang2vec(res['fieldRA'], res['fieldDec'], lonlat=True)
    healpixs = hp.vec2pix(nside, table[:, 0], table[:, 1], table[:, 2])
    print(healpixs)
    res = rf.append_fields(res, 'healpixID', healpixs)
    return res
