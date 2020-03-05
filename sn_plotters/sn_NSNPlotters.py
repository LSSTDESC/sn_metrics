import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_DDSummary(metricValues, forPlot, sntype='faint'):
    """
    Plot to display NSN results for DD fields

    Parameters
    ----------------
    metricValues: numpy array
     array of data to display:
      healpixID: healpixID of the pixel SN
      season: season to plot
      pixRA: RA of the pixel SN
      pixDec: Dec of the pixel SN
      zlim_faint: redshift corresponding to faintest SN (x1=-2.0,color=0.2)
      zlim_medium: redshift corresponding to medium SN (x1=0.0,color=0.0)
      nsn_med_zfaint: number of medium SN with z<zlim_faint
      nsn_med_zmedium: number of medium SN with z<zlim_medium
      nsn_zfaint: number of SN with z<zlim_faint
      nsn_zmedium: number of medium SN with z<zlim_medium
      fieldname: name of the field
      fieldnum: number of the field
      cadence: cadence name
      nside: nside value for healpix tessallation
      pixArea: pixel area
    forPlot: numpy array
     array with a set of plotting information (marker, color) for each cadence:
     dbName: cadence name
     newName: new cadence name
     group: name of the group the cadence is bzelonging to
     Namepl: name of the cadence for plotting
     color: marker color
     marker: marker type
    sntype: str
      type of the supernova (faint or medium) (default: faint) for display


    Returns
    -----------
    Plot (NSN, zlim)


    """

    # select data with zlim_faint>0. and NSN > 10.

    idx = metricValues['zlim_faint'] > 0.
    #idx &= metricValues['nsn_zfaint'] > 10.
    sel = metricValues[idx]

    # estimate some stats to display

    data = pd.DataFrame(np.copy(sel))

    summary = data.groupby(['cadence']).agg({'nsn_zfaint': 'sum',
                                             'nsn_zmedium': 'sum',
                                             'zlim_faint': 'median',
                                             'zlim_medium': 'median', }).reset_index()

    summary_fields = data.groupby(['cadence', 'fieldname']).agg({'nsn_zfaint': 'sum',
                                                                 'nsn_zmedium': 'sum',
                                                                 'zlim_faint': 'median',
                                                                 'zlim_medium': 'median', }).reset_index()

    summary_fields_seasons = data.groupby(['cadence', 'fieldname', 'season']).agg({'nsn_zfaint': 'sum',
                                                                                   'nsn_zmedium': 'sum',
                                                                                   'zlim_faint': 'median',
                                                                                   'zlim_medium': 'median', }).reset_index()

    # change some of the type for printing
    summary.round({'zlim_faint': 2, 'zlim_medium': 2})
    summary['nsn_zfaint'] = summary['nsn_zfaint'].astype(int)
    summary['nsn_zmedium'] = summary['nsn_zmedium'].astype(int)

    # plot the results

    # per field and per season
    Plot_NSN(summary_fields_seasons, forPlot, sntype=sntype)
    # per field, for all seasons
    Plot_NSN(summary_fields, forPlot, sntype=sntype)
    # Summary plot: one (NSN,zlim) per cadence (sum for NSN, median zlim over the fields/seasons)
    Plot_NSN(summary, forPlot, sntype=sntype)


def Plot_NSN(summary, forPlot, sntype='faint'):
    """
    Plot NSN vs redshift limit

    Parameters
    ----------------
    summary: pandas Dataframe
     data to display:
      cadence: name of the cadence
      zlim_faint: redshift corresponding to faintest SN (x1=-2.0,color=0.2)
      zlim_medium: redshift corresponding to medium SN (x1=0.0,color=0.0)
      nsn_zfaint: number of SN with z<zlim_faint
      nsn_zmedium: number of medium SN with z<zlim_medium
    forPlot: numpy array
      array with a set of plotting information (marker, color) for each cadence:
      dbName: cadence name
      newName: new cadence name
      group: name of the group the cadence is bzelonging to
      Namepl: name of the cadence for plotting
      color: marker color
      marker: marker type
    sntype: str,opt
      type of the supernova (faint or medium) for the display (default: faint)

    Returns
    -----------
    Plot (NSN, zlim)


    """

    fontsize = 12
    fig, ax = plt.subplots()
    varx = 'zlim_{}'.format(sntype)
    vary = 'nsn_z{}'.format(sntype)
    xshift = 1.0
    yshift = 1.01

    for group in np.unique(forPlot['group']):
        idx = forPlot['group'] == group
        sel = forPlot[idx]
        #print(group, sel['dbName'])
        marker = sel['marker'].unique()[0]
        color = sel['color'].unique()[0]

        print('ici', sel['dbName'].str.strip(), summary['cadence'])
        selcad = summary[summary['cadence'].str.strip().isin(
            sel['dbName'].str.strip())]

        # plot
        ax.plot(selcad[varx], selcad[vary], color=color,
                marker=marker, lineStyle='None')

        # get the centroid of the data and write it
        centroid_x = selcad[varx].mean()
        centroid_y = selcad[vary].mean()
        ax.text(xshift*centroid_x, yshift*centroid_y, group, color=color)

    ax.grid()
    ax.set_xlabel('$z_{'+sntype+'}$', fontsize=fontsize)
    ax.set_ylabel('$N_{SN} (z<)$', fontsize=fontsize)
