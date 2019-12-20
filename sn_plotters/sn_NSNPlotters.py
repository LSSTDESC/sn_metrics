import numpy as np
import matplotlib.pyplot as plt

#def plot_DDSummary(metricValues,markerdict,colordict,colors_cad):
def plot_DDSummary(metricValues,forPlot):

    print(metricValues.dtype)

    r = []
    rsum = []
    for cadence in np.unique(metricValues['cadence']):
        idxa = (metricValues['cadence']==cadence)&(metricValues['zlim_faint']>0.)
        sela = metricValues[idxa]
        rsum.append((cadence,
                      np.sum(sela['nsn_zfaint']),
                      np.sum(sela['nsn_zmedium']),
                      np.median(sela['zlim_faint']),
                      np.median(sela['zlim_medium'])))

        for fieldname in np.unique(sela['fieldname']):
            idxb = sela['fieldname']==fieldname
            selb = sela[idxb]
            print(cadence,fieldname,
                      np.sum(selb['nsn_zfaint']),
                      np.sum(selb['nsn_zmedium']),
                      np.median(selb['zlim_faint']),
                      np.median(selb['zlim_medium']))
            r.append((cadence,fieldname,
                      np.sum(selb['nsn_zfaint']),
                      np.sum(selb['nsn_zmedium']),
                      np.median(selb['zlim_faint']),
                      np.median(selb['zlim_medium'])))
            """
            for season in np.unique(selb['season']):
                idxc = selb['season']==season
                selc = selb[idxc]
                
                print(cadence,fieldname,season,
                      np.sum(selc['nsn_zfaint']),
                      np.sum(selc['nsn_zmedium']),
                      np.median(selc['zlim_faint']),
                      np.median(selc['zlim_medium']))
            """          
            
    summary_fields = np.rec.fromrecords(r, names=['cadence','fieldname','nsn_zfaint','nsn_zmedium','zlim_faint','zlim_medium'])

    summary= np.rec.fromrecords(rsum, names=['cadence','nsn_zfaint','nsn_zmedium','zlim_faint','zlim_medium'])

    summary.sort(order='nsn_zfaint')

    print('Summary')
    for val in summary:
        if 'nodither' not in val['cadence']:
            print(val['cadence'],int(val['nsn_zfaint']),int(val['nsn_zmedium']),np.round(val['zlim_faint'],2),np.round(val['zlim_medium'],2))

    Plot_NSNTot(summary,forPlot,sntype='faint')
    #Plot_NSNField(summary_fields,colordict,markerdict,sntype='medium')
    plt.show()

#def Plot_NSNTot(summary,colors_cad,markerdict,sntype='faint'):
def Plot_NSNTot(summary,forPlot,sntype='faint'):
    fontsize = 12
    fig, ax = plt.subplots()
    for cadence in np.unique(summary['cadence']):
        idx = summary['cadence']==cadence
        sela = summary[idx]

        idp = forPlot['dbName']==cadence.strip()
        custplot = forPlot[idp]
        #print('oi',cadence,custplot)
        #print(forPlot['dbName'])
        mfc = 'None'
        cadence_small = '_'.join(cadence.split('_')[:2])
        if 'no' not in cadence:
            mfc = 'auto'
            xshift = 1.0
            yshift = 1.005

            if 'descddf_illum15_' in cadence:
                yshift= 0.99
                xshift = 0.98
                
            if 'descddf_illum4_' in cadence:
                yshift= 0.99
                
            if 'descddf_illum5_' in cadence:
                yshift = 1.002

            #ax.text(xshift*sela['zlim_{}'.format(sntype)],yshift*sela['nsn_z{}'.format(sntype)],cadence_small,color=colors_cad[cadence_small])
            ax.text(xshift*sela['zlim_{}'.format(sntype)],yshift*sela['nsn_z{}'.format(sntype)],cadence_small,color=custplot['color'][0])
        ax.plot(sela['zlim_{}'.format(sntype)],sela['nsn_z{}'.format(sntype)],marker=custplot['marker'][0],mfc=mfc,color=custplot['color'][0])

    ax.grid()
    ax.set_xlabel('$z_{'+sntype+'}$',fontsize=fontsize)
    ax.set_ylabel('$N_{SN} (z<)$',fontsize=fontsize)
        #horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,fontsize=15)
    #ax.legend(loc='upper right')

def Plot_NSNField(summary_fields,colordict,markerdict,sntype='faint'):
    al = []
    bl = []
    ac = []
    bc = []
    fontsize = 12
    figb, axb = plt.subplots()
    fdraw = {}
    for icad,cadence in enumerate(np.unique(summary_fields['cadence'])):
        idx = summary_fields['cadence']==cadence
        sela = summary_fields[idx]
        mfc = 'auto'
        if 'no' in cadence:
            mfc = 'None'
        
        for ifd, fieldname in enumerate(np.unique(sela['fieldname'])):
            idxb = sela['fieldname']==fieldname
            selb = sela[idxb]
            print(ifd,'fieldname',fieldname)
            if ifd == 0:
                axb.plot(selb['zlim_{}'.format(sntype)],selb['nsn_z{}'.format(sntype)],color=colordict[fieldname],marker=markerdict[''.join(cadence.split())],mfc=mfc)
                
                ab, = axb.plot(5.*selb['zlim_{}'.format(sntype)],5.*selb['nsn_z{}'.format(sntype)],color = 'k',marker=markerdict[''.join(cadence.split())],linestyle='None')
                
                ac.append(ab)
                ac.append(cadence)
               
                
            else:
                axb.plot(selb['zlim_{}'.format(sntype)],selb['nsn_z{}'.format(sntype)],color=colordict[fieldname],marker=markerdict[''.join(cadence.split())],mfc=mfc)
                
            if not fieldname in fdraw.keys():
                ab, = axb.plot(5.*selb['zlim_{}'.format(sntype)],5.*selb['nsn_z{}'.format(sntype)],color=colordict[fieldname],marker='o',linestyle='None')
                al.append(ab)
                bl.append(fieldname)
                fdraw[fieldname]=1
                

    print(np.min(summary_fields['zlim_{}'.format(sntype)]),np.max(summary_fields['zlim_{}'.format(sntype)]))
    zvals = summary_fields['zlim_{}'.format(sntype)]
    nsn = summary_fields['nsn_z{}'.format(sntype)]
    axb.set_xlim(0.99*np.min(zvals),1.01*np.max(zvals))
    axb.set_ylim(0.95*np.min(nsn),1.05*np.max(nsn))
        
    print(al,bl)
    axb.legend(al,bl,loc='upper left')
    #axb.legend(ac,bc,loc='upper right')
    axb.set_xlabel('$z_{faint}$',fontsize=fontsize)
    axb.set_ylabel('$N_{SN} (z<)$',fontsize=fontsize)
    axb.grid()
    plt.show()

    

