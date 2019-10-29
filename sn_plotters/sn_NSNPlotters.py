import numpy as np
import matplotlib.pyplot as plt

def plot_DDSummary(metricValues,markerdict,colordict):

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


    fig, ax = plt.subplots()
    for cadence in np.unique(summary['cadence']):
        idx = summary['cadence']==cadence
        sela = summary[idx]
        mfc = 'None'
        
        if 'no' not in cadence:
            mfc = 'auto'
            ax.text(0.99*sela['zlim_faint'],0.99*sela['nsn_zfaint'],cadence)
        ax.plot(sela['zlim_faint'],sela['nsn_zfaint'],color='k',marker=markerdict[''.join(cadence.split())],mfc=mfc)
        #horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,fontsize=15)

    al = []
    bl = []
    ac = []
    bc = []
    figb, axb = plt.subplots()
    
    for icad,cadence in enumerate(np.unique(summary_fields['cadence'])):
        idx = summary_fields['cadence']==cadence
        sela = summary_fields[idx]
        mfc = 'auto'
        if 'no' in cadence:
            mfc = 'None'
        
        for ifd, fieldname in enumerate(np.unique(sela['fieldname'])):
            idxb = sela['fieldname']==fieldname
            selb = sela[idxb]
            
            if ifd == 0:
                axb.plot(selb['zlim_faint'],selb['nsn_zfaint'],color=colordict[fieldname],marker=markerdict[''.join(cadence.split())],mfc=mfc)
                
                ab, = axb.plot(5.*selb['zlim_faint'],5.*selb['nsn_zfaint'],color = 'k',marker=markerdict[''.join(cadence.split())],linestyle='None')
                
                ac.append(ab)
                ac.append(cadence)
                
            else:
                axb.plot(selb['zlim_faint'],selb['nsn_zfaint'],color=colordict[fieldname],marker=markerdict[''.join(cadence.split())],mfc=mfc)
                
            if icad == 0:
                ab, = axb.plot(5.*selb['zlim_faint'],5.*selb['nsn_zfaint'],color=colordict[fieldname],marker='o',linestyle='None')
                al.append(ab)
                bl.append(fieldname)

    print(np.min(summary_fields['zlim_faint']),np.max(summary_fields['zlim_faint']))
    zvals = summary_fields['zlim_faint']
    nsn = summary_fields['nsn_zfaint']
    axb.set_xlim(0.99*np.min(zvals),1.01*np.max(zvals))
    axb.set_ylim(0.95*np.min(nsn),1.05*np.max(nsn))
        
    print(al,bl)
    axb.legend(al,bl,loc='upper left')
    #axb.legend(ac,bc,loc='upper right')
    plt.show()

    

