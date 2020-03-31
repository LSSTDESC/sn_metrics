from sn_tools.sn_cadence_tools import ReferenceData
from sn_metrics.sn_obsrate_metric import SNObsRateMetric
from sn_metrics.sn_snr_metric import SNSNRMetric
import sn_plotters.sn_snrPlotters as sn_snr_plot
import sn_plotters.sn_cadencePlotters as sn_cadence_plot
from sn_metrics.sn_cadence_metric import SNCadenceMetric
from sn_metrics.sn_nsn_metric import SNNSNMetric
from sn_tools.sn_utils import GetReference
import matplotlib.pyplot as plt
import yaml
import lsst.utils.tests
import unittest
import numpy as np
from builtins import zip
import matplotlib
matplotlib.use("Agg")

m5_ref = dict(
    zip('ugrizy', [23.60, 24.83, 24.38, 23.92, 23.35, 22.44]))


def load(fname, gamma_reference, instrument):
    """
     Method to load reference LC files

     Parameters
     --------------
     fname: str
       name of the reference file

     Returns
    ----------


     """
    lc_ref = GetReference(
        fname, gamma_reference, instrument)

    return lc_ref


def fakeData(band, season=1):

    # Define fake data
    names = ['observationStartMJD', 'fieldRA', 'fieldDec',
             'fiveSigmaDepth', 'visitExposureTime',
             'numExposures', 'visitTime', 'season',
             'seeingFwhmEff', 'seeingFwhmGeom',
             'airmass', 'sky', 'moonPhase', 'pixRA', 'pixDec']

    types = ['f8']*len(names)
    names += ['night']
    types += ['i2']
    names += ['healpixID']
    types += ['i2']
    names += ['filter']
    types += ['O']

    dayobs = [59948.31957176, 59959.2821412, 59970.26134259,
              59973.25978009, 59976.26383102, 59988.20670139, 59991.18412037,
              60004.1853588, 60032.08975694, 60045.11981481, 60047.98747685,
              60060.02083333, 60071.986875, 60075.96452546]
    day0 = np.min(dayobs)
    npts = len(dayobs)
    data = np.zeros(npts, dtype=list(zip(names, types)))
    data['observationStartMJD'] = dayobs
    data['night'] = np.floor(data['observationStartMJD']-day0+1)
    data['fiveSigmaDepth'] = m5_ref[band]
    data['visitExposureTime'] = 15.
    data['numExposures'] = 2
    data['visitTime'] = 2.*15.
    data['season'] = season
    data['filter'] = band
    data['seeingFwhmEff'] = 0.
    data['seeingFwhmGeom'] = 0.
    data['airmass'] = 1.2
    data['sky'] = 20.0
    data['moonPhase'] = 0.5
    data['pixRA'] = 0.0
    data['pixDec'] = 0.0
    data['healpixID'] = 1

    return data


class TestSNmetrics(unittest.TestCase):

    def testCadenceMetric(self):
        """Test the SN cadence metric """

        # Load required SN info to run the metric
        f = open('config/param_cadence_metric.yaml', 'r')
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

        SNR = dict(zip(config['Observations']['bands'],
                       config['Observations']['SNR']))
        band = 'r'
        SNR = dict(zip('griz', [30., 40., 30., 20.]))  # SNR for WFD
        mag_range = [21., 25.5]  # WFD mag range
        dt_range = [0.5, 30.]  # WFD dt range
        Li_files = ['Li_SNCosmo_-2.0_0.2.npy']
        mag_to_flux_files = ['Mag_to_Flux_SNCosmo.npy']
        # lim_sn = Lims(Li_files, mag_to_flux_files, band, SNR[band],
        #             mag_range=mag_range, dt_range=dt_range)

        # Define fake data
        names = ['observationStartMJD', 'fieldRA', 'fieldDec',
                 'fiveSigmaDepth', 'visitExposureTime',
                 'numExposures', 'visitTime', 'season',
                 'seeingFwhmEff', 'seeingFwhmGeom',
                 'airmass', 'sky', 'moonPhase']

        types = ['f8']*len(names)
        names += ['night']
        types += ['i2']
        names += ['filter']
        types += ['O']

        day0 = 59000
        daylast = day0+250
        cadence = 5
        dayobs = np.arange(day0, daylast, cadence)
        npts = len(dayobs)
        data = np.zeros(npts, dtype=list(zip(names, types)))
        data['observationStartMJD'] = dayobs
        data['night'] = np.floor(data['observationStartMJD']-day0)
        data['fiveSigmaDepth'] = m5_ref[band]
        data['visitExposureTime'] = 15.
        data['numExposures'] = 2
        data['visitTime'] = 2.*15.
        data['filter'] = band
        data['season'] = 1.
        data['seeingFwhmEff'] = 0.
        data['seeingFwhmGeom'] = 0.
        data['airmass'] = 1.2
        data['sky'] = 20.0
        data['moonPhase'] = 0.5

        # Run the metric with these fake data
        slicePoint = [0]
        metric = SNCadenceMetric(coadd=config['Observations']['coadd'])
        result = metric.run(data, slicePoint)
        idx = result['filter'] == band
        result = result[idx]

        # And the result should be...
        refResult = dict(zip(['m5_mean', 'cadence_mean', 'gap_max', 'gap_5',
                              'gap_10', 'gap_15', 'gap_20', 'season_length'],
                             [24.38, 5.0, 5., 1., 0., 0., 0., 245.]))
        for key in refResult.keys():
            assert((np.abs(refResult[key]-result[key].values) < 1.e-5))

        res_z = sn_cadence_plot.plotCadence(band, config['Li file'], config['Mag_to_flux file'],
                                            SNR[band],
                                            result.to_records(),
                                            config['names_ref'],
                                            mag_range=mag_range, dt_range=dt_range,
                                            dbName='Fakes')

        #zlim = 0.3743514031001232
        zlim = 0.3740264590350457
        zres = res_z['zlim_{}'.format(config['names_ref'][0])]
        assert(np.abs(zlim-zres) < 1.e-5)

    def testSNRMetric(self):
        """Test the SNR metric """

        # Load required SN info to run the metric
        f = open('config/param_snr_metric.yaml', 'r')
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

        band = 'r'
        z = 0.3
        season = 1.

        config['Observations']['season'] = [season]

        Li_files = config['Li file']
        mag_to_flux_files = config['Mag_to_flux file']
        names_ref = config['names_ref']
        coadd = False

        lim_sn = ReferenceData(Li_files, mag_to_flux_files, band, z)

        # Define fake data
        data = fakeData(band)

        # Run the metric with these fake data
        slicePoint = [0]
        metric = SNSNRMetric(lim_sn, config['names_ref'], fake_file=config['Fake_file'], coadd=config['Observations']
                             ['coadd'],  z=z)

        result = metric.run(data, slicePoint)

        # And the result should be...
        result_ref = 0.5234375
        # result_metric = result['detec_frac']['frac_obs_{}'.format(
        #    config['names_ref'][0])]
        result_metric = result['frac_obs_{}'.format(config['names_ref'][0])]

        assert(np.abs(result_metric.values-result_ref) < 1.e-5)

        # now let us test the plotter
        """
        snr_obs = result['snr_obs']
        snr_fakes = result['snr_fakes']
        detec_frac = result['detec_frac']
        """
        detec_frac = result.to_records()
        """
        for inum, (RA, Dec, season) in enumerate(np.unique(snr_obs[['fieldRA', 'fieldDec', 'season']])):
            idx = (snr_obs['fieldRA'] == RA) & (
                snr_obs['fieldDec'] == Dec) & (snr_obs['season'] == season)
            sel_obs = snr_obs[idx]
            idxb = (np.abs(snr_fakes['fieldRA'] - RA) < 1.e-5) & (np.abs(
                snr_fakes['fieldDec'] - Dec) < 1.e-5) & (snr_fakes['season'] == season)
            sel_fakes = snr_fakes[idxb]
            sn_snr_plot.SNRPlot(RA, Dec, season, sel_obs,
                                sel_fakes, config, metric, z)
        """
        sn_snr_plot.detecFracPlot(detec_frac, config['Pixelisation']
                                  ['nside'], config['names_ref'])

        sn_snr_plot.detecFracHist(
            detec_frac, config['names_ref'])

        # plt.show()

    def testObsRateMetric(self):
        """Test the  ObsRate metric """

        # Load required SN info to run the metric
        f = open('config/param_obsrate_metric.yaml', 'r')
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

        bands = config['Observations']['bands']
        snr_ref = config['Observations']['SNR']
        z = config['Observations']['z']
        coadd = config['Observations']['coadd']
        season = config['Observations']['season']
        lim_sn = {}
        z = 0.6
        for band in bands:
            lim_sn[band] = ReferenceData(
                config['Li file'], config['Mag_to_flux file'], band, z)

        # metric instance
        coadd = True

        metric = SNObsRateMetric(lim_sn=lim_sn,
                                 names_ref=config['names_ref'],
                                 season=season, coadd=coadd,
                                 z=z, bands=bands,
                                 snr_ref=dict(zip(bands, snr_ref)))

        # make fake data

        data = None
        cadence = dict(zip('grizy', [10, 20, 20, 26, 20]))
        #cadence = dict(zip('grizy', [1, 1, 1, 1, 1]))
        for band in bands:
            for i in range(cadence[band]):
                fakes = fakeData(band)
                if data is None:
                    data = fakes
                else:
                    data = np.concatenate((data, fakes))

        res = metric.run(data)
        result_metric = res['frac_obs_{}'.format(config['names_ref'][0])]
        result_ref = 0.125
        assert(np.abs(result_metric.values-result_ref) < 1.e-5)

    def testNSNMetric(self):
        """Test the NSN metric """

        # reference dir
        dir_ref = '../../reference_files/'
        # template dir
        dir_template = '../../../Templates'
        #dir_template = '../../../..'
        # input parameters for this metric
        name = 'NSN'
        season = 1
        coadd = True,
        fieldType = 'DD',
        nside = 64
        ramin = 0.
        ramax = 360.
        decmin = -1.0,
        decmax = -1.0
        metadata = {}
        outDir = 'MetricOutput'
        proxy_level = 1

        # An instrument is needed
        Instrument = {}
        Instrument['name'] = 'LSST'  # name of the telescope (internal)
        # dir of throughput
        Instrument['throughput_dir'] = 'LSST_THROUGHPUTS_BASELINE'
        Instrument['atmos_dir'] = 'THROUGHPUTS_DIR'  # dir of atmos
        Instrument['airmass'] = 1.2  # airmass value
        Instrument['atmos'] = True  # atmos
        Instrument['aerosol'] = False  # aerosol

        lc_reference = {}
        gamma_reference = '{}/gamma.hdf5'.format(dir_ref)

        x1_colors = [(-2.0, 0.2), (0.0, 0.0)]

        # load ref files here
        for j in range(len(x1_colors)):
            x1 = x1_colors[j][0]
            color = x1_colors[j][1]
            fname = '{}/LC_{}_{}_vstack.hdf5'.format(
                dir_template, x1, color)

            lc_reference[x1_colors[j]] = load(
                fname, gamma_reference, Instrument)

        # LC selection criteria

        N_bef = 2
        N_aft = 5
        snr_min = 5.
        N_phase_min = 1
        N_phase_max = 1

        # additional parameters requested to run the metric
        zmax = 1.0
        season = [1]
        pixArea = 9.6

        # load x1_color_dist

        x1_color_dist = np.genfromtxt('{}/Dist_X1_Color_JLA_high_z.txt'.format(dir_ref), dtype=None,
                                      names=('x1', 'color', 'weight_x1', 'weight_x1', 'weight_tot'))

        # metric instance
        metric = SNNSNMetric(
            lc_reference, season=season, zmax=zmax, pixArea=pixArea,
            verbose=False, timer=False,
            ploteffi=False,
            N_bef=N_bef, N_aft=N_aft,
            snr_min=snr_min,
            N_phase_min=N_phase_min,
            N_phase_max=N_phase_max,
            outputType='zlims',
            proxy_level=proxy_level,
            x1_color_dist=x1_color_dist,
            coadd=coadd, lightOutput=False, T0s='all')

        # get some data to run the metric
        bands = 'grizy'
        cadence = dict(zip(bands, [10, 20, 20, 26, 20]))
        #cadence = dict(zip('grizy', [1, 1, 1, 1, 1]))
        data = None
        for band in bands:
            for i in range(cadence[band]):
                fakes = fakeData(band)
                if data is None:
                    data = fakes
                else:
                    data = np.concatenate((data, fakes))

        # now run the metric
        res = metric.run(data)

        # compare the results to reference: this is the unit test
        zlim_ref = np.asarray([0.599917, 0.763300])
        #print(res['zlim'], zlim_ref, np.isclose(res['zlim'], zlim_ref))
        assert(np.isclose(res['zlim'], zlim_ref).all())


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main(verbosity=5)
