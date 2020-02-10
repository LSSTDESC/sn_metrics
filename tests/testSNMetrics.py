from sn_tools.sn_cadence_tools import ReferenceData
from sn_metrics.sn_obsrate_metric import SNObsRateMetric
from sn_metrics.sn_snr_metric import SNSNRMetric
import sn_plotters.sn_snrPlotters as sn_snr_plot
import sn_plotters.sn_cadencePlotters as sn_cadence_plot
from sn_metrics.sn_cadence_metric import SNCadenceMetric
import matplotlib.pyplot as plt
import yaml
import lsst.utils.tests
import unittest
import numpy as np
from builtins import zip
import matplotlib
matplotlib.use("Agg")
# import lsst.sims.maf.metrics as metrics


m5_ref = dict(
    zip('ugrizy', [23.60, 24.83, 24.38, 23.92, 23.35, 22.44]))


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
                             [24.00371251, 5.0, 5., 1., 0., 0., 0., 245.]))
        for key in refResult.keys():
            assert((np.abs(refResult[key]-result[key]) < 1.e-5))

        res_z = sn_cadence_plot.plotCadence(band, config['Li file'], config['Mag_to_flux file'],
                                            SNR[band],
                                            result,
                                            config['names_ref'],
                                            mag_range=mag_range, dt_range=dt_range,
                                            dbName='Fakes')

        #zlim = 0.3743514031001232
        zlim = 0.3171019707213057
        zres = res_z['zlim_{}'.format(config['names_ref'][0])]
        assert(np.abs(zlim-zres) < 1.e-5)

    def testSNRMetric(self):
        # Test the SN SNR metric

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
        names = ['observationStartMJD', 'fieldRA', 'fieldDec',
                 'fiveSigmaDepth', 'visitExposureTime', 'numExposures', 'visitTime', 'season']
        types = ['f8']*len(names)
        names += ['night']
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
        data['night'] = np.floor(data['observationStartMJD']-day0)
        data['fiveSigmaDepth'] = m5_ref[band]
        data['visitExposureTime'] = 15.
        data['numExposures'] = 2
        data['visitTime'] = 2.*15.
        data['season'] = season
        data['filter'] = band

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

        assert(np.abs(result_metric-result_ref) < 1.e-5)

        # now let us test the plotter
        """
        snr_obs = result['snr_obs']
        snr_fakes = result['snr_fakes']
        detec_frac = result['detec_frac']
        """
        detec_frac = result
        """
        for inum, (Ra, Dec, season) in enumerate(np.unique(snr_obs[['fieldRA', 'fieldDec', 'season']])):
            idx = (snr_obs['fieldRA'] == Ra) & (
                snr_obs['fieldDec'] == Dec) & (snr_obs['season'] == season)
            sel_obs = snr_obs[idx]
            idxb = (np.abs(snr_fakes['fieldRA'] - Ra) < 1.e-5) & (np.abs(
                snr_fakes['fieldDec'] - Dec) < 1.e-5) & (snr_fakes['season'] == season)
            sel_fakes = snr_fakes[idxb]
            sn_snr_plot.SNRPlot(Ra, Dec, season, sel_obs,
                                sel_fakes, config, metric, z)
        """
        sn_snr_plot.detecFracPlot(detec_frac, config['Pixelisation']
                                  ['nside'], config['names_ref'])

        sn_snr_plot.detecFracHist(
            detec_frac, config['names_ref'])

        # plt.show()

    def testObsRateMetric(self):
        """
        Test ObsRate metric

        """


"""


def setup_module(module):
    lsst.utils.tests.init()

"""
if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main(verbosity=5)
