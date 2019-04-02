from builtins import zip
import matplotlib
matplotlib.use("Agg")
import numpy as np
import unittest
#import lsst.sims.maf.metrics as metrics
import lsst.utils.tests
import yaml
import matplotlib.pylab as plt

from sn_metrics.sn_metrics.sn_cadence_metric import SNCadenceMetric
import sn_metrics.sn_plotters.sn_cadencePlotters as sn_plot
from sn_metrics.sn_metrics.sn_snr_metric import SNSNRMetric
from sn_tools.sn_cadence_tools import ReferenceData

m5_ref = dict(
    zip('ugrizy', [23.60, 24.83, 24.38, 23.92, 23.35, 22.44]))


class TestSNmetrics(unittest.TestCase):

    def testCadenceMetric(self):
        """Test the SN cadence metric """

        # Load required SN info to run the metric
        config = yaml.load(open('config/param_cadence_metric.yaml'))
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
                 'fiveSigmaDepth', 'visitExposureTime', 'numExposures', 'visitTime', 'season']
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

        # Run the metric with these fake data
        slicePoint = [0]
        metric = SNCadenceMetric(
            config=config, coadd=config['Observations']['coadd'], names_ref=config['names_ref'])
        result = metric.run(data, slicePoint)

        # And the result should be...
        m5_mean = 24.38
        cadence_mean = 4.97959184
        assert((np.abs(m5_mean-result['m5_mean']) < 1.e-5) &
               (np.abs(cadence_mean-result['cadence_mean']) < 1.e-5))

        res_z = sn_plot.Plot_Cadence(band, config['Li file'], config['Mag_to_flux file'],
                                     SNR[band],
                                     result,
                                     config['names_ref'],
                                     mag_range=mag_range, dt_range=dt_range)

        zlim = 0.3743514031001232
        zres = res_z['zlim_{}'.format(config['names_ref'][0])]
        assert(np.abs(zlim-zres) < 1.e-5)

    def testSNRMetric(self):
        # Test the SN SNR metric

        # Load required SN info to run the metric
        config = yaml.load(open('config/param_snr_metric.yaml'))
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
        metric = SNSNRMetric(config=config, coadd=config['Observations']
                             ['coadd'], lim_sn=lim_sn, names_ref=config['names_ref'], z=z)

        result = metric.run(data, slicePoint)

        # And the result should be...
        result_ref = 0.5234375
        result_metric = result['detec_frac']['frac_obs_{}'.format(
            config['names_ref'][0])]
        assert(np.abs(result_metric-result_ref) < 1.e-5)


"""
def setup_module(module):
    lsst.utils.tests.init()

"""
if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
