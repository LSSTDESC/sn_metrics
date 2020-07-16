# sn_metrics

sn_metrics is a package to estimate and plot supernovae metrics.


```
This software was developed within the LSST DESC using LSST DESC resources, and so meets the criteria given in, and is bound by, the LSST DESC Publication Policy for being a "DESC product". 
We welcome requests to access code for non-DESC use; if you wish to use the code outside DESC please contact the developers.

```
## Release Status

|Release|Date|packages|
|---|---|---|
|v1.0.0|2020/07/15|sn_metrics_v1.0.0|

## Feedback, License etc

If you have comments, suggestions or questions, please [write us an issue](https://github.com/LSSTDESC/sn_metrics/issues).

This is open source software, available for re-use under the modified BSD license.

```
Copyright (c) 2020, the sn_metrics contributors on GitHub, https://github.com/LSSTDESC/sn_metrics/graphs/contributors.
All rights reserved.
```

## sn_metrics content ##

* **config**: set of config files
 * **docs**: python documentation (sphinx-apidoc)
 * **__init__.py**
 * **input**: input files
 * **LICENCE**: licence file
 * **setup.py**: setup file for pip installation
 * [**sn_metrics**](doc_package/sn_metrics.md): set of python classes to process metrics
 * **sn_plotters**: python classes to plot metric results
 * **tests**: unit tests
 * **README.md**: README file
 * **version.py**: package version

## Complete tree ##

```bash
|-- LICENCE
|-- README.md
|-- __init__.py
|-- config
|   |-- Fake_cadence.yaml
|   |-- Li_SNCosmo_-2.0_0.2.npy
|   |-- Mag_to_Flux_SNCosmo.npy
|   |-- param_cadence_metric.yaml
|   |-- param_snr_metric.yaml
|-- doc_package
|   |-- sn_metrics.md
|-- docs
|   |-- Makefile
|   |-- api
|   |   |-- sn_metrics.rst
|   |   |-- sn_metrics.sn_cadence_metric.rst
|   |   |-- sn_metrics.sn_global_metric.rst
|   |   |-- sn_metrics.sn_nsn_metric.rst
|   |   |-- sn_metrics.sn_obsrate_metric.rst
|   |   |-- sn_metrics.sn_sl_metric.rst
|   |   |-- sn_metrics.sn_snr_metric.rst
|   |   |-- sn_plotters.rst
|   |   |-- sn_plotters.sn_NSNPlotters.rst
|   |   |-- sn_plotters.sn_cadencePlotters.rst
|   |   |-- sn_plotters.sn_snrPlotters.rst
|   |-- conf.py
|   |-- index.rst
|   |-- make.bat
|-- input
|   |-- Fake_cadence.yaml
|-- setup.py
|-- sn_metrics
|   |-- __init__.py
|   |-- __pycache__
|   |   |-- __init__.cpython-37.pyc
|   |   `-- version.cpython-37.pyc
|   |-- sn_cadence_metric.py
|   |-- sn_global_metric.py
|   |-- sn_nsn_metric.py
|   |-- sn_obsrate_metric.py
|   |-- sn_sl_metric.py
|   |-- sn_snr_metric.py
|   |-- version.py
|-- tests
|   |-- config
|   |   |-- param_cadence_metric.yaml
|   |   |-- param_obsrate_metric.yaml
|   |   |-- param_snr_metric.yaml
|   |   |-- param_snr_metric.yaml~
|   |-- input
|   |   |-- Fake_cadence.yaml
|   |-- testSNMetrics.py
`-- version.py
```