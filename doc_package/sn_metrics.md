# sn_metrics 

 * \_\_init\_\_.py
 ## sn_cadence_metric.py ##

|name | type | task|
|----|----|----|
|SNCadenceMetric | class |Measuring mean m5, cadence per band and per season for SN depth measurements|

## sn_global_metric.py ##

|name | type | task|
|----|----|----|
|SNGlobalMetric| class|Estimating global properties (per night) of observing strategy|


## sn_nsn_metric.py ##

|name | type | task|
|----|----|----|
|SNNSNMetric | class | Estimating the (nSN, zlim) metric for supernovae|
|time_this|function|decorator for timing purpose|
|verbose_this| function| decorator for printing info purpose|

## sn_obsrate_metric.py ##

|name | type | task|
|----|----|----|
|SNObsRateMetric|class|Measure SN-Signal-to-Noise Ratio as a function of time.|
 | | | Extract observation rate from these measurements|

## sn_sl_metric.py ##

|name | type | task|
|----|----|----|
|SNSLMetric | class | Strongly-lensed SN metric|

## sn_snr_metric.py ##

|name | type | task|
|----|----|----|
|SNSNRMetric | class|Measure SN-Signal-to-Noise Ratio as a function of time|
| | |Extract the detection rate from these measurements|