## Overview
This library prints a summary of your tabular Pandas data. Provided
a metric column of interest (e.g. a 0/1 class label, a numerical metric
like latency, etc.), it prints any values of other columns and
pairs of these values that are unusually correlated with that metric.

Metric columns must be int or double have values in the range [0, 255].
For other columns, string, int, and double types are supported. String
columns with cardinality up to 100 will be considered. Int/double columns
with cardinality up to 50 will be left as is, while such columns with greater
cardinality will be discretized into 15 buckets of equal size.

Any column values whose average metric values are are `z_thresh` standard deviations
away from the global dataset metric average and appear in at least `count_thresh`
rows will be reported in the 1D stats section of the output. For example, for a
continuous column called "AS", we might see the following output:
```
AS (5.0-100.0):
  0: 2.0 (z: 5.5, #: 101)
  1: 1.5 (z: 3.6, #: 75)
```
This indicates that the AS column had a range of [5.0, 100.0] that was used
to discretize it into 15 equally sized buckets. Rows with an AS value in the
first bucket had an average metric value of 2.0, which was 5.5 standard deviations
above the mean. There were 101 such rows. There is a similar story for the second bucket.

Instead of having a range, categorical variables will simply have the indicator "cat"
next to them in the 1D stats section. Their actual values will be used instead of bucket
indices.

The 2D stats section shows any pairs of column values that are interesting. If we had a
second continuous column called "TO", we might see the following output:
```
AS/TO:
  0/12: 4.0 (z: 3.1, #: 20)
```
This shows that the 20 rows with AS in the first bucket and TO in the 13th bucket have an average
metric value of 4.0, which is at least 3.1 standard deviations from both the average
value for all rows with AS in first bucket and the average value for all rows with TO in the
thirteenth bucket. The same `z_thresh` and `count_thresh` from the 1D stats is used for
the 2D stats.

## Installation
Install pandas and pyarrow (python3). Build the C++ library with `./build.sh`.
Modify cube.py to load your dataset. Pass in the dataset, desired metric
column, `z_thresh`, and `count_thresh`. Run with `python3 cube.py` to see printed
results.

## Example Data
We include an example dataset Rookies.csv, which includes rookie-year stats for all NBA
players through 2017. We look at which column values and pairs of values are interesting
with respect to the "IFAS" metric column, which is a binary variable indicating whether
the player ever became an all-star in their career.
