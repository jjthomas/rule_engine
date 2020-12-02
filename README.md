## Pandas Summary Stats
This library prints a summary of your tabular Pandas data. Provided
a metric column of interest (e.g. a 0/1 class label, a numerical metric
like latency, etc.), it prints any values of other columns and
pairs of these values that are unusually correlated with that metric.

String, int, and double columns are supported. String
columns with cardinality up to 100 will be considered. Int/double columns
with cardinality up to 50 will be left as is, while such columns with greater
cardinality will be discretized into 15 buckets of equal size, plus a separate
bucket for nulls. The metric column must be int or double. If it has cardinality
above 50, it will be discretized the same way as other columns. If not, it is
expected to have values in the range [0, 256), and will be floored to int type if it
is double.

Any column values whose average metric values are are `z_thresh` standard deviations
away from the global dataset metric average and appear in at least `count_thresh`
rows will be reported in the 1D stats section of the output. For example, for a
continuous column called "Price" and a discretized metric column called "Total Sales",
we might see the following output:
```
Price (5.0-100.0):
  0: 7.0 (z: 4.8, #: 101)
  1: 5.5 (z: 3.6, #: 75)
```
This indicates that the Price column had a range of [5.0, 100.0] that was used
to discretize it into 15 equally sized buckets. Rows with an Price value in the
first bucket had an average metric value of 7.0, which was 4.8 standard deviations
above the mean. There were 101 such rows. There is a similar story for the second bucket.

Instead of having a range, categorical variables will simply have "cat"
next to them in the 1D stats section. Their actual values will be used instead of bucket
indices.

The 2D stats section shows any pairs of column values that are interesting. If we had a
second categorical column called "Department", we might see the following output:
```
Price/Department:
  0/Shoes: 10.2 (z: 3.1, #: 20)
```
This shows that the 20 rows with Price in the first bucket and Department=Shoes have an average
metric value of 10.2, which is at least 3.1 standard deviations from both the average
value for all rows with Price in first bucket and the average value for all rows with
Department=Shoes. The same `z_thresh` and `count_thresh` from the 1D stats are used for
the 2D stats.

## Installation
Install pandas and pyarrow (python3). Build the C++ library with `./build.sh`.
Modify cube.py to load your dataset. Pass in the dataset, desired metric
column, `z_thresh`, `count_thresh`, and whether results with a null column value
should be shown. Run with `python3 cube.py` to see printed results.

## Example Data
We include an example dataset Rookies.csv, which includes rookie-year stats for all NBA
players through 2017. We look at which column values and pairs of values are interesting
with respect to the "IFAS" metric column, which is a binary variable indicating whether
the player ever became an all-star in their career.
