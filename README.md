## Feature Importance Summary
This tool provides an interpretable summary of the effect of
all features and pairs of features on your output variable. Provided
an output variable of interest (e.g. a 0/1 class label, a numerical metric
like latency, etc.), it prints any feature values and
pairs of feature values that are associated with unusual average values of the output.
The summary can be used for feature selection or to construct new features based on
particular ranges or combinations of the original features. The summaries of different batches
of data can also be compared to determine changes in feature importance.

Any feature value that appears in at least `count_thresh` datapoints and
whose datapoints have an average output value `z_thresh` standard deviations
away from the global dataset average will be reported in the
1D stats section of the printed summary. For example, with
`z_thresh=3.0` and `count_thresh=20`, and an output called "Total Sales", we might see the following as part of our summary if we have a feature called "Price":
```
Total Sales global mean: 4.1, global stddev: 2.2

***1D stats***

Price:
  0: 5.2 (z: 4.8, #: 101)
  1: 3.2 (z: -3.6, #: 75)
```
Continuous features and outputs are discretized into buckets, and the average Total Sales
bucket across the entire dataset is 4.1 with standard deviation 2.2. The 101 datapoints with
Price bucket 0 have an average Total Sales of 5.2, which is 4.8 standard deviations
above the mean of 4.1 and therefore significant at `z_thresh=3.0`. Likewise for the 75 datapoints with Price bucket 1.

The 2D stats section shows any pairs of feature values that are interesting. If we had a
second feature called "Department", we might see the following as part of our summary:
```
***2D stats***

Price/Department:
  0/Shoes: 7.0 (z: 3.1, #: 20)
```
This shows that the 20 datapoints with Price bucket 0 and Department=Shoes have an average
output value of 7.0, which is at least 3.1 standard deviations from both the average
for all datapoints with Price bucket 0 and the average for all datapoints with
Department=Shoes. The same `z_thresh` and `count_thresh` from the 1D stats are used for
the 2D stats.

String, int, and double features are supported. String
features with cardinality up to 100 will be considered categorical, and others will be discarded.
Int/double features  with cardinality up to 50 will be considered categorical, while others will
be discretized into 15 buckets of equal size, plus a separate
bucket for nulls. The output variable must be int or double. If it has cardinality
above 50, it will be discretized the same way as the features. If not, it is
expected to have values in the range [0, 256), and will be floored to int type if it
is double.

## Installation
Install pandas and pyarrow (python3). We recommend installing these
packages inside a virtualenv,
which can be created with `python3 -m venv cube_venv` and then entered
with `source cube_venv/bin/activate`. On macOS, install llvm with
`brew install llvm`.

Build the C++ library with
`<FLAG> ./build.sh`, where `<FLAG>` can be `SIM=1` (CPU), `GPU=1` or
`FPGA=1` depending on the desired accelerator. If there are complaints about
missing `Python.h`, you may need to install the package `python3-dev` or
`python3-devel`. The GPU build requires
the `nvcc` compiler, which should be available in any GPU-specific
AMI on EC2, such as the deep learning AMIs. Details on the FPGA build are below.

Modify rookies.py to load your dataset. Pass in the dataset, desired output 
column, `z_thresh`, `count_thresh`, and whether results with a null feature value
should be shown. Run with `python3 rookies.py` to see the printed summary.
rand.py is an example that uses randomly generated data.

### FPGA Setup
Start an Amazon F1 instance with the latest FPGA AMI (tested with
Centos AMI 1.9.1). Clone the F1 SDK from https://github.com/aws/aws-fpga.
Run `source sdk_setup.sh`. Then follow the XDMA driver installation instructions
at `sdk/linux_kernel_drivers/xdma/xdma_install.md`. Set the environment variable
`F1_SDK` to the path to the sdk directory in the .bashrc or equivalent.

Follow the instructions above to install the Python dependencies and
build the C++ library. If you see an error 
about the `static` keyword in the AWS SDK, simply delete the keyword from the
offending location and everything should work.

To run a Python script, instead of directly calling python, use
`./run_script.sh <script name> <script args...>`.

To see the RTL for the FPGA image and to actually build it yourself, check out
https://github.com/jjthomas/DataCubeFPGA.

## Example Data
We include an example dataset Rookies.csv, which includes rookie-year stats for all NBA
players through 2017. We look at which feature values and pairs of values are interesting
with respect to the "IFAS" output, which is a binary variable indicating whether
the player ever became an all-star in their career.
