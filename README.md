## Rule Engine
Rule Engine (RE) creates an interpretable anomaly classifier from many one-feature and
two-feature decision rules. It works natively on labeled, categorical training data,
and automatically discretizes continuous features. It searches over
all rules of the forms `if (F1 = X) => anomaly` and `if (F1 = X && F2 = Y) => anomaly`, where `F1` and
`F2` are features and `X` and `Y` are possible values for those features.
RE creates a classifier out of all rules classifying at least `c` 
training examples with at least `p` precision, where `c` and
`p` are provided by the user. The classifier classifies a test example as an
anomaly if any of its rules fire. RE can prune overlapping rules
to improve overall precision and display rules in an easily understandable format.

See `examples/rookies.py` for the key APIs and example usage.

A presentation on this work is available here: https://cs.stanford.edu/~jjthomas/dataai-pres.pdf.

## Installation
On macOS, install llvm with `brew install llvm`.

Install `rule_engine` with
`<FLAG> pip install .` from the top of this repo, where `<FLAG>` can be empty (CPU), `GPU=1` or
`FPGA=1` depending on the desired accelerator. If there are complaints about
missing `Python.h`, you may need to install the package `python3-dev` or
`python3-devel`. The GPU build requires
the `nvcc` compiler. Details on the FPGA build are below.

Run our example from the `examples` directory with `python3 rookies.py`.

### FPGA Setup
FPGA-accelerated installation must be performed on an Amazon F1 instance with a recent FPGA AMI (tested with
Centos AMI 1.9.1). Before running `pip install`, clone the F1 SDK from https://github.com/aws/aws-fpga.
Run `source sdk_setup.sh`. Then follow the XDMA driver installation instructions
at `sdk/linux_kernel_drivers/xdma/xdma_install.md`. Set the environment variable
`F1_SDK` to the path to the sdk directory in the .bashrc or equivalent.

If you see an error during `pip install`
about the `static` keyword in the AWS SDK, simply delete the keyword from the
offending location and everything should work.

To run a Python script using `rule_engine` with FPGA acceleration, instead of directly calling python, use
`./fpga_run.sh <script name> <script args...>`.

To see the RTL for the FPGA image and to actually build it yourself, check out
https://github.com/jjthomas/DataCubeFPGA.

## Example Data
We include an example dataset `examples/Rookies.csv`, which includes rookie-year stats for all NBA
players through 2017. The class variable "IFAS" indicates whether
the player ever became an all-star in their career.

Other anomaly detection datasets are available here: https://github.com/GuansongPang/anomaly-detection-datasets.
