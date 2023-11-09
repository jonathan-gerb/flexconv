## Version: 0.1.0

# flexconv

Easy to use Pytorch implementation of FlexConv and CCNN from:

> Modelling long range dependencies in nd: from task-specific to a general purpose cnn.

by Knigge et al. 

link: [https://arxiv.org/pdf/2301.10540.pdf](https://arxiv.org/pdf/2301.10540.pdf)

Adapted fromt the original codebase:
https://github.com/david-knigge/ccnn/tree/main


# Installation
install directly from git:
```
pip install git+https://github.com/jonathan-gerb/flexconv/tree/main
```

# Usage
Import the FlexConv modules from the package, can be used directly in your torch models.
```
from flexconv import FlexConv, SeparableFlexConv
in_channels = 3
out_channels = 32
n_dims = 2 # 1 for sequential data, 2 for image etc.
conv = FlexConv(in_channels, out_channels, n_dims)
sep_conv = SeparableFlexConv(in_channels, out_channels, n_dims)
```
The package also contains The CCNN and CCNNBlock:
```
from flexconv import CCNN, CCNNBlock

in_channels = 3
out_channels = 32
n_dims = 2 # 1 for sequential data, 2 for image etc.

model = CCNN(in_channels, out_channels, data_dim=n_dims, no_hidden=380, no_blocks=6)
ccnn_block = CCNNBlock(in_channels, out_channels, data_dim=n_dims)

```
For additional options, such as kernel network options and masking options etc, see the docstrings. Other modules such as TCNBlock are also available, but have not been tested extensively. 
For some additional initialization examples and a training example, see the `notebooks` folder

### FFT Convolutions
By default FFT convolutions are used for any input size/kernel size bigger than 100 as this is when FFT convolutions are faster than spatial convolutions.
