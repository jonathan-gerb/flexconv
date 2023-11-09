## Version: 0.1.0

# flexconv

Easy to use Pytorch implementation of FlexConv and CCNN from:

> Modelling Long Range Dependencies in N-D: From Task-Specific to a General Purpose CNN](https://arxiv.org/abs/2301.10540)

by Knigge et al.

**Abstract**

Performant Convolutional Neural Network (CNN) architectures must be tailored to specific tasks in order to consider the length, resolution, and dimensionality of the input data. In this work, we tackle the need for problem-specific CNN architectures.\break We present the \textit{Continuous Convolutional Neural Network} (CCNN): a single CNN able to process data of arbitrary resolution, dimensionality and length without any structural changes.  Its key component are its \textit{continuous convolutional kernels} which model long-range dependencies at every layer, and thus remove the need of current CNN architectures for task-dependent downsampling and depths. We showcase the generality of our method by using the \emph{same architecture} for tasks on sequential (1D), visual (2D) and point-cloud (3D) data. Our CCNN matches and often outperforms the current state-of-the-art across all tasks considered.
### Installation

Adapted from the original codebase with a few minor changes:
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

### Cite
If you found this work useful in your research, please consider citing:

```
@article{knigge2023modelling,
  title={Modelling Long Range Dependencies in N-D: From Task-Specific to a General Purpose CNN},
  author={Knigge, David M and Romero, David W and Gu, Albert and Bekkers, Erik J and Gavves, Efstratios and Tomczak, Jakub M and Hoogendoorn, Mark and Sonke, Jan-Jakob},
  journal={International Conference on Learning Representations},
  year={2023}
}
