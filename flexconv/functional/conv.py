import torch
import torch.nn.functional as F
import torch.fft
from fft_conv_pytorch import fft_conv

from typing import Optional

def causal_padding(
    x: torch.Tensor,
    kernel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. Pad the input signal & kernel tensors.
    # Check if the size of the kernel is odd. If not, add a pad of zero to make them odd.
    if kernel.shape[-1] % 2 == 0:
        kernel = F.pad(kernel, [1, 0], value=0.0)
        # x = torch.nn.functional.pad(x, [1, 0], value=0.0)

    # 2. Perform padding on the input so that output equals input in length
    x = F.pad(x, [kernel.shape[-1] - 1, 0], value=0.0)
    return x, kernel


# due to the way the rest of the code was written, 
# it was easier to create these methods instead of adding an argument.
def fftconv(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor],
    separable: bool = False,
    causal: bool = False,
):
    return conv_base(
        x=x,
        kernel=kernel,
        bias=bias,
        separable=separable,
        causal=causal,
        fft=True
    )


def conv(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor],
    separable: bool = False,
    causal: bool = False,
):
    return conv_base(
        x=x,
        kernel=kernel,
        bias=bias,
        separable=separable,
        causal=causal,
        fft=False
    )


def conv_base(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor],
    separable: bool = False,
    fft: bool = True,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Args:
        x: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (Optional, Tensor) Number of zero samples to pad the input on the last dimension.
    Returns:
        (Tensor) Convolved tensor
    """
    # Input tensors are assumed to have dimensionality [batch_size, no_channels, spatial_dim1, .., spatial_dimN].

    data_dim = len(x.shape) - 2
    assert data_dim in [
        1,
        2,
        3,
    ], f"Convolution is not implemented for inputs of spatial dimension {data_dim}"
    
    kernel_size = torch.tensor(kernel.shape[-data_dim:])
    
    if data_dim == 1:
        assert causal, f"Causal is set to False even though you are using a 1d convolution, you probably want to set causal to true."

    if causal:
        assert data_dim == 1, f"Causal only available for data of dimension 1, not {data_dim}"
        x, kernel = causal_padding(x, kernel)
    else:
        padding = (kernel_size // 2).tolist()
        assert torch.all(
            kernel_size % 2 != 0
        ), f"Convolutional kernels must have odd dimensionality. Received {kernel.shape}"

    if separable:
        groups = kernel.shape[1]
        kernel = kernel.view(kernel.shape[1], 1, *kernel.shape[2:])
    else:
        groups = 1

    if fft:
        # fft_conv has the same arguments as regular pytorch conv, depthwise seperable is done by passin groups
        conv_function = fft_conv
    else:
        # old implementation for regular non fft conv
        conv_function = getattr(torch.nn.functional, f"conv{data_dim}d")
    
    return conv_function(x, kernel, bias=bias, padding=padding, stride=1, groups=groups)


# Aliases to handle data dimensionality
conv1d = conv2d = conv3d = conv
fftconv1d = fftconv2d = fftconv3d = fftconv
