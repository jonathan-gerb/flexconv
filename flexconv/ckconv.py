
import math

import torch
import torch.fft
import torch.nn

from . import kernel_nets
from . import functional as ckconv_F
from . import linear
from .utils import linspace_grid

class CKConvBase(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_dim: int,
        separable: bool,
        kernel_type: str,
        kernel_no_hidden: int,
        kernel_no_layers: int,
        kernel_nonlinear: str,
        kernel_norm: str,
        kernel_omega_0: float,  # this needs tuning
        kernel_bias: bool,
        kernel_size: str,  # what the final size of the kernel is
        kernel_chang_initialize: bool,
        # 'Only != 1.0 if FlexConvs are used.' But I don't see it set differently anywhere
        kernel_init_spatial_value: float,
        conv_use_fft: bool,
        conv_bias: bool,
        conv_padding: str,
        conv_stride: int,
        conv_causal: bool,
        **kwargs,
    ):
        super().__init__()

        # TODO cache

        # Gather kernel nonlinear and norm type
        kernel_norm = getattr(torch.nn, kernel_norm)
        kernel_nonlinear = getattr(torch.nn, kernel_nonlinear)

        # Define the kernel size
        if separable:
            kernel_out_channels = in_channels
        else:
            kernel_out_channels = in_channels * out_channels

        # Create the kernel
        KernelClass = getattr(kernel_nets, kernel_type)
        self.Kernel = KernelClass(
            data_dim=data_dim,
            out_channels=kernel_out_channels,
            hidden_channels=kernel_no_hidden,
            no_layers=kernel_no_layers,
            bias=kernel_bias,
            causal=conv_causal,
            # MFN
            omega_0=kernel_omega_0,
            steerable=False,  # TODO
            init_spatial_value=kernel_init_spatial_value,
            # SIREN
            learn_omega_0=False,  # TODO
            # MLP & RFNet
            NonlinearType=kernel_nonlinear,
            NormType=kernel_norm,
            weight_norm=False,
        )

        # The kernel must have an output_linear layer. Used for chang initialization
        assert hasattr(self.Kernel, "output_linear")

        # Define convolution type
        conv_type = f"conv{data_dim}d"
        if conv_use_fft:
            conv_type = "fft" + conv_type
        self.conv = getattr(ckconv_F, conv_type)

        # Add bias
        if conv_bias:
            if separable:
                bias_size = in_channels
            else:
                bias_size = out_channels
            self.bias = torch.nn.Parameter(torch.Tensor(bias_size))
            self.bias.data.fill_(value=0.0)
        else:
            self.bias = None

        # Save arguments in self
        # ---------------------
        # 1. Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = conv_padding
        self.stride = conv_stride
        self.data_dim = data_dim
        self.kernel_size = kernel_size
        self.conv_use_fft = conv_use_fft
        self.chang_initialize = kernel_chang_initialize
        self.separable = separable
        self.causal = conv_causal
        # 3. Variable placeholders
        self.register_buffer(
            "train_length", torch.zeros(1).int(), persistent=True)
        # 2. Non-persistent values
        self.register_buffer("conv_kernel", torch.zeros(1), persistent=False)
        self.register_buffer("linspace_stepsize",
                             torch.zeros(1), persistent=False)
        self.register_buffer("kernel_positions",
                             torch.zeros(1), persistent=False)
        self.register_buffer("initialized", torch.zeros(
            1).bool(), persistent=False)

    def construct_kernel(self, x):
        # Construct kernel
        # 1. Get kernel positions & Chang-initialize self.Kernel if not done yet.
        kernel_pos = self.handle_kernel_positions(x)
        self.chang_initialization(kernel_pos)
        # 2. Sample the kernel
        x_shape = x.shape
        conv_kernel = self.Kernel(kernel_pos).view(
            -1, x_shape[1], *kernel_pos.shape[2:]
        )
        # 3. Save the sampled kernel for computation of "weight_decay"
        self.conv_kernel = conv_kernel
        return self.conv_kernel

    def handle_kernel_positions(self, x):
        """
        Handles the vector or relative positions which is given to KernelNet.
        """
        if (
            self.kernel_positions.shape[-1] == 1
        ):  # The conv. receives input signals of length > 1
            if self.train_length[0] == 0:
                # Decide the extend of the rel_positions vector
                if self.kernel_size == "full":
                    self.train_length[0] = (2 * x.shape[-1]) - 1
                elif self.kernel_size == "same":
                    self.train_length[0] = x.shape[-1]
                elif int(self.kernel_size) % 2 == 1:
                    # Odd number
                    self.train_length[0] = int(self.kernel_size)
                elif int(self.kernel_size) % 2 == 0:
                    # Even number todo check whether this causes any problems
                    self.train_length[0] = int(self.kernel_size)
                else:
                    raise ValueError(
                        f"The size of the kernel must be either 'full', 'same' or an odd number"
                        f" in string format. Current: {self.kernel_size}"
                    )
            # Creates the vector of relative positions.
            kernel_positions = linspace_grid(
                grid_sizes=self.train_length.repeat(self.data_dim)
            )
            # TODO: Rectangular grids.
            kernel_positions = kernel_positions.unsqueeze(0)
            self.kernel_positions = kernel_positions.type_as(
                self.kernel_positions)
            # -> With form: [batch_size=1, dim, x_dimension, y_dimension, ...]

            # Save the step size for the calculation of dynamic cropping
            # The step is max - min / (no_steps - 1)
            self.linspace_stepsize = (
                (1.0 - (-1.0)) / (self.train_length[0] - 1)
            ).type_as(self.linspace_stepsize)
        return self.kernel_positions

    def chang_initialization(self, kernel_positions):
        if not self.initialized[0] and self.chang_initialize:
            # Initialization - Initialize the last layer of self.Kernel as in Chang et al. (2020)
            with torch.no_grad():
                kernel_size = kernel_positions.reshape(
                    *kernel_positions.shape[: -self.data_dim], -1
                )
                kernel_size = kernel_size.shape[-1]
                if self.separable:
                    normalization_factor = kernel_size
                else:
                    normalization_factor = self.in_channels * kernel_size
                self.Kernel.output_linear.weight.data *= math.sqrt(
                    1.0 / normalization_factor
                )
            # Set the initialization flag to true
            self.initialized[0] = True


class CKConv(CKConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_dim: int,
        kernel_type: str,
        kernel_no_hidden: int,
        kernel_no_layers: int,
        kernel_nonlinear: str,
        kernel_norm: str,
        kernel_omega_0: float,  # this needs tuning
        kernel_bias: bool,
        kernel_size: str,  # what the final size of the kernel is
        kernel_chang_initialize: bool,
        # 'Only != 1.0 if FlexConvs are used.' But I don't see it set differently anywhere
        kernel_init_spatial_value: float,
        conv_use_fft: bool,
        conv_bias: bool,
        conv_padding: str,
        conv_stride: int,
        conv_causal: bool,
        **kwargs,
    ):
        """
        Continuous Kernel Convolution.
        :param in_channels: Number of channels in the input signal
        :param out_channels: Number of channels produced by the convolution
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            data_dim=data_dim,
            separable=False,
            kernel_type=kernel_type,
            kernel_no_hidden=kernel_no_hidden,
            kernel_no_layers=kernel_no_layers,
            kernel_nonlinear=kernel_nonlinear,
            kernel_norm=kernel_norm,
            kernel_omega_0=kernel_omega_0,
            kernel_bias=kernel_bias,
            kernel_size=kernel_size,
            kernel_chang_initialize=kernel_chang_initialize,
            kernel_init_spatial_value=kernel_init_spatial_value,
            conv_use_fft=conv_use_fft,
            conv_bias=conv_bias,
            conv_padding=conv_padding,
            conv_stride=conv_stride,
            conv_causal=conv_causal,
        )

    def forward(self, x):
        # 1. Construct kernel
        conv_kernel = self.construct_kernel(x)
        # 4. Compute convolution & return result
        return self.conv(x, conv_kernel, self.bias, separable=False, causal=self.causal)


class SeparableCKConv(CKConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_dim: int,
        kernel_type: str = "MAGNet",
        kernel_no_hidden: int = 32,
        kernel_no_layers: int = 3,
        kernel_nonlinear: str = "GELU",
        kernel_norm: str = "Identity",
        kernel_omega_0: float = 100,  # this needs tuning
        kernel_bias: bool = True,
        kernel_size: str = "same",  # what the final size of the kernel is
        kernel_chang_initialize: bool = True,
        # 'Only != 1.0 if FlexConvs are used.' But I don't see it set differently anywhere
        kernel_init_spatial_value: float = 1.0,
        conv_use_fft: bool = True,
        conv_bias: bool = True,
        conv_padding: str = "same",
        conv_stride: int = 1,
        conv_causal: bool = True,
        **kwargs,
    ):
        """
        Continuous Kernel Convolution.
        :param in_channels: Number of channels in the input signal
        :param out_channels: Number of channels produced by the convolution
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            data_dim=data_dim,
            kernel_type=kernel_type,
            separable=True,
            kernel_no_hidden=kernel_no_hidden,
            kernel_no_layers=kernel_no_layers,
            kernel_nonlinear=kernel_nonlinear,
            kernel_norm=kernel_norm,
            kernel_omega_0=kernel_omega_0,
            kernel_bias=kernel_bias,
            kernel_size=kernel_size,
            kernel_chang_initialize=kernel_chang_initialize,
            kernel_init_spatial_value=kernel_init_spatial_value,
            conv_use_fft=conv_use_fft,
            conv_bias=conv_bias,
            conv_padding=conv_padding,
            conv_stride=conv_stride,
            conv_causal=conv_causal,
        )
        # Create the point-wise convolution
        ChannelMixerClass = getattr(linear, f"Linear{data_dim}d")
        self.channel_mixer = ChannelMixerClass(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=conv_bias,
        )
        # initialize
        torch.nn.init.kaiming_uniform_(
            self.channel_mixer.weight, nonlinearity="linear")
        if self.channel_mixer.bias is not None:
            torch.nn.init._no_grad_fill_(self.channel_mixer.bias, 0.0)

    def forward(self, x):
        # 1. Construct kernel
        conv_kernel = self.construct_kernel(x)
        # 4. Compute depthwise convolution
        out = self.channel_mixer(
            self.conv(x, conv_kernel, self.bias,
                      separable=True, causal=self.causal)
        )
        return out
