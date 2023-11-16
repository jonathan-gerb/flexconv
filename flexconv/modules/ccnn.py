import torch
from .s4_block import S4Block
from ..flexconv import FlexConv, SeparableFlexConv
from ..modules import norm
from .. import linear
from .resnet import ResNetSequence, ResNetImage, ResNetBase

from functools import partial


class CCNNBlock(S4Block):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_dim: int,
        separable: bool = True,
        normtype: str = "batchnorm",
        dropout: float = 0.15,
        prenorm: bool = True,
        conv_use_fft: bool = True,
        kernel_size: str = "same",
        kernel_omega_0: float = 100,
        kernel_no_hidden: int = 32,
        kernel_no_layers: int = 3
    ):
        """CCNNBlock from 
        "modelling long range dependencies in nd:
        from task-specific to a general purpose cnn" by Knigge et al. 
        https://arxiv.org/pdf/2301.10540.pdf

        Args:
            in_channels (int): nubmer of input channels
            out_channels (int): number of output channels
            n_dims (int): dimensionality of the data
            separable (bool, optional): use seperable convolutions. Defaults to True.
            normtype (str, optional): norm to apply. Defaults to "batchnorm".
            dropout (float, optional): dropout percentage. Defaults to 0.15.
            prenorm (bool, optional): Apply norm before rest of the block. Defaults to True.
            conv_use_fft (bool, optional): Automatically use FFT convolutions for kernel sizes above 100. Defaults to True.
            kernel_size (str, optional): size of the kernel
            kernel_omega_0 (float, optional): weight initialization parameter
            kernel_no_hidden (int, optional): size of hidden layer of the kernel network
            kernel_no_layers (int, optional): number of layers of the kernel network
        """

        if data_dim == 1:
            causal = True
        else:
            causal = False

        if separable:
            ConvType = partial(SeparableFlexConv, data_dim=data_dim, conv_use_fft=conv_use_fft, kernel_size=kernel_size,
                               kernel_omega_0=kernel_omega_0, kernel_no_hidden=kernel_no_hidden, kernel_no_layers=kernel_no_layers, conv_causal=causal)
        else:
            ConvType = partial(FlexConv, data_dim=data_dim, conv_use_fft=conv_use_fft, kernel_size=kernel_size,
                               kernel_omega_0=kernel_omega_0, kernel_no_hidden=kernel_no_hidden, kernel_no_layers=kernel_no_layers, conv_causal=causal)
        NonlinearType = torch.nn.GELU

        # Define NormType
        if normtype.lower() == "batchnorm":
            norm_name = f"BatchNorm{data_dim}d"
        else:
            norm_name = normtype
        if hasattr(norm, normtype):
            lib = norm
        else:
            lib = torch.nn

        NormType = getattr(lib, norm_name)

        LinearType = getattr(linear, f"Linear{data_dim}d")
        DropoutType = getattr(torch.nn, f"Dropout{data_dim}d")

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            ConvType=ConvType,
            NonlinearType=NonlinearType,
            NormType=NormType,
            LinearType=LinearType,
            DropoutType=DropoutType,
            dropout=dropout,
            prenorm=prenorm
        )


class CCNN(ResNetBase):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            no_hidden: int,
            no_blocks: int,
            data_dim: int,
            norm: str = "batchnorm",
            dropout: float = 0.15,
            dropout_in: float = 0,
            block_prenorm: bool = True,
            block_width_factors: list[float] = [0.0],
            downsampling: list = [],
            downsampling_size: int = -1,
            nonlinearity: str = "GELU",
            kernel_type: str = "MAGNet",
            kernel_no_hidden: int = 32,
            kernel_no_layers: int = 3,
            kernel_nonlinear: str = "GELU",
            kernel_norm: str = "Identity",
            kernel_omega_0: float = 976,  # this needs tuning
            kernel_bias: bool = True,
            kernel_size: str = "same",  # what the final size of the kernel is
            kernel_chang_initialize: bool = True,
            # 'Only != 1.0 if FlexConvs are used.' But I don't see it set differently anywhere
            kernel_init_spatial_value: float = 0.125025,
            conv_use_fft: bool = True,
            conv_bias: bool = True,
            conv_padding: str = "same",
            conv_stride: int = 1,
            conv_causal: bool = False,
            mask_type: str = 'gaussian',
            mask_threshold: float = 0.1,
            mask_init_value: float = 0.075,
            mask_dynamic_cropping: bool = True,
            mask_learn_mean: bool = False,
            mask_temperature: float = 0.0,
    ):
        # had to do this one manually :P
        dropout_type = f"Dropout{data_dim}d"
        # set causal true for 1-d data
        if data_dim == 1:
            conv_causal = True

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            no_hidden=no_hidden,
            no_blocks=no_blocks,
            data_dim=data_dim,
            norm=norm,
            dropout=dropout,
            dropout_in=dropout_in,
            dropout_type=dropout_type,
            block_type="S4",
            block_prenorm=block_prenorm,
            block_width_factors=block_width_factors,
            downsampling=downsampling,
            downsampling_size=downsampling_size,
            nonlinearity=nonlinearity,
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
            mask_type=mask_type,
            mask_threshold=mask_threshold,
            mask_init_value=mask_init_value,
            mask_dynamic_cropping=mask_dynamic_cropping,
            mask_learn_mean=mask_learn_mean,
            mask_temperature=mask_temperature,
        )
        self.data_dim = data_dim
        assert data_dim in [
            1, 2], "CCNN only implemented for 1 or 2d data! if you want to use it for N-d, please adjust the forward pass."

    def forward(self, x):
        # Dropout in
        x = self.dropout_in(x)
        # First layers
        out = self.nonlinear(self.norm1(self.conv1(x)))
        # Blocks
        out = self.blocks(out)
        # Final layer on last sequence element
        out = self.out_norm(out)

        if self.data_dim == 1:
            # Take the mean of all predictions until the last element
            out = out.mean(dim=-1, keepdim=True)
            # Pass through final projection layer, squeeze & return
            out = self.out_layer(out)
            return out.squeeze(-1)

        elif self.data_dim == 2:
            # Pool
            out = torch.nn.functional.adaptive_avg_pool2d(
                out,
                (1,) * self.data_dim,
            )
            # Final layer
            out = self.out_layer(out)
            return out.squeeze()
        else:
            raise NotImplementedError(
                "please implement forward for data_dim > 2")
