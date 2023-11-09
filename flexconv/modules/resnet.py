from functools import partial

import torch
from flexconv import SeparableFlexConv, modules
from flexconv import linear
from flexconv.modules import norm
from flexconv.utils import pairwise as pairwise_iterable

class ResNetBase(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        no_hidden: int,
        no_blocks: int,
        data_dim: int,
        norm: str,
        dropout: float,
        dropout_in: float,
        dropout_type: str,
        block_type: str,
        block_prenorm: bool,
        block_width_factors: list[float],
        # After the indices of these blocks place a downsampling layer.
        downsampling: list,
        downsampling_size: int,  # -1
        nonlinearity: str,  # GELU
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
        conv_causal: bool = True,
        mask_type: str = 'gaussian',
        mask_threshold: float = 0.1,
        mask_init_value: float = 0.075,
        mask_dynamic_cropping: bool = True,
        mask_learn_mean: bool = False,
        mask_temperature: float = 0.0,
    ):
        super().__init__()

        # Define dropout_in
        self.dropout_in = torch.nn.Dropout(dropout_in)

        # Unpack conv_type
        # Define partials for types of convs
        ConvType = partial(
            SeparableFlexConv,
            data_dim=data_dim,
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
        # -------------------------

        # Define NormType
        if norm.lower() == "batchnorm":
            norm_name = f"BatchNorm{data_dim}d"
        else:
            norm_name = norm
        if hasattr(norm, norm):
            lib = norm
        else:
            lib = torch.nn
        NormType = getattr(lib, norm_name)

        # Define NonlinearType
        NonlinearType = getattr(torch.nn, nonlinearity)

        # Define LinearType
        LinearType = getattr(linear, f"Linear{data_dim}d")

        # Define DownsamplingType
        DownsamplingType = getattr(torch.nn, f"MaxPool{data_dim}d")

        # Define Dropout layer type
        DropoutType = getattr(torch.nn, dropout_type)

        # Create Input Layers
        self.conv1 = ConvType(in_channels=in_channels,
                              out_channels=no_hidden)
        self.norm1 = NormType(no_hidden)
        self.nonlinear = NonlinearType()

        # Create Blocks
        # -------------------------
        if block_type == "default":
            BlockType = modules.ResNetBlock
        else:
            BlockType = getattr(modules, f"{block_type}Block")
        # 1. Create vector of width_factors:
        # If value is zero, then all values are one
        if block_width_factors[0] == 0.0:
            width_factors = (1,) * no_blocks
        else:
            width_factors = [
                (factor,) * n_blcks
                for factor, n_blcks in pairwise_iterable(
                    block_width_factors
                )
            ]
            width_factors = [
                factor for factor_tuple in width_factors for factor in factor_tuple
            ]
        if len(width_factors) != no_blocks:
            raise ValueError(
                "The size of the width_factors does not matched the number of blocks in the network."
            )
        # 2. Create blocks
        blocks = []
        for i in range(no_blocks):
            # print(f"Block {i}/{no_blocks}")

            if i == 0:
                input_ch = no_hidden
                hidden_ch = int(no_hidden * width_factors[i])
            else:
                input_ch = int(no_hidden * width_factors[i - 1])
                hidden_ch = int(no_hidden * width_factors[i])

            blocks.append(
                BlockType(
                    in_channels=input_ch,
                    out_channels=hidden_ch,
                    ConvType=ConvType,
                    NonlinearType=NonlinearType,
                    NormType=NormType,
                    LinearType=LinearType,
                    DropoutType=DropoutType,
                    dropout=dropout,
                    prenorm=block_prenorm,
                )
            )

            # Check whether we need to add a downsampling block here.
            if i in downsampling:
                blocks.append(DownsamplingType(kernel_size=downsampling_size))

        self.blocks = torch.nn.Sequential(*blocks)
        # -------------------------

        # Define Output Layers:
        # -------------------------
        # 1. Calculate output channels of blocks
        if block_width_factors[0] == 0.0:
            final_no_hidden = no_hidden
        else:
            final_no_hidden = int(no_hidden * block_width_factors[-2])
        # 2. instantiate last layer
        self.out_layer = LinearType(
            in_channels=final_no_hidden, out_channels=out_channels
        )
        # 3. Initialize finallyr
        torch.nn.init.kaiming_normal_(self.out_layer.weight)
        self.out_layer.bias.data.fill_(value=0.0)
        # -------------------------
        if block_type == "S4" and block_prenorm:
            self.out_norm = NormType(final_no_hidden)
        else:
            self.out_norm = torch.nn.Identity()

        # Save variables in self
        self.data_dim = data_dim

    def forward(self, x):
        raise NotImplementedError


class ResNetSequence(ResNetBase):
    def forward(self, x):
        # Dropout in
        x = self.dropout_in(x)
        # First layers
        out = self.nonlinear(self.norm1(self.conv1(x)))
        # Blocks
        out = self.blocks(out)
        # Final layer on last sequence element
        out = self.out_norm(out)
        # Take the mean of all predictions until the last element
        out = out.mean(dim=-1, keepdim=True)
        # Pass through final projection layer, squeeze & return
        out = self.out_layer(out)
        return out.squeeze(-1)


class ResNetImage(ResNetBase):
    def forward(self, x):
        # Dropout in
        x = self.dropout_in(x)
        # First layers
        out = self.nonlinear(self.norm1(self.conv1(x)))
        # Blocks
        out = self.blocks(out)
        # Final layer on last sequence element
        out = self.out_norm(out)
        # Pool
        out = torch.nn.functional.adaptive_avg_pool2d(
            out,
            (1,) * self.data_dim,
        )
        # Final layer
        out = self.out_layer(out)
        return out.squeeze()
