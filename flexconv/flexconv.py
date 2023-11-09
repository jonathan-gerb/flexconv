import torch
from .ckconv import CKConvBase
from . import linear
from . import functional as ckconv_F

class FlexConvBase(CKConvBase):
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
        mask_type: str,
        mask_threshold: float,
        mask_init_value: float,
        mask_dynamic_cropping: bool,
        mask_learn_mean: bool,
        mask_temperature: float,
        **kwargs,
    ):

        if mask_type == "gaussian":
            init_spatial_value = mask_init_value * 1.667
        elif mask_type == "sigmoid":
            init_spatial_value = 1.0 - mask_init_value
        elif mask_type == "hann":
            init_spatial_value = mask_init_value
        else:
            raise NotImplementedError(
                f"Mask of type '{mask_type}' not implemented.")

        # Overwrite init_spatial value
        kernel_init_spatial_value = init_spatial_value

        # call super class
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
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
            separable=separable,
        )

        # define convolution types
        conv_types = {
            "spatial": f"conv{self.data_dim}d",
            "fft": f"fftconv{self.data_dim}d",
        }
        # Save convolution functions in self:
        for (key, value) in conv_types.items():
            conv_types[key] = getattr(ckconv_F, value)
        self.conv_types = conv_types

        # Define mask constructor
        self.mask_constructor = globals()[f"{mask_type}_mask"]
        # Define root finder & cropper functions
        if self.causal:
            root_function = f"{mask_type}_min_root"
            crop_function = self.crop_kernel_positions_causal
        else:
            root_function = f"{mask_type}_max_abs_root"
            crop_function = self.crop_kernel_positions_centered
        self.root_function = globals()[root_function]
        self.crop_function = crop_function

        # Define learnable parameters of the mask
        mask_width_param = {
            "gaussian": {
                1: torch.Tensor([mask_init_value]),
                2: torch.Tensor([mask_init_value, mask_init_value]),
            },
            "sigmoid": {  # Not used
                1: torch.Tensor([0.0]),
                2: torch.Tensor([0.0, 0.0]),
            },
            "hann": {  # Not used
                1: torch.Tensor([init_spatial_value]),
                2: torch.Tensor([init_spatial_value, init_spatial_value]),
            },
        }[mask_type][self.data_dim]
        self.mask_width_param = torch.nn.Parameter(mask_width_param)

        mask_mean_param = {
            "gaussian": {
                1: torch.Tensor([1.0 if self.causal else 0.0]),
                2: torch.Tensor([0.0, 0.0]),
            },
            "sigmoid": {  # Not used
                1: torch.Tensor([mask_init_value]),  # TODO: Non-causal
                2: torch.Tensor([mask_init_value, mask_init_value]),
            },
            # "hann": {  # Not used
            #     1: torch.Tensor([1.0 - 0.5 / init_spatial_value]),  # TODO: Non-causal
            #     2: torch.Tensor(
            #         [1.0 - 0.5 / init_spatial_value, 1.0 - 0.5 / init_spatial_value]
            #     ),
            # },
        }[mask_type][self.data_dim]
        if mask_learn_mean:
            self.mask_mean_param = torch.nn.Parameter(mask_mean_param)
        else:
            self.register_buffer("mask_mean_param", mask_mean_param)

        # Define threshold of mask for dynamic cropping
        mask_threshold = mask_threshold * torch.ones(1)
        self.register_buffer("mask_threshold", mask_threshold, persistent=True)

        self.mask_temperature = mask_temperature

        # Save values in self
        self.dynamic_cropping = mask_dynamic_cropping

    def crop_kernel_positions_causal(
        self,
        kernel_pos: torch.Tensor,
        root: float,
    ):
        # In 1D, only one part of the array must be cut.
        if abs(root) >= 1.0:
            return kernel_pos
        else:
            # We not find the index from which the positions must be cropped
            # index = value - initial_linspace_value / step_size
            index = (
                torch.floor((root + 1.0) / self.linspace_stepsize).int().item()
            )  # TODO: zero?
            return kernel_pos[..., index:]

    def crop_kernel_positions_centered(
        self,
        kernel_pos: torch.Tensor,
        root: float,
    ):
        crops_needed = torch.abs(root) < 1.0

        mid_point = (
            torch.tensor(
                kernel_pos.shape[-self.data_dim:], device=kernel_pos.device)
            // 2
        )
        index = torch.ceil((root - 0.0) / self.linspace_stepsize + 1e-8).int()
        index_1 = mid_point - (index - 1)
        index_2 = mid_point + index

        slices = [
            slice(None),
        ] * len(kernel_pos.shape)

        for i, crop_needed in enumerate(crops_needed):
            if not crop_needed:
                pass
            else:
                slices[i + 2] = slice(index_1[i], index_2[i])

        return kernel_pos[slices]

    def construct_masked_kernel(self, x):
        # Construct kernel
        # 1. Get kernel positions
        kernel_pos = self.handle_kernel_positions(x)
        
        # 2. dynamic cropping
        if self.dynamic_cropping:
            # Based on the current mean and sigma values, compute the [min, max] values of the array.
            with torch.no_grad():
                roots = self.root_function(
                    thresh=self.mask_threshold,
                    mean=self.mask_mean_param,
                    sigma=self.mask_width_param,
                    temperature=self.mask_temperature,  # Only used for sigmoid
                )
                kernel_pos = self.crop_function(kernel_pos, roots)

        # 3. chang-initialize self.Kernel if not done yet.
        self.chang_initialization(kernel_pos)
        # 4. sample the kernel
        x_shape = x.shape
        conv_kernel = self.Kernel(kernel_pos).view(
            -1, x_shape[1], *kernel_pos.shape[2:]
        )
        # 5. construct mask and multiply with conv-kernel
        mask = self.mask_constructor(
            kernel_pos,
            self.mask_mean_param.view(1, -1, *(1,) * self.data_dim),
            self.mask_width_param.view(1, -1, *(1,) * self.data_dim),
            temperature=self.mask_temperature,
        )
        self.conv_kernel = mask * conv_kernel
        # Return the masked kernel
        return self.conv_kernel


class FlexConv(FlexConvBase):
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
        conv_causal: bool = False,
        mask_type: str = 'gaussian',
        mask_threshold: float = 0.1,
        mask_init_value: float = 0.075,
        mask_dynamic_cropping: bool = True,
        mask_learn_mean: bool = False,
        mask_temperature: float = 0.0,
    ):
        # call super class
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
            mask_type=mask_type,
            mask_threshold=mask_threshold,
            mask_init_value=mask_init_value,
            mask_dynamic_cropping=mask_dynamic_cropping,
            mask_learn_mean=mask_learn_mean,
            mask_temperature=mask_temperature,
        )

    def forward(self, x):
        # 1. Compute the masked kernel
        conv_kernel = self.construct_masked_kernel(x)
        # 2. Compute convolution & return result
        size = torch.tensor(conv_kernel.shape[2:])
        # if the kernel is larger than 100, use fftconv, see experiments here https://github.com/fkodom/fft-conv-pytorch
        if self.conv_use_fft and torch.all(size > 100):
            conv_type = self.conv_types["fft"]
        else:
            conv_type = self.conv_types["spatial"]
        out = conv_type(x, conv_kernel, self.bias,
                        separable=False, causal=self.causal)
        return out


class SeparableFlexConv(FlexConvBase):
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
        conv_causal: bool = False,
        mask_type: str = 'gaussian',
        mask_threshold: float = 0.1,
        mask_init_value: float = 0.075,
        mask_dynamic_cropping: bool = True,
        mask_learn_mean: bool = False,
        mask_temperature: float = 0.0,
    ):
        # call super class
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            data_dim=data_dim,
            separable=True,
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
        # Create the point-wise convolution
        ChannelMixerClass = getattr(linear, f"Linear{data_dim}d")
        self.channel_mixer = ChannelMixerClass(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=conv_bias,
        )
        # initialize
        torch.nn.init.kaiming_normal_(self.channel_mixer.weight)
        if self.channel_mixer.bias is not None:
            torch.nn.init._no_grad_fill_(self.channel_mixer.bias, 0.0)

    def forward(self, x):
        # 1. Compute the masked kernel
        conv_kernel = self.construct_masked_kernel(x)
        # 2. Select convolution type
        size = torch.tensor(conv_kernel.shape[2:])
        # if the kernel is larger than 100, use fftconv, see experiments here https://github.com/fkodom/fft-conv-pytorch
        if self.conv_use_fft and torch.all(size > 100):
            conv_type = self.conv_types["fft"]
        else:
            conv_type = self.conv_types["spatial"]
        # 3. Compute depthwise convolution
        out = self.channel_mixer(
            conv_type(x, conv_kernel, self.bias,
                      separable=True, causal=self.causal)
        )
        return out


###############################
# Gaussian Masks / Operations #
###############################
def gaussian_mask(
    kernel_pos: torch.Tensor,
    mask_mean_param: torch.Tensor,
    mask_width_param: torch.Tensor,
    **kwargs,
):
    # mask.shape = [1, 1, Y, X] in 2D or [1, 1, X] in 1D
    return torch.exp(
        -0.5
        * (1.0 / (mask_width_param ** 2 + 1e-8) * (kernel_pos - mask_mean_param) ** 2).sum(
            1, keepdim=True
        )
    )


def gaussian_inv_thresh(
    thresh: float,
    mean: float,
    sigma: float,
    **kwargs,
):
    # Based on the threshold value, compute the value of the roots
    aux = sigma * torch.sqrt(-2.0 * torch.log(thresh))
    return torch.stack([mean - aux, mean + aux], dim=1)


def gaussian_min_root(
    thresh: float,
    mean: float,
    sigma: float,
    **kwargs,
):
    return torch.min(gaussian_inv_thresh(thresh, mean, sigma))


def gaussian_max_abs_root(
    thresh: float,
    mean: float,
    sigma: float,
    **kwargs,
):
    return torch.max(torch.abs(gaussian_inv_thresh(thresh, mean, sigma)), dim=1).values


###############################
# Sigmoid Masks / Operations #
###############################
def sigmoid_mask_1d(
    rel_positions: torch.Tensor,
    mask_params: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return torch.sigmoid(
        kwargs["temperature"] * (rel_positions[0, 0] - mask_params[0, 0])
    )


def sigmoid_inv_thresh(
    thresh: float,
    mean: float,
    **kwargs,
):
    # Based on the threshold value, compute the value of the root
    #  = - 1/temp * ln(1/thresh - 1) + offset
    return -1.0 / kwargs["temperature"] * torch.log(1.0 / thresh - 1.0) + mean


# Alias for the function
sigmoid_min_root = sigmoid_inv_thresh


###############################
# Sigmoid Masks / Operations #
###############################
def hann_mask_1d(
    rel_positions: torch.Tensor,
    mask_params: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return hann_mask(rel_positions[0, 0], mask_params[0, 0], mask_params[0, 1])


def hann_mask(
    x: torch.Tensor,
    mean: float,
    scale: float,
) -> torch.Tensor:
    return torch.sin(torch.pi * scale * (x - mean)) ** 2


def hann_inv_thresh(
    thresh: float,
    mean: float,
    sigma: float,
    **kwargs,
):
    return mean


# Alias for the function
hann_min_root = hann_inv_thresh
