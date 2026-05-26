"""Pure-PyTorch fallback for spconv.pytorch (macOS / CPU-only installs)."""

from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def inject_spconv_mock() -> None:
    """Register this module as ``spconv`` and ``spconv.pytorch`` in ``sys.modules``."""
    if "spconv.pytorch" in sys.modules:
        return
    pkg = types.ModuleType("spconv")
    pkg.pytorch = sys.modules[__name__]
    sys.modules["spconv"] = pkg
    sys.modules["spconv.pytorch"] = sys.modules[__name__]

class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size, grid=None, voxel_num=None, indice_dict=None):
        # trellis `SparseTensor.replace` assigns updated feats via `_features`
        self._features = features
        self.indices = indices  # [N, 4] where column 0 is batch index, columns 1, 2, 3 are z, y, x coords
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.grid = grid
        self.voxel_num = voxel_num
        self.indice_dict = indice_dict if indice_dict is not None else {}
        self.benchmark = False
        self.benchmark_record = None
        self.thrust_allocator = None
        self._timer = None
        self.force_algo = None
        self.int8_scale = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    def dense(self):
        # Convert to dense tensor of shape [B, H, W, D, C]
        device = self.features.device
        dtype = self.features.dtype
        B = self.batch_size
        C = self.features.shape[1]
        S = self.spatial_shape
        
        dense = torch.zeros((B, *S, C), dtype=dtype, device=device)
        idx = self.indices.long()
        dense[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3], :] = self.features
        return dense

    def replace_feature(self, new_features):
        out = SparseConvTensor(
            new_features,
            self.indices,
            self.spatial_shape,
            self.batch_size,
            self.grid,
            self.voxel_num,
            self.indice_dict,
        )
        return out

class ConvAlgo:
    Native = 0
    MaskImplicitGemm = 1


def _kernel_tuple(kernel_size):
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size, kernel_size)
    return tuple(kernel_size)


def _spconv_weight_shape(out_channels, in_channels, kernel_size):
    k = _kernel_tuple(kernel_size)
    return (out_channels, k[0], k[1], k[2], in_channels)


def _to_conv3d_weight(spconv_weight: torch.Tensor) -> torch.Tensor:
    """spconv stores [O, Kd, Kh, Kw, I]; PyTorch Conv3d expects [O, I, Kd, Kh, Kw]."""
    return spconv_weight.permute(0, 4, 1, 2, 3).contiguous()


class SubMConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, bias=True, indice_key=None, algo=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.indice_key = indice_key

        p = padding if padding != 0 else (_kernel_tuple(kernel_size)[0] // 2)
        # Match spconv weight layout [O, K, K, K, I]
        self.register_parameter(
            "weight",
            nn.Parameter(torch.zeros(_spconv_weight_shape(out_channels, in_channels, kernel_size))),
        )
        self.register_parameter(
            "bias", nn.Parameter(torch.zeros(out_channels)) if bias else None
        )
        self._padding = p
        self._dilation = dilation

    def forward(self, x):
        # x is a SparseConvTensor
        device = x.features.device
        dtype = x.features.dtype
        B = x.batch_size
        C = x.features.shape[1]
        S = x.spatial_shape

        # Scatter to dense grid [B, C, H, W, D]
        dense = torch.zeros((B, C, *S), dtype=dtype, device=device)
        idx = x.indices.long()
        dense[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]] = x.features

        # Convolve dense grid
        dense_out = F.conv3d(
            dense,
            _to_conv3d_weight(self.weight),
            self.bias,
            stride=1,
            dilation=self._dilation,
            padding=self._padding,
        )

        # Gather features back at original indices (SubMConv3d keeps active coords identical)
        out_features = dense_out[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]]

        return SparseConvTensor(out_features, x.indices, x.spatial_shape, x.batch_size, indice_dict=x.indice_dict)


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, bias=True, indice_key=None, algo=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding if padding is not None else 0
        self.indice_key = indice_key

        self.register_parameter(
            "weight",
            nn.Parameter(torch.zeros(_spconv_weight_shape(out_channels, in_channels, kernel_size))),
        )
        self.register_parameter(
            "bias", nn.Parameter(torch.zeros(out_channels)) if bias else None
        )
        self._stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
        self._dilation = dilation
        self._padding = self.padding

    def forward(self, x):
        # Save current coordinates in the shared dict for future matching in SparseInverseConv3d
        if self.indice_key is not None:
            x.indice_dict[self.indice_key] = (x.indices, x.spatial_shape)

        device = x.features.device
        dtype = x.features.dtype
        B = x.batch_size
        C = x.features.shape[1]
        S = x.spatial_shape

        # Scatter to dense grid [B, C, H, W, D]
        dense = torch.zeros((B, C, *S), dtype=dtype, device=device)
        idx = x.indices.long()
        dense[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]] = x.features

        # Convolve dense grid
        dense_out = F.conv3d(
            dense,
            _to_conv3d_weight(self.weight),
            self.bias,
            stride=self._stride,
            dilation=self._dilation,
            padding=self._padding,
        )

        # Find active coordinates in the output using a convolved binary mask
        mask = torch.zeros((B, 1, *S), dtype=dtype, device=device)
        mask[idx[:, 0], 0, idx[:, 1], idx[:, 2], idx[:, 3]] = 1.0
        
        k_size = self.kernel_size
        if isinstance(k_size, int):
            k_size = (k_size, k_size, k_size)
        weight_ones = torch.ones((1, 1, *k_size), dtype=dtype, device=device)
        
        mask_out = F.conv3d(mask, weight_ones, stride=self.stride, padding=self.padding) > 1e-5
        
        # Active indices: [M, 4] with columns (batch, z, y, x)
        out_indices = mask_out.nonzero()[:, [0, 2, 3, 4]]
        
        # Gather output features
        out_features = dense_out[out_indices[:, 0], :, out_indices[:, 1], out_indices[:, 2], out_indices[:, 3]]
        out_spatial_shape = list(dense_out.shape[2:])
        
        return SparseConvTensor(out_features, out_indices, out_spatial_shape, x.batch_size, indice_dict=x.indice_dict)

class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.indice_key = indice_key

        self.register_parameter(
            "weight",
            nn.Parameter(torch.zeros(_spconv_weight_shape(out_channels, in_channels, kernel_size))),
        )
        self.register_parameter(
            "bias", nn.Parameter(torch.zeros(out_channels)) if bias else None
        )
        self._stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
        self._dilation = dilation

    def forward(self, x):
        device = x.features.device
        dtype = x.features.dtype
        B = x.batch_size
        C = x.features.shape[1]
        S = x.spatial_shape

        # Scatter to dense grid [B, C, H, W, D]
        dense = torch.zeros((B, C, *S), dtype=dtype, device=device)
        idx = x.indices.long()
        dense[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]] = x.features

        # Convolve dense grid
        dense_out = F.conv_transpose3d(
            dense,
            _to_conv3d_weight(self.weight),
            self.bias,
            stride=self._stride,
            dilation=self._dilation,
        )

        # Retrieve original active coordinates from corresponding downsampling layer's key
        if self.indice_key is not None and self.indice_key in x.indice_dict:
            out_indices, out_spatial_shape = x.indice_dict[self.indice_key]
        else:
            # Fallback by computing transposed convolved mask
            mask = torch.zeros((B, 1, *S), dtype=dtype, device=device)
            mask[idx[:, 0], 0, idx[:, 1], idx[:, 2], idx[:, 3]] = 1.0
            
            k_size = self.kernel_size
            if isinstance(k_size, int):
                k_size = (k_size, k_size, k_size)
            weight_ones = torch.ones((1, 1, *k_size), dtype=dtype, device=device)
            
            mask_out = F.conv_transpose3d(mask, weight_ones, stride=self.stride) > 1e-5
            out_indices = mask_out.nonzero()[:, [0, 2, 3, 4]]
            out_spatial_shape = list(dense_out.shape[2:])
            
        # Gather output features
        out_features = dense_out[out_indices[:, 0], :, out_indices[:, 1], out_indices[:, 2], out_indices[:, 3]]
        
        return SparseConvTensor(out_features, out_indices, out_spatial_shape, x.batch_size, indice_dict=x.indice_dict)
