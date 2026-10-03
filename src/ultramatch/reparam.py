"""Structural reparameterization for the feature extractor (sec 3.1).

Training-time forward of one block is the sum of parallel convolutions
(3x3, 1x1, 1x3, 3x1) followed by one shared BatchNorm. At inference the
padded kernels are added into a single 3x3 and the shared BN is folded
into that convolution. This is not per-branch BN (RepVGG-style); the
paper puts one BN after the sum.

Weights are HWIO. Activations are NHWC. Population variance (divide by N)
is the statistic folded into the kernel, matching the training forward.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

BN_EPS = 1e-5


def _conv_weight(key, kh, kw, cin, cout):
    fan_in = cin * kh * kw
    std = jnp.sqrt(2.0 / fan_in)
    return jax.random.normal(key, (kh, kw, cin, cout)) * std


def init_rep_block(key, cin, cout):
    """One reparameterizable block, cin -> cout, stride 1."""
    keys = jax.random.split(key, 4)
    zeros = jnp.zeros((cout,))
    return {
        "w_3x3": _conv_weight(keys[0], 3, 3, cin, cout),
        "b_3x3": zeros,
        "w_1x1": _conv_weight(keys[1], 1, 1, cin, cout),
        "b_1x1": zeros,
        "w_1x3": _conv_weight(keys[2], 1, 3, cin, cout),
        "b_1x3": zeros,
        "w_3x1": _conv_weight(keys[3], 3, 1, cin, cout),
        "b_3x1": zeros,
        "bn_scale": jnp.ones((cout,)),
        "bn_bias": zeros,
        "bn_mean": zeros,
        "bn_var": jnp.ones((cout,)),
    }


def init_rep_stack(key, channel_pairs):
    """Tiny stack. channel_pairs is a sequence of (cin, cout)."""
    keys = jax.random.split(key, len(channel_pairs))
    blocks = [
        init_rep_block(k, cin, cout) for k, (cin, cout) in zip(keys, channel_pairs)
    ]
    return {"blocks": blocks}


def _same_padding(kh, kw):
    ph = kh - 1
    pw = kw - 1
    return ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2))


def _conv2d(x, weight, bias):
    kh, kw = weight.shape[0], weight.shape[1]
    y = jax.lax.conv_general_dilated(
        x,
        weight,
        window_strides=(1, 1),
        padding=_same_padding(kh, kw),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + jnp.reshape(bias, (1, 1, 1, -1))


def sum_branches(params, x):
    """Training-time pre-activation: sum of the four conv branches."""
    y = _conv2d(x, params["w_3x3"], params["b_3x3"])
    y = y + _conv2d(x, params["w_1x1"], params["b_1x1"])
    y = y + _conv2d(x, params["w_1x3"], params["b_1x3"])
    y = y + _conv2d(x, params["w_3x1"], params["b_3x1"])
    return y


def batch_stats(y):
    """Mean and population variance over N, H, W. Same values the training BN uses."""
    mean = jnp.mean(y, axis=(0, 1, 2))
    var = jnp.var(y, axis=(0, 1, 2))
    return mean, var


def _apply_bn(y, mean, var, scale, shift, eps):
    inv = jax.lax.rsqrt(var + eps)
    return (y - mean) * inv * scale + shift


def apply_rep_block(params, x, training=False, eps=BN_EPS):
    """Training-time block. training=True uses the batch mean and variance."""
    summed = sum_branches(params, x)
    if training:
        mean, var = batch_stats(summed)
    else:
        mean, var = params["bn_mean"], params["bn_var"]
    return _apply_bn(summed, mean, var, params["bn_scale"], params["bn_bias"], eps)


def apply_rep_stack(params, x, training=False, eps=BN_EPS):
    for block in params["blocks"]:
        x = apply_rep_block(block, x, training=training, eps=eps)
    return x


def _pad_kernel_to_3x3(weight):
    kh, kw = weight.shape[0], weight.shape[1]
    if (kh, kw) == (3, 3):
        return weight
    ph = 3 - kh
    pw = 3 - kw
    top = ph // 2
    left = pw // 2
    return jnp.pad(
        weight,
        ((top, ph - top), (left, pw - left), (0, 0), (0, 0)),
    )


def _fold_bn(kernel, bias, scale, shift, mean, var, eps):
    # y = scale * (W x + b - mean) / sqrt(var + eps) + shift
    factor = scale * jax.lax.rsqrt(var + eps)
    kernel_f = kernel * jnp.reshape(factor, (1, 1, 1, -1))
    bias_f = factor * (bias - mean) + shift
    return kernel_f, bias_f


def fuse_rep_block(params, mean=None, var=None, eps=BN_EPS):
    """Fuse padded branch kernels and fold the shared BN into one 3x3 conv.

    Pass mean and var to match a specific training forward (batch statistics).
    Omit them to fold the stored running statistics used in eval mode.
    Returns (kernel, bias) with kernel shape (3, 3, cin, cout).
    """
    kernel = (
        _pad_kernel_to_3x3(params["w_3x3"])
        + _pad_kernel_to_3x3(params["w_1x1"])
        + _pad_kernel_to_3x3(params["w_1x3"])
        + _pad_kernel_to_3x3(params["w_3x1"])
    )
    bias = params["b_3x3"] + params["b_1x1"] + params["b_1x3"] + params["b_3x1"]
    if mean is None:
        mean = params["bn_mean"]
    if var is None:
        var = params["bn_var"]
    return _fold_bn(kernel, bias, params["bn_scale"], params["bn_bias"], mean, var, eps)


def fuse_rep_stack(params, stats=None, eps=BN_EPS):
    """Fuse every block. stats is an optional list of (mean, var) pairs."""
    fused = []
    for i, block in enumerate(params["blocks"]):
        if stats is None:
            fused.append(fuse_rep_block(block, eps=eps))
        else:
            mean, var = stats[i]
            fused.append(fuse_rep_block(block, mean=mean, var=var, eps=eps))
    return fused


def apply_fused_conv(kernel, bias, x):
    """Inference forward of one fused 3x3."""
    return _conv2d(x, kernel, bias)


def apply_fused_stack(fused_blocks, x):
    for kernel, bias in fused_blocks:
        x = apply_fused_conv(kernel, bias, x)
    return x
