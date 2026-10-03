"""Shared-parameter tiny fine matching head (eq 6-7).

q' = q + Ed(q), r' = r + Ed(r), with one encoder Ed.
z = LN([q'; r'] + Ep([q'; r'])).
Ed and Ep are two-linear GELU MLPs. Two axis heads predict offset
distributions over bins uniformly spaced on [-0.5, 0.5] (Appendix A, 17
bins). Soft-argmax of that distribution is the continuous displacement.
The same head runs on (q, r) and on (r, q).

Appendix A scales the displacement by 8 to the resized-image pixel range
[-4, 4]. delta is the bin coordinate; delta_px is that coordinate times
pixel_scale (default 8). Each axis head also emits one uncertainty logit,
mapped by a sigmoid. The RLE clamp to [0.1, 1] is a training detail and
is not applied here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

N_BINS = 17
PIXEL_SCALE = 8.0


def _init_linear(key, in_dim, out_dim):
    w = jax.random.normal(key, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
    b = jnp.zeros((out_dim,))
    return {"w": w, "b": b}


def _apply_linear(params, x):
    return x @ params["w"] + params["b"]


def init_mlp(key, in_dim, hidden, out_dim):
    k1, k2 = jax.random.split(key)
    return {
        "fc1": _init_linear(k1, in_dim, hidden),
        "fc2": _init_linear(k2, hidden, out_dim),
    }


def apply_mlp(params, x):
    hidden = jax.nn.gelu(_apply_linear(params["fc1"], x))
    return _apply_linear(params["fc2"], hidden)


def _init_ln(dim):
    return {"gamma": jnp.ones((dim,)), "beta": jnp.zeros((dim,))}


def _apply_ln(params, x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    y = (x - mean) * jax.lax.rsqrt(var + eps)
    return y * params["gamma"] + params["beta"]


def init_fine_head(key, dim, ed_hidden=None, ep_hidden=None, axis_hidden=None, n_bins=N_BINS):
    """One parameter tree. Ed is stored once and applied to both sides."""
    if ed_hidden is None:
        ed_hidden = dim
    if ep_hidden is None:
        ep_hidden = 2 * dim
    if axis_hidden is None:
        axis_hidden = dim
    k_ed, k_ep, k_x, k_y = jax.random.split(key, 4)
    return {
        "ed": init_mlp(k_ed, dim, ed_hidden, dim),
        "ep": init_mlp(k_ep, 2 * dim, ep_hidden, 2 * dim),
        "ln": _init_ln(2 * dim),
        "head_x": init_mlp(k_x, 2 * dim, axis_hidden, n_bins + 1),
        "head_y": init_mlp(k_y, 2 * dim, axis_hidden, n_bins + 1),
    }


def bin_coordinates(n_bins=N_BINS, low=-0.5, high=0.5):
    """Offset bin centers, uniform on [low, high], endpoints included."""
    return jnp.linspace(low, high, n_bins)


def soft_argmax(logits, coordinates):
    """Soft-argmax of offset logits. A one-hot bin returns that bin coordinate."""
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.sum(probs * coordinates, axis=-1)


def soft_argmax_probs(probs, coordinates):
    """Expectation of bin coordinates under an already-normalized distribution."""
    return jnp.sum(probs * coordinates, axis=-1)


def encode_side(params, x):
    """Ed(x). The same params['ed'] is used for the query and the reference."""
    return apply_mlp(params["ed"], x)


def pair_embed(params, q, r):
    """Eq 6-7. Returns q', r', z."""
    q_prime = q + encode_side(params, q)
    r_prime = r + encode_side(params, r)
    pair = jnp.concatenate([q_prime, r_prime], axis=-1)
    pair = pair + apply_mlp(params["ep"], pair)
    z = _apply_ln(params["ln"], pair)
    return q_prime, r_prime, z


def _axis_from_z(params, z, pixel_scale):
    n_bins = int(params["head_x"]["fc2"]["b"].shape[0]) - 1
    coords = bin_coordinates(n_bins)
    raw_x = apply_mlp(params["head_x"], z)
    raw_y = apply_mlp(params["head_y"], z)
    dx = soft_argmax(raw_x[..., :n_bins], coords)
    dy = soft_argmax(raw_y[..., :n_bins], coords)
    delta = jnp.stack([dx, dy], axis=-1)
    sigma = jax.nn.sigmoid(jnp.stack([raw_x[..., n_bins], raw_y[..., n_bins]], axis=-1))
    return {
        "delta": delta,
        "delta_px": delta * pixel_scale,
        "sigma": sigma,
    }


def refine_direction(params, q, r, pixel_scale=PIXEL_SCALE):
    """One direction: q is the query, r is the reference."""
    q_prime, r_prime, z = pair_embed(params, q, r)
    out = _axis_from_z(params, z, pixel_scale)
    out["q_prime"] = q_prime
    out["r_prime"] = r_prime
    out["z"] = z
    return out


def fine_match(params, q, r, pixel_scale=PIXEL_SCALE):
    """Bidirectional refinement with one shared head."""
    return {
        "forward": refine_direction(params, q, r, pixel_scale),
        "backward": refine_direction(params, r, q, pixel_scale),
    }
