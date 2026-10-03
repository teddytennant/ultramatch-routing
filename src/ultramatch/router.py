"""Transport Path Routing (sec 3.3) and routing losses (eq 8-9).

Defaults from sec 3.6 and Appendix A:
  B = 4, R = 6 at inference, halo h = 1,
  tau_r = 0.1, tau_c = 0.1, mu = 0.5, lambda_rank = 0.5,
  descriptor dim 64.

tau_r is a positive temperature. The paper does specify a number (0.1),
so that is the default rather than 1.0.

Eq 8-9 are written in one direction. Appendix A also averages the reverse
direction; this module implements the displayed equations (one direction).

Working storage for Dual-Softmax is the edge list of
G = union_u T_B(u) x C_u, not an N x N matrix.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_max, segment_sum

DEFAULT_BLOCK_SIZE = 4
DEFAULT_ROUTING_SIZE = 6
DEFAULT_HALO = 1
DEFAULT_TAU_R = 0.1
DEFAULT_TAU_C = 0.1
DEFAULT_MU = 0.5
DEFAULT_LAMBDA_RANK = 0.5
DEFAULT_DESC_DIM = 64


def init_router(key, in_channels, desc_dim=DEFAULT_DESC_DIM):
    """Shared 1x1 projection phi, plus omega_hat initialized at 0."""
    k_w, _ = jax.random.split(key)
    std = 1.0 / jnp.sqrt(jnp.asarray(in_channels, dtype=jnp.float32))
    return {
        "phi_w": jax.random.normal(k_w, (in_channels, desc_dim)) * std,
        "phi_b": jnp.zeros((desc_dim,)),
        "omega_hat": jnp.array(0.0, dtype=jnp.float32),
    }


def prior_weight(omega_hat):
    """omega = tanh(omega_hat). Zero at initialization."""
    return jnp.tanh(omega_hat)


def l2_normalize(x, axis=-1, eps=1e-12):
    norm = jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True))
    return x / jnp.maximum(norm, eps)


def align_to_routing_grid(features, block_size, eighth_hw):
    """A(·): align features so each spatial unit is one B x B block on the 1/8 grid.

    eighth_hw is the spatial size of the 1/8 token grid. The routing grid is
    (H8 / B, W8 / B).

    If features are already on that grid (native 1/32 features when B = 4),
    A is the identity. If features are an integer multiple of the routing
    grid (1/8 features, or any finer aligned map), A average-pools each
    routing cell. That is the transfer setting in Appendix B, which pools
    4 x 4 on 1/8 features.
    """
    features = jnp.asarray(features)
    if features.ndim != 3:
        raise ValueError("features must have shape (H, W, C)")
    h8, w8 = int(eighth_hw[0]), int(eighth_hw[1])
    if h8 % block_size or w8 % block_size:
        raise ValueError("1/8 grid must be divisible by block size B")
    out_h = h8 // block_size
    out_w = w8 // block_size
    height, width, channels = features.shape
    if (height, width) == (out_h, out_w):
        return features
    if height % out_h or width % out_w:
        raise ValueError(
            f"cannot align features {(height, width)} to routing grid {(out_h, out_w)}"
        )
    kh = height // out_h
    kw = width // out_w
    pooled = features.reshape(out_h, kh, out_w, kw, channels).mean(axis=(1, 3))
    return pooled


def routing_descriptors(phi_w, phi_b, features, block_size, eighth_hw, eps=1e-12):
    """Eq 1. D = Norm2(Flatten(phi(A(F)))).

    Flatten turns (Hr, Wr, d) into (Hr * Wr, d) in row-major order, which is
    the block index u = br * Wr + bc. Norm2 is per-descriptor L2, not a single
    norm of the whole flattened vector: d_u is the descriptor of block u
    (Appendix A: l2 normalization, dr = 64).
    """
    aligned = align_to_routing_grid(features, block_size, eighth_hw)
    projected = aligned @ phi_w + phi_b
    flat = projected.reshape(-1, projected.shape[-1])
    return l2_normalize(flat, eps=eps), (aligned.shape[0], aligned.shape[1])


def routing_affinity(d0, d1, tau_r=DEFAULT_TAU_R):
    """Eq 2, scores only. S^r_uv = (d0_u)^T d1_v / tau_r."""
    tau = jnp.asarray(tau_r)
    if float(tau) <= 0.0:
        raise ValueError("tau_r must be positive")
    return (d0 @ d1.T) / tau


def top_r_indices(scores, routing_size=DEFAULT_ROUTING_SIZE):
    """N_r(u) = TopR(S_u,:). Returns (U, K) target-block indices, K = min(R, V)."""
    if routing_size <= 0:
        raise ValueError("R must be positive")
    k = min(int(routing_size), int(scores.shape[-1]))
    return jnp.argsort(-scores, axis=-1)[:, :k]


def _block_origin(block_index, n_block_w, block_size):
    br = int(block_index) // int(n_block_w)
    bc = int(block_index) % int(n_block_w)
    return br * block_size, bc * block_size


def _halo_bounds(r0, c0, block_size, halo, height, width):
    return (
        max(0, r0 - halo),
        min(height, r0 + block_size + halo),
        max(0, c0 - halo),
        min(width, c0 + block_size + halo),
    )


def candidate_tokens(target_blocks, n_block_w, block_size, halo, height, width):
    """Eq 3 for one source block: union of H_h(T_B(v)) over routed v, overlaps merged.

    Tokens are flat row-major indices. The result is sorted so the set is
    independent of route order.
    """
    seen = set()
    for v in target_blocks:
        r0, c0 = _block_origin(v, n_block_w, block_size)
        rs, re, cs, ce = _halo_bounds(r0, c0, block_size, halo, height, width)
        for r in range(rs, re):
            for c in range(cs, ce):
                seen.add(r * width + c)
    return sorted(seen)


def block_of_token(token_index, width, block_size, n_block_w):
    """b(j): target block containing token j. Halo does not change ownership."""
    token_index = jnp.asarray(token_index)
    row = token_index // width
    col = token_index % width
    return (row // block_size) * n_block_w + (col // block_size)


def _route_lists(routes, n_src):
    if isinstance(routes, (list, tuple)):
        if len(routes) != n_src:
            raise ValueError(f"expected {n_src} source routes, got {len(routes)}")
        return [list(map(int, row)) for row in routes]
    arr = np.asarray(routes)
    if arr.ndim != 2 or arr.shape[0] != n_src:
        raise ValueError(f"routes must be (U, R) with U={n_src}")
    return [list(map(int, row)) for row in arr]


def build_support(routes, block_size, src_hw, tgt_hw, halo):
    """Sparse support G = union_u T_B(u) x C_u.

    Source tokens are the B x B block, not halo-expanded. Target candidates
    are the merged halo expansion of the routed blocks. tgt_block is b(j),
    the block that contains token j, which can differ from the routed block
    when j sits in the halo.
    """
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    tgt_h, tgt_w = int(tgt_hw[0]), int(tgt_hw[1])
    if any(s % block_size for s in (src_h, src_w, tgt_h, tgt_w)):
        raise ValueError("1/8 grids must be divisible by block size B")
    n_src_w = src_w // block_size
    n_tgt_w = tgt_w // block_size
    n_src = (src_h // block_size) * n_src_w
    n_tgt = (tgt_h // block_size) * n_tgt_w
    route_lists = _route_lists(routes, n_src)

    src_index = []
    tgt_index = []
    src_block = []
    for u, routed in enumerate(route_lists):
        r0, c0 = _block_origin(u, n_src_w, block_size)
        src_tokens = [
            r * src_w + c
            for r in range(r0, r0 + block_size)
            for c in range(c0, c0 + block_size)
        ]
        cands = candidate_tokens(routed, n_tgt_w, block_size, halo, tgt_h, tgt_w)
        for i in src_tokens:
            for j in cands:
                src_index.append(i)
                tgt_index.append(j)
                src_block.append(u)

    src_index = jnp.asarray(src_index, dtype=jnp.int32)
    tgt_index = jnp.asarray(tgt_index, dtype=jnp.int32)
    src_block = jnp.asarray(src_block, dtype=jnp.int32)
    if src_index.shape[0] == 0:
        tgt_block = jnp.zeros((0,), dtype=jnp.int32)
    else:
        tgt_block = block_of_token(tgt_index, tgt_w, block_size, n_tgt_w).astype(jnp.int32)
    return {
        "src_index": src_index,
        "tgt_index": tgt_index,
        "src_block": src_block,
        "tgt_block": tgt_block,
        "num_src_tokens": src_h * src_w,
        "num_tgt_tokens": tgt_h * tgt_w,
        "n_src_blocks": n_src,
        "n_tgt_blocks": n_tgt,
        "n_src_w": n_src_w,
        "n_tgt_w": n_tgt_w,
    }


def sparse_similarity(f0, f1, support, routing_scores, omega, tau_c=DEFAULT_TAU_C):
    """Eq 4, only on routed pairs.

    S_ij = (F0_i)^T F1_j / (C * tau_c) + omega * S^r_{u, b(j)}
    C is the token channel count. b(j) is support['tgt_block'], not the
    route that admitted j via the halo.
    """
    f0 = jnp.asarray(f0).reshape(-1, f0.shape[-1])
    f1 = jnp.asarray(f1).reshape(-1, f1.shape[-1])
    channels = f0.shape[-1]
    dots = jnp.sum(f0[support["src_index"]] * f1[support["tgt_index"]], axis=-1)
    prior = routing_scores[support["src_block"], support["tgt_block"]]
    return dots / (channels * tau_c) + omega * prior


def _segment_softmax(values, segment_ids, num_segments):
    values = jnp.asarray(values, dtype=jnp.float32)
    segment_ids = jnp.asarray(segment_ids, dtype=jnp.int32)
    if values.shape[0] == 0:
        return values
    seg_max = segment_max(values, segment_ids, num_segments)
    shifted = values - seg_max[segment_ids]
    exp = jnp.exp(shifted)
    seg_sum = segment_sum(exp, segment_ids, num_segments)
    return exp / seg_sum[segment_ids]


def sparse_dual_softmax(src_index, tgt_index, scores, num_src_tokens, num_tgt_tokens):
    """Eq 5 on the sparse support. No N x N working matrix.

    P_ij = softmax over routed targets of S_ij
           * softmax over routed sources of S_ij.
    Row groups are source tokens. Column groups are target tokens. A target
    reached from several source blocks competes across those blocks.
    """
    row = _segment_softmax(scores, src_index, num_src_tokens)
    col = _segment_softmax(scores, tgt_index, num_tgt_tokens)
    return row * col


def sparse_match(
    params,
    route_f0,
    route_f1,
    token_f0,
    token_f1,
    eighth_hw,
    block_size=DEFAULT_BLOCK_SIZE,
    routing_size=DEFAULT_ROUTING_SIZE,
    halo=DEFAULT_HALO,
    tau_r=DEFAULT_TAU_R,
    tau_c=DEFAULT_TAU_C,
):
    """Route on interacted features, then sparse-match 1/8 token features.

    eighth_hw is (H8, W8) or a pair of grids when the two images differ.
    Token features must already live on the 1/8 grid. Route features are
    aligned by A.
    """
    if (
        isinstance(eighth_hw, (list, tuple))
        and len(eighth_hw) == 2
        and isinstance(eighth_hw[0], (list, tuple))
    ):
        eighth0, eighth1 = eighth_hw
    else:
        eighth0 = eighth1 = eighth_hw
    d0, _ = routing_descriptors(
        params["phi_w"], params["phi_b"], route_f0, block_size, eighth0
    )
    d1, _ = routing_descriptors(
        params["phi_w"], params["phi_b"], route_f1, block_size, eighth1
    )
    routing_scores = routing_affinity(d0, d1, tau_r)
    routes = top_r_indices(routing_scores, routing_size)
    support = build_support(
        routes,
        block_size,
        token_f0.shape[:2],
        token_f1.shape[:2],
        halo,
    )
    scores = sparse_similarity(
        token_f0,
        token_f1,
        support,
        routing_scores,
        prior_weight(params["omega_hat"]),
        tau_c,
    )
    probs = sparse_dual_softmax(
        support["src_index"],
        support["tgt_index"],
        scores,
        support["num_src_tokens"],
        support["num_tgt_tokens"],
    )
    return {
        "probs": probs,
        "scores": scores,
        "support": support,
        "routing_scores": routing_scores,
        "routes": routes,
    }


def multi_positive_softmax_loss(scores, positive_mask):
    """Eq 8. Mean over source blocks that have at least one positive.

    L = -mean_u log( sum_{v in P_u} exp(S_uv) / sum_v exp(S_uv) )
    The denominator runs over every target block, not only positives.
    """
    scores = jnp.asarray(scores, dtype=jnp.float32)
    positive_mask = jnp.asarray(positive_mask, dtype=bool)
    valid = jnp.any(positive_mask, axis=-1)
    lse_all = jax.nn.logsumexp(scores, axis=-1)
    masked = jnp.where(positive_mask, scores, -jnp.inf)
    lse_pos = jax.nn.logsumexp(masked, axis=-1)
    nll = -(lse_pos - lse_all)
    nll = jnp.where(valid, nll, 0.0)
    denom = jnp.sum(valid).astype(jnp.float32)
    return jnp.where(denom > 0, jnp.sum(nll) / jnp.maximum(denom, 1.0), 0.0)


def halo_reachability(n_block_h, n_block_w, block_size, halo):
    """reach[v, p] is true when H_h(T_B(v)) intersects T_B(p)."""
    height = n_block_h * block_size
    width = n_block_w * block_size
    count = n_block_h * n_block_w
    reach = np.zeros((count, count), dtype=bool)
    for v in range(count):
        r0, c0 = _block_origin(v, n_block_w, block_size)
        rs, re, cs, ce = _halo_bounds(r0, c0, block_size, halo, height, width)
        for p in range(count):
            pr, pc = _block_origin(p, n_block_w, block_size)
            if pr < re and pr + block_size > rs and pc < ce and pc + block_size > cs:
                reach[v, p] = True
    return reach


def enlarge_positive_mask(positive_mask, n_block_h, n_block_w, block_size, halo):
    """Halo-aware positive set P_hat_u.

    A target block v is positive for coverage if its halo-expanded tokens
    intersect some ground-truth positive block of u. Selecting v then keeps
    at least one token of that positive block inside C_u. The geometry is
    the target block grid; positive_mask is (U, V).
    """
    pos = np.asarray(positive_mask, dtype=bool)
    reach = halo_reachability(n_block_h, n_block_w, block_size, halo)
    if pos.shape[-1] != reach.shape[0]:
        raise ValueError("positive_mask width must equal the number of target blocks")
    enlarged = (pos.astype(np.int32) @ reach.T.astype(np.int32)) > 0
    return jnp.asarray(enlarged)


def coverage_loss(scores, positive_mask, n_block_h, n_block_w, block_size, halo):
    """Eq 8 on the halo-enlarged positive set."""
    enlarged = enlarge_positive_mask(
        positive_mask, n_block_h, n_block_w, block_size, halo
    )
    return multi_positive_softmax_loss(scores, enlarged)


def covering_blocks_for_tokens(
    token_indices, width, height, block_size, halo, n_block_w, n_block_h
):
    """Blocks v with j in H_h(T_B(v)). These are the blocks covering token j."""
    groups = []
    for token in token_indices:
        row = int(token) // int(width)
        col = int(token) % int(width)
        cover = []
        for v in range(n_block_h * n_block_w):
            r0, c0 = _block_origin(v, n_block_w, block_size)
            rs, re, cs, ce = _halo_bounds(r0, c0, block_size, halo, height, width)
            if rs <= row < re and cs <= col < ce:
                cover.append(v)
        groups.append(cover)
    return groups


def ranking_loss(scores, src_blocks, covering_blocks, covering_mask, routing_size, mu):
    """Eq 9. L_rank = mean softplus(beta_ij - s_ij + mu).

    s_ij is the best routing score among blocks covering the GT target.
    beta_ij is the R-th largest routing score of the source block (1-based).
    If fewer than R target blocks exist, the cutoff is the smallest score.
    """
    scores = jnp.asarray(scores, dtype=jnp.float32)
    src_blocks = jnp.asarray(src_blocks, dtype=jnp.int32)
    covering_blocks = jnp.asarray(covering_blocks, dtype=jnp.int32)
    covering_mask = jnp.asarray(covering_mask, dtype=bool)
    n_tgt = int(scores.shape[-1])
    kth = min(int(routing_size), n_tgt)
    if kth <= 0:
        raise ValueError("R must be positive")
    sorted_scores = jnp.sort(scores, axis=-1)
    beta_per_block = sorted_scores[:, n_tgt - kth]
    beta = beta_per_block[src_blocks]
    gathered = scores[src_blocks[:, None], covering_blocks]
    gathered = jnp.where(covering_mask, gathered, -jnp.inf)
    best = jnp.max(gathered, axis=-1)
    return jnp.mean(jax.nn.softplus(beta - best + mu))


def route_loss(
    scores,
    positive_mask,
    src_blocks,
    covering_blocks,
    covering_mask,
    n_tgt_block_h,
    n_tgt_block_w,
    block_size,
    halo,
    routing_size=DEFAULT_ROUTING_SIZE,
    mu=DEFAULT_MU,
    lambda_rank=DEFAULT_LAMBDA_RANK,
):
    """L_route = L_r + L_cov + lambda_rank * L_rank."""
    l_r = multi_positive_softmax_loss(scores, positive_mask)
    l_cov = coverage_loss(
        scores, positive_mask, n_tgt_block_h, n_tgt_block_w, block_size, halo
    )
    l_rank = ranking_loss(
        scores, src_blocks, covering_blocks, covering_mask, routing_size, mu
    )
    total = l_r + l_cov + lambda_rank * l_rank
    return {"loss": total, "l_r": l_r, "l_cov": l_cov, "l_rank": l_rank}
