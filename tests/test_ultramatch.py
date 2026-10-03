"""Faithfulness checks for the UltraMatch routing core."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from ultramatch.fine import (
    apply_mlp,
    bin_coordinates,
    encode_side,
    fine_match,
    init_fine_head,
    pair_embed,
    refine_direction,
    soft_argmax,
    soft_argmax_probs,
)
from ultramatch.reparam import (
    apply_fused_conv,
    apply_fused_stack,
    apply_rep_block,
    apply_rep_stack,
    batch_stats,
    fuse_rep_block,
    fuse_rep_stack,
    init_rep_block,
    init_rep_stack,
    sum_branches,
)
from ultramatch.router import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_HALO,
    DEFAULT_LAMBDA_RANK,
    DEFAULT_MU,
    DEFAULT_ROUTING_SIZE,
    DEFAULT_TAU_C,
    DEFAULT_TAU_R,
    align_to_routing_grid,
    build_support,
    candidate_tokens,
    coverage_loss,
    covering_blocks_for_tokens,
    enlarge_positive_mask,
    init_router,
    multi_positive_softmax_loss,
    prior_weight,
    ranking_loss,
    route_loss,
    routing_affinity,
    routing_descriptors,
    sparse_dual_softmax,
    sparse_match,
    sparse_similarity,
    top_r_indices,
)
from ultramatch.train import train_router_synthetic


def test_paper_defaults():
    assert DEFAULT_BLOCK_SIZE == 4
    assert DEFAULT_ROUTING_SIZE == 6
    assert DEFAULT_HALO == 1
    assert DEFAULT_TAU_R == 0.1
    assert DEFAULT_TAU_C == 0.1
    assert DEFAULT_MU == 0.5
    assert DEFAULT_LAMBDA_RANK == 0.5


def test_fused_inference_matches_training_time_output():
    params = init_rep_block(jax.random.PRNGKey(0), 4, 5)
    params = dict(params)
    params["bn_scale"] = jnp.array([0.7, 1.3, 0.2, 1.8, 0.9], dtype=jnp.float32)
    params["bn_bias"] = jnp.array([0.1, -0.4, 0.3, -0.2, 0.5], dtype=jnp.float32)
    params["bn_mean"] = jnp.array([0.2, -0.1, 0.5, 0.0, -0.3], dtype=jnp.float32)
    params["bn_var"] = jnp.array([0.4, 1.2, 0.8, 2.0, 0.3], dtype=jnp.float32)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 5, 4))

    # Eval-mode training architecture (shared BN with stored stats) vs fused conv.
    y_arch = apply_rep_block(params, x, training=False)
    kernel, bias = fuse_rep_block(params)
    assert kernel.shape == (3, 3, 4, 5)
    y_fused = apply_fused_conv(kernel, bias, x)
    assert jnp.allclose(y_arch, y_fused, atol=1e-4, rtol=1e-4)

    # Training-mode forward uses batch statistics. Folding those same stats
    # into the 3x3 reproduces that output.
    y_train = apply_rep_block(params, x, training=True)
    mean, var = batch_stats(sum_branches(params, x))
    kernel_t, bias_t = fuse_rep_block(params, mean=mean, var=var)
    y_fused_t = apply_fused_conv(kernel_t, bias_t, x)
    assert jnp.allclose(y_train, y_fused_t, atol=1e-4, rtol=1e-4)
    assert float(jnp.max(jnp.abs(y_train - y_fused_t))) < 1e-4


def test_stack_fusion_matches():
    stack = init_rep_stack(jax.random.PRNGKey(2), [(3, 4), (4, 4)])
    blocks = []
    for i, block in enumerate(stack["blocks"]):
        block = dict(block)
        cout = block["bn_scale"].shape[0]
        block["bn_scale"] = jnp.arange(1, cout + 1, dtype=jnp.float32) * (0.4 + 0.2 * i)
        block["bn_bias"] = jnp.linspace(-0.3, 0.5, cout, dtype=jnp.float32)
        block["bn_mean"] = jnp.linspace(-0.2, 0.2, cout, dtype=jnp.float32)
        block["bn_var"] = jnp.linspace(0.3, 1.7, cout, dtype=jnp.float32)
        blocks.append(block)
    stack = {"blocks": blocks}
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 6, 8, 3))

    y_arch = apply_rep_stack(stack, x, training=False)
    y_fused = apply_fused_stack(fuse_rep_stack(stack), x)
    assert jnp.allclose(y_arch, y_fused, atol=1e-4, rtol=1e-4)

    # Sequential training-mode forwards, each fused with its own batch stats.
    y = x
    y_hat = x
    for block in stack["blocks"]:
        mean, var = batch_stats(sum_branches(block, y))
        y = apply_rep_block(block, y, training=True)
        kernel, bias = fuse_rep_block(block, mean=mean, var=var)
        y_hat = apply_fused_conv(kernel, bias, y_hat)
    assert jnp.allclose(y, y_hat, atol=1e-4, rtol=1e-4)


def test_routing_descriptors_affinity_and_topr():
    features = jnp.arange(8 * 8 * 2, dtype=jnp.float32).reshape(8, 8, 2)
    aligned = align_to_routing_grid(features, block_size=4, eighth_hw=(8, 8))
    assert aligned.shape == (2, 2, 2)
    assert jnp.allclose(aligned[0, 0], features[:4, :4].mean(axis=(0, 1)))
    assert jnp.allclose(aligned[1, 1], features[4:, 4:].mean(axis=(0, 1)))

    already = jnp.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]
    )
    ident = align_to_routing_grid(already, block_size=4, eighth_hw=(8, 8))
    assert jnp.allclose(ident, already)

    phi_w = jnp.eye(3)
    phi_b = jnp.zeros((3,))
    desc, grid = routing_descriptors(phi_w, phi_b, already, 4, (8, 8))
    assert grid == (2, 2)
    assert desc.shape == (4, 3)
    assert jnp.allclose(jnp.linalg.norm(desc, axis=-1), jnp.ones((4,)), atol=1e-5)
    # Last cell was (1, 1, 1); L2 norm is sqrt(3), so each coord is 1/sqrt(3).
    assert jnp.allclose(desc[3], jnp.ones((3,)) / jnp.sqrt(3.0), atol=1e-5)

    d0 = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    d1 = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    scores = routing_affinity(d0, d1, tau_r=0.1)
    assert jnp.allclose(scores[0], jnp.array([10.0, 0.0, 10.0]))
    assert jnp.allclose(scores[1], jnp.array([0.0, 10.0, 0.0]))
    idx = top_r_indices(scores, 2)
    assert idx.shape == (2, 2)
    # Stable argsort: the two 10s on row 0 keep index order 0 then 2.
    assert list(np.asarray(idx[0])) == [0, 2]
    assert list(np.asarray(idx[1])) == [1, 0]


def test_halo_expansion_clips_and_merges():
    # 8 x 8 tokens, B = 4, four blocks:
    # 0 1
    # 2 3
    block0 = candidate_tokens([0], n_block_w=2, block_size=4, halo=1, height=8, width=8)
    # Block 0 occupies [0, 4) x [0, 4). Halo clips at the origin and reaches 5.
    assert block0[0] == 0
    assert len(block0) == 25
    assert (4 * 8 + 4) in block0
    assert (5 * 8 + 5) not in block0
    assert len(block0) == len(set(block0))

    merged = candidate_tokens([0, 1], n_block_w=2, block_size=4, halo=1, height=8, width=8)
    # Rows [0, 5), cols [0, 8). Overlap around col 3..4 is stored once.
    assert len(merged) == 5 * 8
    assert len(merged) == len(set(merged))
    assert len(merged) < 25 + 5 * 5


def test_sparse_similarity_uses_containing_block():
    # Halo token (0, 4) belongs to block 1, even if the route was block 0.
    f0 = jnp.ones((8, 8, 2), dtype=jnp.float32)
    f1 = jnp.ones((8, 8, 2), dtype=jnp.float32)
    routing_scores = jnp.zeros((4, 4), dtype=jnp.float32)
    routing_scores = routing_scores.at[0, 0].set(0.5)
    routing_scores = routing_scores.at[0, 1].set(1.5)
    support = {
        "src_index": jnp.array([0], dtype=jnp.int32),
        "tgt_index": jnp.array([4], dtype=jnp.int32),
        "src_block": jnp.array([0], dtype=jnp.int32),
        "tgt_block": jnp.array([1], dtype=jnp.int32),
    }
    omega = 0.25
    tau_c = 0.1
    score = sparse_similarity(f0, f1, support, routing_scores, omega, tau_c)
    # dot = 2, C = 2, so 2 / (2 * 0.1) + 0.25 * S^r[0, b(j)=1]
    assert jnp.allclose(score, jnp.array([10.0 + 0.25 * 1.5]))
    assert float(prior_weight(jnp.array(0.0))) == 0.0

    routes = [[0], [0], [0], [0]]
    built = build_support(routes, block_size=4, src_hw=(8, 8), tgt_hw=(8, 8), halo=1)
    # Token (0, 4) = 4 is inside the halo of target block 0 and owned by block 1.
    halo_edges = np.asarray(built["tgt_index"]) == 4
    assert halo_edges.any()
    assert set(np.asarray(built["tgt_block"])[halo_edges].tolist()) == {1}
    pairs = list(zip(np.asarray(built["src_index"]).tolist(), np.asarray(built["tgt_index"]).tolist()))
    assert len(pairs) == len(set(pairs))


def _dense_dual_softmax(num_src, num_tgt, src, tgt, scores):
    dense = jnp.full((num_src, num_tgt), -jnp.inf)
    dense = dense.at[src, tgt].set(scores)
    probs = jax.nn.softmax(dense, axis=1) * jax.nn.softmax(dense, axis=0)
    return probs[src, tgt]


def _independent_block_dual_softmax(support, scores):
    """Sec 4.5 ablation: Dual-Softmax inside each source block, not across blocks."""
    src = np.asarray(support["src_index"])
    tgt = np.asarray(support["tgt_index"])
    blocks = np.asarray(support["src_block"])
    scores_np = np.asarray(scores)
    out = np.zeros_like(scores_np)
    for u in np.unique(blocks):
        mask = blocks == u
        local = sparse_dual_softmax(
            jnp.asarray(src[mask]),
            jnp.asarray(tgt[mask]),
            jnp.asarray(scores_np[mask]),
            support["num_src_tokens"],
            support["num_tgt_tokens"],
        )
        out[mask] = np.asarray(local)
    return out


def test_sparse_dual_softmax_matches_dense_not_per_block():
    # Two source tokens, two target tokens, B = 1, h = 0.
    # Source 0 routes to both targets. Source 1 routes only to target 0.
    # Column competition on target 0 is the global-vs-per-block difference.
    routes = [[0, 1], [0]]
    support = build_support(routes, block_size=1, src_hw=(1, 2), tgt_hw=(1, 2), halo=0)
    scores = jnp.zeros((support["src_index"].shape[0],), dtype=jnp.float32)
    got = sparse_dual_softmax(
        support["src_index"],
        support["tgt_index"],
        scores,
        support["num_src_tokens"],
        support["num_tgt_tokens"],
    )
    ref = _dense_dual_softmax(
        support["num_src_tokens"],
        support["num_tgt_tokens"],
        support["src_index"],
        support["tgt_index"],
        scores,
    )
    assert jnp.allclose(got, ref, atol=1e-6, rtol=1e-6)
    per_block = _independent_block_dual_softmax(support, scores)
    assert not np.allclose(np.asarray(got), per_block, atol=1e-3)

    # Larger grid with halo overlap, so a missed merge would disagree with dense.
    routes_h = np.array([[0, 1], [0, 1], [1, 2], [0, 3]], dtype=np.int32)
    support_h = build_support(routes_h, block_size=2, src_hw=(4, 4), tgt_hw=(4, 4), halo=1)
    edges = list(
        zip(
            np.asarray(support_h["src_index"]).tolist(),
            np.asarray(support_h["tgt_index"]).tolist(),
        )
    )
    assert len(edges) == len(set(edges))
    scores_h = jnp.arange(len(edges), dtype=jnp.float32) * 0.17 - 0.4
    got_h = sparse_dual_softmax(
        support_h["src_index"],
        support_h["tgt_index"],
        scores_h,
        support_h["num_src_tokens"],
        support_h["num_tgt_tokens"],
    )
    ref_h = _dense_dual_softmax(
        support_h["num_src_tokens"],
        support_h["num_tgt_tokens"],
        support_h["src_index"],
        support_h["tgt_index"],
        scores_h,
    )
    assert jnp.allclose(got_h, ref_h, atol=1e-5, rtol=1e-5)
    per_h = _independent_block_dual_softmax(support_h, scores_h)
    assert not np.allclose(np.asarray(got_h), per_h, atol=1e-3)


def _row_nll(row, positive_ids):
    exps = [math.exp(v) for v in row]
    pos = sum(exps[i] for i in positive_ids)
    return -math.log(pos / sum(exps))


def _softplus(x):
    return math.log(1.0 + math.exp(x))


def test_routing_losses_hand_computed():
    # 2 source blocks, 3 target blocks.
    row0 = [1.0, 0.2, 3.0]
    row1 = [0.0, 4.0, 1.0]
    scores = jnp.array([row0, row1], dtype=jnp.float32)
    # Source 0 matches target block 1. Source 1 matches target block 0.
    positive = jnp.array(
        [[False, True, False], [True, False, False]],
        dtype=bool,
    )
    l_r_expected = 0.5 * (_row_nll(row0, [1]) + _row_nll(row1, [0]))
    l_r = multi_positive_softmax_loss(scores, positive)
    assert abs(float(l_r) - l_r_expected) < 1e-5

    # A source block with no valid match is not in U, so it does not dilute L_r.
    scores_extra = jnp.concatenate([scores, jnp.array([[5.0, 5.0, 5.0]])], axis=0)
    positive_extra = jnp.array(
        [[False, True, False], [True, False, False], [False, False, False]],
        dtype=bool,
    )
    l_r_extra = multi_positive_softmax_loss(scores_extra, positive_extra)
    assert abs(float(l_r_extra) - l_r_expected) < 1e-5

    # Target grid: 1 x 3 blocks, B = 2, h = 1. Width 6.
    # Halo of block 1 reaches blocks 0 and 2. Halo of block 0 does not reach block 2.
    enlarged = enlarge_positive_mask(positive, n_block_h=1, n_block_w=3, block_size=2, halo=1)
    assert enlarged.tolist() == [[True, True, True], [True, True, False]]
    l_cov_expected = 0.5 * (_row_nll(row0, [0, 1, 2]) + _row_nll(row1, [0, 1]))
    assert abs(float(l_r) - 0.0) > 1e-3  # not the all-positive case
    assert abs(l_cov_expected - 0.5 * _row_nll(row1, [0, 1])) < 1e-12
    l_cov = coverage_loss(scores, positive, 1, 3, 2, 1)
    assert abs(float(l_cov) - l_cov_expected) < 1e-5

    # Ranking, one covering block each. R = 2, mu = 0.5.
    # Row 0 sorted desc: 3.0, 1.0, 0.2. beta = 1.0. s = 0.2. term = softplus(1.3).
    # Row 1 sorted desc: 4.0, 1.0, 0.0. beta = 1.0. s = 0.0. term = softplus(1.5).
    l_rank_expected = 0.5 * (_softplus(1.3) + _softplus(1.5))
    l_rank = ranking_loss(
        scores,
        src_blocks=jnp.array([0, 1], dtype=jnp.int32),
        covering_blocks=jnp.array([[1, 0], [0, 0]], dtype=jnp.int32),
        covering_mask=jnp.array([[True, False], [True, False]]),
        routing_size=2,
        mu=0.5,
    )
    assert abs(float(l_rank) - l_rank_expected) < 1e-5

    # R = 1 cutoff is the largest score, not the second.
    l_rank_r1 = ranking_loss(
        scores,
        src_blocks=jnp.array([0], dtype=jnp.int32),
        covering_blocks=jnp.array([[1]], dtype=jnp.int32),
        covering_mask=jnp.array([[True]]),
        routing_size=1,
        mu=0.5,
    )
    assert abs(float(l_rank_r1) - _softplus(3.0 - 0.2 + 0.5)) < 1e-5

    # Best score among blocks covering the GT target, not only b(j).
    # Token col 2 on a width-6 grid is the left edge of block 1, covered by blocks 0 and 1.
    groups = covering_blocks_for_tokens(
        [2], width=6, height=2, block_size=2, halo=1, n_block_w=3, n_block_h=1
    )
    assert groups == [[0, 1]]
    l_rank_cover = ranking_loss(
        scores,
        src_blocks=jnp.array([0], dtype=jnp.int32),
        covering_blocks=jnp.array([groups[0]], dtype=jnp.int32),
        covering_mask=jnp.array([[True, True]]),
        routing_size=2,
        mu=0.5,
    )
    # s = max(1.0, 0.2) = 1.0, beta = 1.0, term = softplus(0.5)
    assert abs(float(l_rank_cover) - _softplus(0.5)) < 1e-5

    lam = 0.5
    total_expected = l_r_expected + l_cov_expected + lam * l_rank_expected
    out = route_loss(
        scores,
        positive,
        src_blocks=jnp.array([0, 1], dtype=jnp.int32),
        covering_blocks=jnp.array([[1, 0], [0, 0]], dtype=jnp.int32),
        covering_mask=jnp.array([[True, False], [True, False]]),
        n_tgt_block_h=1,
        n_tgt_block_w=3,
        block_size=2,
        halo=1,
        routing_size=2,
        mu=0.5,
        lambda_rank=lam,
    )
    assert abs(float(out["l_r"]) - l_r_expected) < 1e-5
    assert abs(float(out["l_cov"]) - l_cov_expected) < 1e-5
    assert abs(float(out["l_rank"]) - l_rank_expected) < 1e-5
    assert abs(float(out["loss"]) - total_expected) < 1e-5


def test_shared_fine_head_and_soft_argmax():
    params = init_fine_head(
        jax.random.PRNGKey(0),
        dim=8,
        ed_hidden=12,
        ep_hidden=16,
        axis_hidden=10,
        n_bins=17,
    )
    assert "ed" in params
    assert "ed_q" not in params and "ed_r" not in params
    assert set(params["ed"]) == {"fc1", "fc2"}

    q = jax.random.normal(jax.random.PRNGKey(1), (5, 8))
    r = jax.random.normal(jax.random.PRNGKey(2), (5, 8))
    manual_ed_q = apply_mlp(params["ed"], q)
    hidden = jax.nn.gelu(q @ params["ed"]["fc1"]["w"] + params["ed"]["fc1"]["b"])
    manual_gelu = hidden @ params["ed"]["fc2"]["w"] + params["ed"]["fc2"]["b"]
    assert jnp.allclose(encode_side(params, q), manual_ed_q)
    assert jnp.allclose(encode_side(params, q), manual_gelu)
    assert jnp.allclose(encode_side(params, r), apply_mlp(params["ed"], r))

    q_prime, r_prime, z = pair_embed(params, q, r)
    assert jnp.allclose(q_prime, q + encode_side(params, q))
    assert jnp.allclose(r_prime, r + encode_side(params, r))
    pair = jnp.concatenate([q_prime, r_prime], axis=-1)
    manual_z_in = pair + apply_mlp(params["ep"], pair)
    mean = jnp.mean(manual_z_in, axis=-1, keepdims=True)
    var = jnp.var(manual_z_in, axis=-1, keepdims=True)
    manual_z = (manual_z_in - mean) * jax.lax.rsqrt(var + 1e-5)
    manual_z = manual_z * params["ln"]["gamma"] + params["ln"]["beta"]
    assert jnp.allclose(z, manual_z, atol=1e-5)

    both = fine_match(params, q, r)
    swapped = fine_match(params, r, q)
    assert jnp.allclose(both["backward"]["delta"], swapped["forward"]["delta"])
    assert jnp.allclose(both["forward"]["delta"], swapped["backward"]["delta"])
    assert jnp.all(both["forward"]["sigma"] > 0.0)
    assert jnp.all(both["forward"]["sigma"] < 1.0)

    coords = bin_coordinates(17)
    assert jnp.allclose(coords[0], -0.5)
    assert jnp.allclose(coords[-1], 0.5)
    for k in range(17):
        probs = jnp.zeros((17,)).at[k].set(1.0)
        got = soft_argmax_probs(probs, coords)
        assert jnp.allclose(got, coords[k])
        logits = jnp.full((17,), -1e9).at[k].set(0.0)
        assert jnp.allclose(soft_argmax(logits, coords), coords[k], atol=1e-5)

    # Force the axis heads to a one-hot bin and check the head, not only the helper.
    peaked = dict(params)
    peaked["head_x"] = {
        "fc1": params["head_x"]["fc1"],
        "fc2": {
            "w": jnp.zeros_like(params["head_x"]["fc2"]["w"]),
            "b": jnp.zeros((18,)).at[4].set(40.0),
        },
    }
    peaked["head_y"] = {
        "fc1": params["head_y"]["fc1"],
        "fc2": {
            "w": jnp.zeros_like(params["head_y"]["fc2"]["w"]),
            "b": jnp.zeros((18,)).at[11].set(40.0),
        },
    }
    forced = refine_direction(peaked, q, r)
    assert jnp.allclose(forced["delta"][..., 0], jnp.full((5,), coords[4]), atol=1e-5)
    assert jnp.allclose(forced["delta"][..., 1], jnp.full((5,), coords[11]), atol=1e-5)
    assert jnp.allclose(forced["sigma"], jnp.full((5, 2), 0.5), atol=1e-5)


def test_sparse_match_wires_shared_phi_and_stays_sparse():
    params = init_router(jax.random.PRNGKey(4), in_channels=4, desc_dim=8)
    assert float(params["omega_hat"]) == 0.0
    route = jax.random.normal(jax.random.PRNGKey(5), (2, 2, 4))
    tokens = jax.random.normal(jax.random.PRNGKey(6), (8, 8, 4))
    out = sparse_match(
        params,
        route,
        route,
        tokens,
        tokens,
        eighth_hw=(8, 8),
        block_size=4,
        routing_size=2,
        halo=1,
    )
    n = 8 * 8
    assert out["probs"].shape[0] < n * n
    assert out["support"]["src_index"].shape == out["probs"].shape
    d0, _ = routing_descriptors(params["phi_w"], params["phi_b"], route, 4, (8, 8))
    scores = routing_affinity(d0, d0, DEFAULT_TAU_R)
    assert jnp.allclose(out["routing_scores"], scores)


def test_synthetic_router_training_decreases_loss():
    losses = train_router_synthetic(steps=30, seed=0, lr=0.4)
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0]
