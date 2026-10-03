"""A few SGD steps of the router loss on synthetic block correspondences.

Not a MegaDepth run. CPU, well under a minute. Optimizes only phi so that
identity block matches become the multi-positive softmax targets.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ultramatch.router import (
    init_router,
    multi_positive_softmax_loss,
    routing_affinity,
    routing_descriptors,
)


def train_router_synthetic(steps=40, seed=0, lr=0.3, desc_dim=8):
    key = jax.random.PRNGKey(seed)
    k_init, k_noise = jax.random.split(key)
    # 2 x 2 routing grid. eighth_hw (8, 8) and B = 4 make A the identity.
    eye = jnp.eye(4, dtype=jnp.float32).reshape(2, 2, 4)
    noise = 0.05 * jax.random.normal(k_noise, eye.shape)
    f0 = eye
    f1 = eye + noise
    pos = jnp.eye(4, dtype=bool)
    params = init_router(k_init, in_channels=4, desc_dim=desc_dim)

    def loss_fn(phi):
        d0, _ = routing_descriptors(phi["phi_w"], phi["phi_b"], f0, 4, (8, 8))
        d1, _ = routing_descriptors(phi["phi_w"], phi["phi_b"], f1, 4, (8, 8))
        scores = routing_affinity(d0, d1, 0.1)
        return multi_positive_softmax_loss(scores, pos)

    losses = []
    for _ in range(steps):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        losses.append(float(loss))
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    losses.append(float(loss_fn(params)))
    return losses
