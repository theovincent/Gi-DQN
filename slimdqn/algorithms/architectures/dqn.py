from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import time
import numpy as np


class Stack(nn.Module):
    """Stack of pooling and convolutional blocks with residual connections."""

    stack_size: int

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        x = nn.Conv(
            features=self.stack_size,
            kernel_size=(3, 3),
            kernel_init=initializer,
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), padding="SAME", strides=(2, 2))

        for _ in range(2):
            block_input = x
            x = nn.relu(x)
            x = nn.relu(nn.Conv(features=self.stack_size, kernel_size=(3, 3))(x))
            x = nn.Conv(features=self.stack_size, kernel_size=(3, 3))(x)
            x += block_input

        return x


class Head(nn.Module):
    features: int
    n_actions: int
    initializer: nn.initializers.Initializer
    layer_norm: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features, kernel_init=self.initializer)(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.relu(x)

        return nn.Dense(self.n_actions, kernel_init=self.initializer)(x)


def make_heads(n_heads):
    return nn.vmap(
        Head, variable_axes={"params": 0}, split_rngs={"params": True}, in_axes=None, out_axes=0, axis_size=n_heads
    )


class DQNNet(nn.Module):
    features: Sequence[int]
    architecture_type: str
    layer_norm: bool
    n_actions: int
    n_heads: int = None
    n_h_heads: int = None

    @nn.compact
    def __call__(self, x):
        if self.architecture_type == "cnn":
            initializer = nn.initializers.xavier_uniform()
            idx_feature_start = 3
            x = nn.Conv(features=self.features[0], kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer)(
                jnp.array(x, ndmin=4) / 255.0
            )
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)

            x = nn.Conv(features=self.features[1], kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Conv(features=self.features[2], kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))
            # x = jnp.mean(x, axis=(1, 2))
        elif self.architecture_type == "impala":
            initializer = nn.initializers.xavier_uniform()
            idx_feature_start = 3
            x = Stack(self.features[0])(jnp.array(x, ndmin=4) / 255.0)
            x = Stack(self.features[1])(x)
            x = nn.relu(Stack(self.features[2])(x))
            x = x.reshape((x.shape[0], -1))
        elif self.architecture_type == "fc":
            initializer = nn.initializers.lecun_normal()
            idx_feature_start = 0
        x = jnp.squeeze(x)

        for idx_layer in range(idx_feature_start, len(self.features) - 1):
            x = nn.Dense(self.features[idx_layer], kernel_init=initializer)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)

        hidden_dim = self.features[-1]
        if self.n_heads is None and self.n_h_heads is None:
            return Head(hidden_dim, self.n_actions, initializer, self.layer_norm, name="q_heads")(x)

        latent = x
        q_vals = make_heads(self.n_heads)(
            features=hidden_dim,
            n_actions=self.n_actions,
            layer_norm=self.layer_norm,
            initializer=initializer,
            name="q_heads",
        )(latent)

        if self.n_h_heads is None:
            return q_vals

        h_vals = make_heads(self.n_h_heads)(
            features=hidden_dim,
            n_actions=self.n_actions,
            layer_norm=self.layer_norm,
            initializer=initializer,
            name="h_heads",
        )(jax.lax.stop_gradient(latent))

        return q_vals, h_vals
