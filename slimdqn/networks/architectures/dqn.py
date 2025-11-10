from typing import Sequence

import flax.linen as nn
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


class DQNNet(nn.Module):
    features: Sequence[int]
    architecture_type: str
    n_actions: int
    n_heads: int = None
    n_h_heads: int = None
    n_bins: int = None

    @nn.compact
    def __call__(self, x):
        if self.architecture_type == "cnn":
            initializer = nn.initializers.xavier_uniform()
            idx_feature_start = 3
            x = nn.relu(
                nn.Conv(features=self.features[0], kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer)(
                    jnp.array(x, ndmin=4) / 255.0
                )
            )
            x = nn.relu(
                nn.Conv(features=self.features[1], kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer)(x)
            )
            """  x = nn.relu(
                nn.Conv(features=self.features[2], kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer)(x)
            ) """

            x = x.reshape((x.shape[0], -1))
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

        for idx_layer in range(idx_feature_start, len(self.features)):
            x = nn.relu((nn.Dense(self.features[idx_layer], kernel_init=initializer)(x)))

        if self.n_bins is not None:
            if self.n_heads is None and self.n_h_heads is None:
                return nn.Dense(self.n_actions * self.n_bins, name="Dense_final", kernel_init=initializer)(x).reshape(
                    self.n_actions, self.n_bins
                )
            elif self.n_h_heads is None:
                return nn.Dense(
                    self.n_heads * self.n_actions * self.n_bins, name="Dense_final", kernel_init=initializer
                )(x).reshape((self.n_heads, self.n_actions, self.n_bins))
            else:
                q_vals = nn.Dense(
                    self.n_heads * self.n_actions * self.n_bins, name="Dense_final", kernel_init=initializer
                )(x)
                h_vals = nn.Dense(
                    self.n_h_heads * self.n_actions * self.n_bins, name="Dense_final_h", kernel_init=initializer
                )(x)

                return q_vals.reshape((self.n_heads, self.n_actions, self.n_bins)), h_vals.reshape(
                    (self.n_h_heads, self.n_actions, self.n_bins)
                )

        if self.n_heads is None and self.n_h_heads is None:
            return nn.Dense(self.n_actions, name="Dense_final", kernel_init=initializer)(x)
        elif self.n_h_heads is None:
            return nn.Dense(self.n_heads * self.n_actions, name="Dense_final", kernel_init=initializer)(x).reshape(
                (self.n_heads, self.n_actions)
            )
        else:
            q_vals = nn.Dense(self.n_heads * self.n_actions, name="Dense_final", kernel_init=initializer)(x)
            h_vals = nn.Dense(self.n_h_heads * self.n_actions, name="Dense_final_h", kernel_init=initializer)(x)

            return q_vals.reshape((self.n_heads, self.n_actions)), h_vals.reshape((self.n_h_heads, self.n_actions))
