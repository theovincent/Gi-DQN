from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class DQNNet(nn.Module):
    features: Sequence[int]
    n_actions: int
    n_heads: int
    n_h_heads: int

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        x = nn.Conv(features=self.features[0], kernel_size=(2, 2), strides=(2, 2), kernel_init=initializer)(
            jnp.array(x, ndmin=4) / 255.0
        )
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))

        x = jnp.squeeze(x)
        for idx_layer in range(1, len(self.features)):
            x = nn.Dense(self.features[idx_layer], kernel_init=initializer)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)

        if self.n_heads == 1:
            q_vals = nn.Dense(self.n_actions, kernel_init=initializer, name="q_heads")(x)
        else:
            q_vals = nn.Dense(self.n_heads * self.n_actions, kernel_init=initializer, name="q_heads")(x).reshape(
                (self.n_heads, self.n_actions)
            )

        if self.n_h_heads == 0:
            return q_vals
        elif self.n_h_heads == 1 and self.n_heads == 1: # QRC case
            h_vals = nn.Dense(self.n_actions, kernel_init=initializer, name="h_heads")(jax.lax.stop_gradient(x))
            return q_vals, h_vals
        else:
            h_vals = nn.Dense(self.n_h_heads * self.n_actions, kernel_init=initializer, name="h_heads")(
                jax.lax.stop_gradient(x)
            ).reshape((self.n_h_heads, self.n_actions))
            return q_vals, h_vals #GiDQN case
