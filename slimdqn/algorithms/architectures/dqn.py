from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class Stack(nn.Module):
    """Stack of pooling and convolutional blocks with residual connections."""

    stack_size: int

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        x = nn.Conv(features=self.stack_size, kernel_size=(3, 3), kernel_init=initializer)(x)
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
    n_fc: int

    @nn.compact
    def __call__(self, x):
        if self.n_fc == 1:
            return nn.Dense(self.n_actions, kernel_init=self.initializer)(x)
        elif self.n_fc == 2:
            if self.features is not None:
                x = nn.Dense(self.features, kernel_init=self.initializer)(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = nn.relu(x)

            return nn.Dense(self.n_actions, kernel_init=self.initializer)(x)


def make_heads(n_heads):
    # the axis for the Bellman iterations in params should be 0 -> variable_axes={"params": 0}
    # the entire input should be vmapped -> in_axes=None
    # the Bellman iterations are before the action axis -> out_axes=0
    return nn.vmap(
        Head, variable_axes={"params": 0}, split_rngs={"params": True}, in_axes=None, out_axes=0, axis_size=n_heads
    )


class DQNNet(nn.Module):
    features: Sequence[int]
    architecture_type: str
    layer_norm: tuple[bool, bool]
    gap: bool
    linear_heads: bool
    n_actions: int
    n_heads: int
    n_h_heads: int
    low_scale: bool
    n_conv: int
    n_fc: int

    @nn.compact
    def __call__(self, x):
        if self.architecture_type == "cnn":
            initializer = nn.initializers.xavier_uniform()
            idx_feature_start = self.n_conv
            first_kernel, first_stride = (
                ((4, 4), (2, 2)) if self.low_scale else [] # 84 x 84 pixel: ((8, 8), (4, 4))
                )  # for 42 x 42 pxl: ((4, 4), (2, 2)); for 10 x 10 pixels: ((3, 3), (1, 1)); for 24 x 24 pixels: ((5,5), (1,1))
            x = nn.Conv(
                features=self.features[0], kernel_size=first_kernel, strides=first_stride, kernel_init=initializer
            )(jnp.array(x, ndmin=4) / 255.0)
            if self.layer_norm[0]:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
            if self.n_conv > 1:
                x = nn.Conv(features=self.features[1], kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer)(x)
                if self.layer_norm[0]:
                    x = nn.LayerNorm()(x)
                x = nn.relu(x)

            if self.n_conv > 2:
                x = nn.Conv(features=self.features[2], kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer)(x)
                if self.layer_norm[0]:
                    x = nn.LayerNorm()(x)
                x = nn.relu(x)
            if self.gap:
                x = jnp.mean(x, axis=(1, 2))
            else:
                x = x.reshape((x.shape[0], -1))
        elif self.architecture_type == "impala":
            raise NotImplementedError
            # initializer = nn.initializers.xavier_uniform()
            # idx_feature_start = 3
            # x = Stack(self.features[0])(jnp.array(x, ndmin=4) / 255.0)
            # x = Stack(self.features[1])(x)
            # x = nn.relu(Stack(self.features[2])(x))
            # if self.gap:
            #     x = jnp.mean(x, axis=(1, 2))
            # else:
            #     x = x.reshape((x.shape[0], -1))
        elif self.architecture_type == "fc":
            raise NotImplementedError
            # initializer = nn.initializers.lecun_normal()
            # idx_feature_start = 0

        x = jnp.squeeze(x)
        for idx_layer in range(idx_feature_start, len(self.features) - 1 + int(self.linear_heads)):
            x = nn.Dense(self.features[idx_layer], kernel_init=initializer)(x)
            if self.layer_norm[1]:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)

        if self.n_heads == 1:
            q_vals = Head(
                None if self.linear_heads else self.features[-1],
                self.n_actions,
                initializer,
                self.layer_norm[1],
                self.n_fc,
                name="q_heads",
            )(x)
        else:
            if not self.linear_heads:
                q_vals = make_heads(self.n_heads)(
                    self.features[-1], self.n_actions, initializer, self.layer_norm[1], self.n_fc, name="q_heads"
                )(x)
            else:
                q_vals = Head(None, self.n_heads * self.n_actions, initializer, False, self.n_fc, name="q_heads")(
                    x
                ).reshape((self.n_heads, self.n_actions))

        if self.n_h_heads == 0:
            return q_vals
        elif self.n_h_heads == 1:
            h_vals = Head(
                None if self.linear_heads else self.features[-1],
                self.n_actions,
                initializer,
                self.layer_norm[1],
                self.n_fc,
                name="h_heads",
            )(jax.lax.stop_gradient(x))
            return q_vals, h_vals
        else:
            if not self.linear_heads:
                h_vals = make_heads(self.n_h_heads)(
                    None if self.linear_heads else self.features[-1],
                    self.n_actions,
                    initializer,
                    self.layer_norm[1],
                    self.n_fc,
                    name="h_heads",
                )(jax.lax.stop_gradient(x))
            else:
                h_vals = Head(
                    None, self.n_h_heads * self.n_actions, initializer, False, n_fc=self.n_fc, name="h_heads"
                )(jax.lax.stop_gradient(x)).reshape((self.n_h_heads, self.n_actions))
            return q_vals, h_vals
