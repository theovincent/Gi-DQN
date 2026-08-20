from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.algorithms.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class C51RCShared:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        features: list,
        architecture_type: str,
        layer_norm: bool,
        gap: bool,
        linear_heads: bool,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_period: int,
        weight_decay: float,
        min_value: float,
        max_value: float,
        adam_eps: float = 1e-8,
    ):
        self.n_actions = n_actions
        self.network = DQNNet(
            features, architecture_type, layer_norm, gap, linear_heads, n_actions, n_heads=1, n_h_heads=1, n_bins=51
        )
        # initialize online network
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        # regularize the TD-error estimator network
        self.optimizer = optax.adamw(
            learning_rate,
            eps=adam_eps,
            weight_decay=weight_decay,
            mask=jax.tree_util.tree_map_with_path(lambda path, _: "h_heads" in path[1].key, self.params),
        )
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_period = target_update_period
        self.support = jnp.linspace(min_value, max_value, 51, dtype=jnp.float32)
        self.bin_size = (max_value - min_value) / 50
        self.max_value = max_value
        self.min_value = min_value
        self.clip_target = lambda target: jnp.clip(target, min_value, max_value)
        self.cumulative_q_losses = 0
        self.cumulative_h_losses = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        # Update the network parameters every `update_to_data` steps
        for _ in range(int(max(self.update_to_data, 1))):
            # if update_to_data < 1, only perform one update if step = 0 [1 / self.update_to_data]
            if self.update_to_data < 1 and step % (1 / self.update_to_data) != 0:
                return None
            batch_samples, (sample_keys, importance_weights) = replay_buffer.sample()

            self.params, self.optimizer_state, per_sample_q_losses, per_sample_h_losses = self.learn_on_batch(
                self.params, self.optimizer_state, batch_samples, importance_weights
            )

            replay_buffer.update(sample_keys, per_sample_q_losses + per_sample_h_losses)

            self.cumulative_q_losses += per_sample_q_losses.mean()
            self.cumulative_h_losses += per_sample_h_losses.mean()

    def update_target_params(self, step: int):
        # shift the network parameters every `target_update_period` steps. This starts the next Bellman iteration
        if step % self.target_update_period == 0:

            self.logs = {
                "n_training_steps": step,
                "loss": np.mean(self.cumulative_q_losses) / (self.target_update_period * self.update_to_data),
                "h_loss": np.mean(self.cumulative_h_losses) / (self.target_update_period * self.update_to_data),
            }

            self.cumulative_q_losses = 0
            self.cumulative_h_losses = 0

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples, importance_weights):
        (grad_loss), (per_sample_q_losses, per_sample_h_losses) = jax.grad(self.loss_on_batch, has_aux=True)(
            params, batch_samples, importance_weights
        )

        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)

        params = optax.apply_updates(params, updates)

        return (params, optimizer_state, per_sample_q_losses, per_sample_h_losses)

    def loss_on_batch(self, params: FrozenDict, samples, importance_weights):
        # vmap to compute the loss for all samples
        total_losses, per_sample_q_losses, per_sample_h_losses = jax.vmap(self.loss, in_axes=(None, 0, 0))(
            params, samples, importance_weights
        )

        return total_losses.mean(), (per_sample_q_losses, per_sample_h_losses)

    def loss(self, params: FrozenDict, sample: ReplayElement, importance_weight):
        # computes the loss for a single sample
        q_logits, h_logits = self.network.apply(params, sample.state)
        q_logits_a, h_logits_a = q_logits[sample.action], h_logits[sample.action]
        q_value_probs = jax.nn.softmax(q_logits_a)
        best_action = self.best_action(params, sample.next_state)

        next_q_probs = jax.nn.softmax(self.network.apply(params, sample.next_state)[0], axis=-1)[best_action]
        projected_target = self.compute_target(next_q_probs, sample)
        kl = jnp.sum(jax.lax.stop_gradient(h_logits_a) * projected_target) + optax.softmax_cross_entropy(
            q_logits_a, jax.lax.stop_gradient(projected_target)
        )
        h_loss = -jnp.sum(h_logits_a * jax.lax.stop_gradient(projected_target), axis=-1) + jax.scipy.special.logsumexp(
            h_logits_a, axis=-1, b=jax.lax.stop_gradient(q_value_probs)
        )
        return (
            importance_weight * (kl + h_loss),
            optax.softmax_cross_entropy(q_logits_a, projected_target),
            optax.softmax_cross_entropy(jnp.log(q_value_probs) + h_logits_a, projected_target),
        )

    def compute_target(self, next_q, sample: ReplayElement):
        # computes the target value for single sample
        target_atoms = sample.reward + (1 - sample.is_terminal) * self.gamma * self.support
        clipped_target_atoms = self.clip_target(target_atoms)
        b = (clipped_target_atoms - self.min_value) / self.bin_size
        l = jnp.clip(jnp.floor(b).astype(jnp.int32), 0, 50)
        u = jnp.clip(jnp.ceil(b).astype(jnp.int32), 0, 50)

        m = jnp.zeros(self.support.shape)
        m = m.at[l].add(next_q * (u.astype(b.dtype) - b))
        m = m.at[u].add(next_q * (b - l.astype(b.dtype)))
        m = m.at[l].add(next_q * (l == u))

        return m

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return jnp.argmax(jax.nn.softmax(self.network.apply(params, state)[0], axis=-1) @ self.support)

    def get_model(self):
        return {"params": self.params}
