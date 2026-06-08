from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from slimdqn.algorithms.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class C51:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        features: list,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_period: int,
        adam_eps: float = 1e-8,
        n_bins: int = 51,
    ):
        self.vmin, self.vmax = -10, 10
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.support = jnp.linspace(self.vmin, self.vmax, self.n_bins)
        self.network = DQNNet(features, self.n_actions * self.n_bins, n_heads=1, n_h_heads=0)
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)
        self.target_params = self.params.copy()

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_period = target_update_period
        self.cumulative_loss = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        for _ in range(int(max(self.update_to_data, 1))):
            # if update_to_data < 1, only perform one update if step = 0 [1 / self.update_to_data]
            if self.update_to_data < 1 and step % (1 / self.update_to_data) != 0:
                return None
            batch_samples, (sample_keys, importance_weights) = replay_buffer.sample()

            self.params, self.optimizer_state, per_sample_q_loss = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples, importance_weights
            )

            replay_buffer.update(sample_keys, per_sample_q_loss)

            self.cumulative_loss += per_sample_q_loss.mean()

    def update_target_params(self, step: int):
        if step % self.target_update_period == 0:
            self.target_params = self.params.copy()

            self.logs = {
                "n_training_steps": step,
                "loss": self.cumulative_loss / (self.target_update_period * self.update_to_data),
            }
            self.cumulative_loss = 0

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: FrozenDict, params_target: FrozenDict, optimizer_state, batch_samples, importance_weights
    ):
        grad_loss, per_sample_q_loss = jax.grad(self.loss_on_batch, has_aux=True)(
            params, params_target, batch_samples, importance_weights
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, per_sample_q_loss

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples, importance_weights):
        losses, q_losses = jax.vmap(self.loss, in_axes=(None, None, 0, 0))(
            params, params_target, samples, importance_weights
        )
        return losses.mean(), q_losses

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample: ReplayElement, importance_weight):
        distribution_logits = self.network.apply(params, sample.state).reshape(self.n_actions, self.n_bins)
        distribution_probabilities = jax.nn.log_softmax(
            distribution_logits, axis=-1
        )  # log softmax for numerical stability

        target_distribution = self.compute_target(params_target, sample)

        cross_entropy = -jnp.sum(target_distribution * distribution_probabilities[sample.action, :])

        return cross_entropy * importance_weight, cross_entropy

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        distributional_next_logits = self.network.apply(params, sample.next_state).reshape(self.n_actions, self.n_bins)
        distributional_next_probabilities = jax.nn.softmax(distributional_next_logits, axis=-1)
        q_values = distributional_next_probabilities @ self.support
        best_action_index = jnp.argmax(q_values)
        distribution_next_probabilities_target = distributional_next_probabilities[best_action_index, :]

        non_aligned_target_atoms = (
            sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * self.support
        )
        clipped_non_aligned_atoms = jnp.clip(non_aligned_target_atoms, self.vmin, self.vmax)

        fractional_coordinate = (clipped_non_aligned_atoms - self.support[0]) / (
            (self.vmax - self.vmin) / (self.n_bins - 1)
        )
        lower, upper = jnp.floor(fractional_coordinate).astype(jnp.int32), jnp.ceil(fractional_coordinate).astype(
            jnp.int32
        )

        target_distribution = jnp.zeros(self.n_bins)
        target_distribution = target_distribution.at[lower].add(
            distribution_next_probabilities_target * (upper - fractional_coordinate)
        )
        target_distribution = target_distribution.at[upper].add(
            distribution_next_probabilities_target * (fractional_coordinate - lower)
        )
        target_distribution = target_distribution.at[lower].add(
            jnp.where(lower == upper, distribution_next_probabilities_target, 0.0)
        )
        return target_distribution

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        distribution_logits = self.network.apply(params, state).reshape(self.n_actions, self.n_bins)
        distributional_next_probabilities = jax.nn.softmax(distribution_logits, axis=-1)
        q_values = distributional_next_probabilities @ self.support
        return jnp.argmax(q_values)

    def get_model(self):
        return {"params": self.params}
