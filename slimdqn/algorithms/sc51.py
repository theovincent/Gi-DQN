from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from slimdqn.algorithms.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@partial(jax.jit, static_argnames="n_actions")
def shift_params(params, n_actions):
    q_heads = jax.tree.map(lambda p: p.at[..., :-n_actions].set(p[..., n_actions:]), params["params"]["q_heads"])
    return optax.tree_utils.tree_set(params, q_heads=q_heads)


class SC51:
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
        vmin: int = -10,
        vmax: int = 10,
    ):
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.support = jnp.linspace(vmin, vmax, self.n_bins)
        self.network = DQNNet(features, self.n_actions * self.n_bins, n_heads=2, n_h_heads=0)
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.n_actions = n_actions
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
                self.params, self.optimizer_state, batch_samples, importance_weights
            )

            replay_buffer.update(sample_keys, per_sample_q_loss)

            self.cumulative_loss += per_sample_q_loss.mean()

    def update_target_params(self, step: int):
        if step % self.target_update_period == 0:
            self.params = shift_params(self.params, self.n_actions * self.n_bins)

            self.logs = {
                "n_training_steps": step,
                "loss": self.cumulative_loss / (self.target_update_period * self.update_to_data),
            }
            self.cumulative_loss = 0

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples, importance_weights):
        grad_loss, per_sample_q_loss = jax.grad(self.loss_on_batch, has_aux=True)(
            params, batch_samples, importance_weights
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, per_sample_q_loss

    def loss_on_batch(self, params: FrozenDict, samples, importance_weights):
        losses, q_losses = jax.vmap(self.loss, in_axes=(None, 0, 0))(params, samples, importance_weights)
        return losses.mean(), q_losses

    def loss(self, params: FrozenDict, sample: ReplayElement, importance_weight):
        logits = self.network.apply(params, sample.state)[1].reshape(self.n_actions, self.n_bins)[
            sample.action, :
        ]  # (n_actions, n_bins) -> (n_bins,)
        log_distribution = jax.nn.log_softmax(logits, axis=-1)  # (n_bins,)

        target_distribution = self.compute_target(params, sample)  # (n_bins,)

        cross_entropy = -jnp.sum(jax.lax.stop_gradient(target_distribution) * log_distribution)  # (1,)

        return cross_entropy * importance_weight, cross_entropy

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        next_logits = self.network.apply(params, sample.next_state)[0].reshape(
            self.n_actions, self.n_bins
        )  # (n_actions, n_bins)
        next_probabilities = jax.nn.softmax(next_logits, axis=-1)  # (n_actions, n_bins)
        next_q_values = next_probabilities @ self.support  # (n_actions,)
        best_action_index = jnp.argmax(next_q_values)  # (1,)
        next_probabilities_target = next_probabilities[best_action_index, :]  # (n_bins,)

        non_aligned_target_atoms = (
            sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * self.support
        )  # (n_bins,)
        clipped_non_aligned_target_atoms = jnp.clip(
            non_aligned_target_atoms, self.support[0], self.support[-1]
        )  # (n_bins,)

        fractional_coordinates = (clipped_non_aligned_target_atoms - self.support[0]) / (
            (self.support[-1] - self.support[0]) / (self.n_bins - 1)
        )  # (n_bins,)
        lower, upper = jnp.floor(fractional_coordinates).astype(jnp.int32), jnp.ceil(fractional_coordinates).astype(
            jnp.int32
        )  # (n_bins,), (n_bins,)

        target_distribution = jnp.zeros(self.n_bins)  # (n_bins,)
        target_distribution = target_distribution.at[lower].add(
            next_probabilities_target * (upper - fractional_coordinates)
        )  # (n_bins,)
        target_distribution = target_distribution.at[upper].add(
            next_probabilities_target * (fractional_coordinates - lower)
        )  # (n_bins,)
        target_distribution = target_distribution.at[lower].add(
            jnp.where(lower == upper, next_probabilities_target, 0.0)
        )  # (n_bins,)
        return target_distribution  # (n_bins,)

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        logits = self.network.apply(params, state)[1].reshape(self.n_actions, self.n_bins)
        probabilities = jax.nn.softmax(logits, axis=-1)
        q_values = probabilities @ self.support
        return jnp.argmax(q_values)

    def get_model(self):
        return {"params": self.params}
