from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.algorithms.architectures.dqn import DQNNet

from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@partial(jax.jit, static_argnames="n_actions")
def set_target_params(params, n_actions):
    q_heads = jax.tree.map(lambda p: p[..., :n_actions], params["params"]["q_heads"])
    return optax.tree_utils.tree_set(params, q_heads=q_heads, h_heads=None)


@partial(jax.jit, static_argnames="n_actions")
def shift_params(params, n_actions):
    q_heads = jax.tree.map(lambda p: p.at[..., :-n_actions].set(p[..., n_actions:]), params["params"]["q_heads"])
    h_heads = jax.tree.map(lambda p: p.at[..., :-n_actions].set(p[..., n_actions:]), params["params"]["h_heads"])

    return optax.tree_utils.tree_set(params, q_heads=q_heads, h_heads=h_heads)


class GiC51Shared:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_bellman_iterations: int,
        features: list,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_period: int,
        weight_decay: float,
        adam_eps: float = 1e-8,
        n_bins: int = 51,
        vmin: int = -10,
        vmax: int = 10,
    ):
        self.vmin, self.vmax = vmin, vmax
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.support = jnp.linspace(self.vmin, self.vmax, self.n_bins)
        self.n_bellman_iterations = n_bellman_iterations
        self.n_actions = n_actions
        # 2K Networks: Q_0, Q_1 to Q_K, TD-Surrogate_1 to TD-Surrogate_K-1
        self.online_networks = DQNNet(
            features, n_actions * n_bins, n_heads=self.n_bellman_iterations, n_h_heads=self.n_bellman_iterations - 1
        )
        self.root_network = DQNNet(features, n_actions * n_bins, n_heads=1, n_h_heads=0)

        # initialize 1 network with K q-heads and K OR K-1 h-heads
        self.params = self.online_networks.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))
        # initialize the target network
        self.target_params = set_target_params(self.params, self.n_actions * self.n_bins)

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
        self.cumulative_q_losses = np.zeros(self.n_bellman_iterations)
        self.cumulative_h_losses = np.zeros(self.n_bellman_iterations - 1)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        for _ in range(int(max(self.update_to_data, 1))):
            # if update_to_data < 1, only perform one update if step = 0 [1 / self.update_to_data]
            if self.update_to_data < 1 and step % (1 / self.update_to_data) != 0:
                return None
            batch_samples, (sample_keys, importance_weights) = replay_buffer.sample()

            self.params, self.optimizer_state, per_sample_q_losses, per_sample_h_losses = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples, importance_weights
            )

            replay_buffer.update(
                sample_keys, per_sample_q_losses.mean(axis=1) + jnp.maximum(per_sample_h_losses.mean(axis=1), 0.0)
            )  # jnp.max because Donsker-Varadhan term can be negative in early training

            self.cumulative_q_losses += per_sample_q_losses.mean(axis=0)
            self.cumulative_h_losses += per_sample_h_losses.mean(axis=0)

    def update_target_params(self, step: int):
        if step % self.target_update_period == 0:
            self.target_params = set_target_params(self.params, self.n_actions * self.n_bins)
            self.params = shift_params(self.params, self.n_actions * self.n_bins)

            self.logs = {
                "n_training_steps": step,
                "loss": np.mean(self.cumulative_q_losses) / (self.target_update_period * self.update_to_data),
                "h_loss": np.mean(self.cumulative_h_losses) / (self.target_update_period * self.update_to_data),
            }
            # for idx_network in range(0, min(5, self.n_bellman_iterations)):
            #     self.logs[f"networks/{idx_network}_loss"] = self.cumulative_q_losses[idx_network] / (
            #         self.target_update_period * self.update_to_data
            #     )
            # for idx_network in range(min(5, self.n_bellman_iterations - 1)):
            #     self.logs[f"h_networks/{idx_network}_loss"] = self.cumulative_h_losses[idx_network] / (
            #         self.target_update_period * self.update_to_data
            #     )

            self.cumulative_q_losses = np.zeros(self.n_bellman_iterations)
            self.cumulative_h_losses = np.zeros(self.n_bellman_iterations - 1)

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: FrozenDict, target_params: FrozenDict, optimizer_state, batch_samples, importance_weights
    ):
        grad_loss, (per_sample_q_losses, per_sample_h_losses) = jax.grad(self.loss_on_batch, has_aux=True)(
            params, target_params, batch_samples, importance_weights
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, per_sample_q_losses, per_sample_h_losses

    def loss_on_batch(self, params: FrozenDict, target_params: FrozenDict, samples, importance_weights):
        total_losses, q_losses, h_losses = jax.vmap(self.loss, in_axes=(None, None, 0, 0))(
            params, target_params, samples, importance_weights
        )

        return total_losses.mean(axis=0).sum(), (q_losses, h_losses)

    def loss(self, params: FrozenDict, target_params: FrozenDict, sample: ReplayElement, importance_weight):
        q_logits, h_logits = self.online_networks.apply(params, sample.state)
        q_logits, h_logits = q_logits.reshape(-1, self.n_actions, self.n_bins), h_logits.reshape(
            -1, self.n_actions, self.n_bins
        )  # (K, n_actions, n_bins), (K-1, n_actions, n_bins)
        q_log_distribution = jax.nn.log_softmax(q_logits, axis=-1)  # (K, n_actions, n_bins)
        h_logits = h_logits[:, sample.action, :]  # (K-1, n_bins)

        next_q_logits_first_target = self.root_network.apply(target_params, sample.next_state).reshape(
            self.n_actions, self.n_bins
        )  # (n_actions, n_bins)
        next_q_distribution_first_target = jax.nn.softmax(next_q_logits_first_target, axis=-1)  # (n_actions, n_bins)
        next_q_logits_remaining_targets = self.online_networks.apply(params, sample.next_state)[0].reshape(
            -1, self.n_actions, self.n_bins
        )[
            :-1
        ]  # (K-1, n_actions, n_bins)
        next_q_distribution_remaining_targets = jax.nn.softmax(next_q_logits_remaining_targets, axis=-1)

        next_q_distribution = jnp.concatenate(
            [next_q_distribution_first_target[None, :, :], next_q_distribution_remaining_targets], axis=0
        )  # (K, n_actions, n_bins)

        target_distribution = self.compute_target(next_q_distribution, sample)  # (K, n_bins)

        target_loss = jnp.sum(target_distribution[1:, :] * jax.lax.stop_gradient(h_logits), axis=-1)  # (K-1,)

        h_loss = jnp.sum(
            jax.lax.stop_gradient(target_distribution[1:, :]) * h_logits, axis=-1
        ) - jax.scipy.special.logsumexp(
            h_logits + jax.lax.stop_gradient(q_log_distribution[1:, sample.action, :]), axis=-1
        )  # (K-1,)

        target_loss = jnp.append(jnp.zeros(1), target_loss)  # (K,)
        h_loss = jnp.append(jnp.zeros(1), h_loss)  # (K,)

        td_loss = target_loss - jnp.sum(
            jax.lax.stop_gradient(target_distribution) * q_log_distribution[:, sample.action, :], axis=-1
        )  # (K,)

        cross_entropy = -jnp.sum(
            jax.lax.stop_gradient(target_distribution) * q_log_distribution[:, sample.action, :], axis=-1
        )  # (K,)

        suboptimality = (
            cross_entropy + jnp.sum(jax.scipy.special.xlogy(target_distribution, target_distribution), axis=-1) - h_loss
        )  # KL - DV >= 0

        return (
            importance_weight * (td_loss - h_loss),
            cross_entropy,
            suboptimality[1:],
        )

    def compute_target(self, next_probabilities, sample: ReplayElement):
        next_q_values = next_probabilities @ self.support  # (K, n_actions)
        best_action_index = jnp.argmax(next_q_values, axis=-1)  # (K,)
        next_probabilities_target = next_probabilities[
            jnp.arange(self.n_bellman_iterations), best_action_index, :
        ]  # (K, n_bins)

        non_aligned_target_atoms = (
            sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * self.support
        )  # (K, bins)
        clipped_non_aligned_target_atoms = jnp.clip(non_aligned_target_atoms, self.vmin, self.vmax)  # (K, n_bins)

        fractional_coordinates = (clipped_non_aligned_target_atoms - self.support[0]) / (
            (self.vmax - self.vmin) / (self.n_bins - 1)
        )  # (K, n_bins)
        lower, upper = jnp.floor(fractional_coordinates).astype(jnp.int32), jnp.ceil(fractional_coordinates).astype(
            jnp.int32
        )  # (K, n_bins), (K, n_bins)

        target_distribution = jnp.zeros((self.n_bellman_iterations, self.n_bins))  # (K, n_bins)
        target_distribution = target_distribution.at[:, lower].add(
            next_probabilities_target * (upper - fractional_coordinates)
        )  # (K, n_bins)
        target_distribution = target_distribution.at[:, upper].add(
            next_probabilities_target * (fractional_coordinates - lower)
        )  # (K, n_bins)
        target_distribution = target_distribution.at[:, lower].add(
            jnp.where(lower == upper, next_probabilities_target, 0.0)
        )  # (K, n_bins)
        return target_distribution  # (K, n_bins)

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        logits = self.online_networks.apply(params, state)[0].reshape(-1, self.n_actions, self.n_bins)
        probabilities = jax.nn.softmax(logits, axis=-1)
        q_values = (probabilities @ self.support).mean(axis=0)
        return jnp.argmax(q_values)

    def get_model(self):
        return {"params": self.params}
