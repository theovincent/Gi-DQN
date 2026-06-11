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
    return optax.tree_utils.tree_set(params, q_heads=q_heads)


@partial(jax.jit, static_argnames="n_actions")
def shift_params(params, n_actions):
    q_heads = jax.tree.map(lambda p: p.at[..., :-n_actions].set(p[..., n_actions:]), params["params"]["q_heads"])
    return optax.tree_utils.tree_set(params, q_heads=q_heads)


class ic51Shared:
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
        adam_eps: float = 1e-8,
        n_bins: int = 51,
        vmin: int = -10,
        vmax: int = 10,
    ):
        self.vmin, self.vmax = vmin, vmax
        self.n_bins = n_bins
        self.support = jnp.linspace(self.vmin, self.vmax, self.n_bins)
        self.n_actions = n_actions
        self.n_bellman_iterations = n_bellman_iterations
        self.online_networks = DQNNet(features, n_actions * n_bins, n_heads=self.n_bellman_iterations, n_h_heads=0)
        self.root_network = DQNNet(features, n_actions * n_bins, n_heads=1, n_h_heads=0)

        # initialize 1 network with K heads
        self.params = self.online_networks.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))
        # initialize the target network
        self.target_params = set_target_params(self.params, self.n_actions * self.n_bins)

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_period = target_update_period
        self.cumulative_losses = np.zeros(self.n_bellman_iterations)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        for _ in range(int(max(self.update_to_data, 1))):
            # if update_to_data < 1, only perform one update if step = 0 [1 / self.update_to_data]
            if self.update_to_data < 1 and step % (1 / self.update_to_data) != 0:
                return None
            batch_samples, (sample_keys, importance_weights) = replay_buffer.sample()

            self.params, self.optimizer_state, per_sample_q_losses = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples, importance_weights
            )

            replay_buffer.update(sample_keys, per_sample_q_losses.mean(axis=1))

            self.cumulative_losses += per_sample_q_losses.mean(axis=0)

    def update_target_params(self, step: int):
        if step % self.target_update_period == 0:
            self.target_params = set_target_params(self.params, self.n_actions * self.n_bins)
            self.params = shift_params(self.params, self.n_actions * self.n_bins)

            self.logs = {
                "n_training_steps": step,
                "loss": np.mean(self.cumulative_losses) / (self.target_update_period * self.update_to_data),
            }
            # for idx_network in range(0, min(5, self.n_bellman_iterations)):
            #     self.logs[f"networks/{idx_network}_loss"] = self.cumulative_losses[idx_network] / (
            #         self.target_update_period * self.update_to_data
            #     )

            self.cumulative_losses = np.zeros(self.n_bellman_iterations)

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: FrozenDict, target_params: FrozenDict, optimizer_state, batch_samples, importance_weights
    ):
        grad_loss, (per_sample_q_losses) = jax.grad(self.loss_on_batch, has_aux=True)(
            params, target_params, batch_samples, importance_weights
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, per_sample_q_losses

    def loss_on_batch(self, params: FrozenDict, target_params: FrozenDict, samples, importance_weights):
        total_losses, q_losses = jax.vmap(self.loss, in_axes=(None, None, 0, 0))(
            params, target_params, samples, importance_weights
        )

        return total_losses.mean(axis=0).sum(), q_losses

    def loss(self, params: FrozenDict, target_params: FrozenDict, sample: ReplayElement, importance_weight):
        logits = self.online_networks.apply(params, sample.state).reshape(
            -1, self.n_actions, self.n_bins
        )  # (K, n_actions, n_bins)
        log_distribution = jax.nn.log_softmax(logits, axis=-1)  # (K, n_actions, n_bins)

        next_logits_first_target = self.root_network.apply(target_params, sample.next_state).reshape(
            self.n_actions, self.n_bins
        )  # (n_actions, n_bins)
        next_distribution_first_target = jax.nn.softmax(next_logits_first_target, axis=-1)  # (n_actions, n_bins)

        next_logits_remaining_targets = self.online_networks.apply(params, sample.next_state).reshape(
            -1, self.n_actions, self.n_bins
        )[
            :-1
        ]  # (K-1, n_actions, n_bins)
        next_distribution_remaining_targets = jax.nn.softmax(
            next_logits_remaining_targets, axis=-1
        )  # ( K-1, n_actions, n_bins)

        next_distribution = jnp.concatenate(
            [next_distribution_first_target[None, :, :], next_distribution_remaining_targets], axis=0
        )  # ( K, n_actions, n_bins)

        target_distribution = self.compute_target(next_distribution, sample)  # (K, n_bins)

        cross_entropy = -jnp.sum(
            jax.lax.stop_gradient(target_distribution) * log_distribution[:, sample.action, :], axis=-1
        )  # (K,)

        return importance_weight * cross_entropy, cross_entropy

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
        logits = self.online_networks.apply(params, state).reshape(-1, self.n_actions, self.n_bins)
        probabilities = jax.nn.softmax(logits, axis=-1)
        q_values = (probabilities @ self.support).mean(axis=0)
        return jnp.argmax(q_values)

    def get_model(self):
        return {"params": self.params}
