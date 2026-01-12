from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from slimdqn.algorithms.architectures.dqn import DQNNet

from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@jax.jit
def set_target_params(params):
    return optax.tree_utils.tree_set(params, q_heads=jax.tree.map(lambda p: p[0], params["params"]["q_heads"]))


@jax.jit
def shift_params(params):
    q_heads = jax.tree.map(lambda p: p.at[:-1].set(p[1:]), params["params"]["q_heads"])
    h_heads = jax.tree.map(lambda p: p.at[:-1].set(p[1:]), params["params"]["h_heads"])
    return optax.tree_utils.tree_set(params, q_heads=q_heads, h_heads=h_heads)


class GiDQNShared:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_bellman_iterations: int,
        features: list,
        architecture_type: str,
        layer_norm: bool,
        gap: bool,
        linear_heads: bool,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        freeze_first_head: bool,
        target_update_period: int,
        weight_decay: float,
        adam_eps: float = 1e-8,
    ):
        self.n_bellman_iterations = n_bellman_iterations
        self.n_actions = n_actions
        # 2K Networks: Q_1 to Q_K, TD-Surrogate_1 to TD-Surrogate_K-1
        self.online_networks = DQNNet(
            features,
            architecture_type,
            layer_norm,
            gap,
            linear_heads,
            n_actions,
            n_heads=self.n_bellman_iterations,
            n_h_heads=self.n_bellman_iterations - int(freeze_first_head),
        )
        self.root_network = DQNNet(
            features, architecture_type, layer_norm, gap, linear_heads, n_actions, n_heads=1, n_h_heads=0
        )

        # initialize 1 network with K q-heads and K OR K-1 h-heads
        self.params = self.online_networks.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))
        # initialize the target networks
        self.target_params = set_target_params(self.params)

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
        self.freeze_first_head = freeze_first_head
        self.target_update_period = target_update_period
        self.cumulative_q_losses = np.zeros(self.n_bellman_iterations)
        self.cumulative_h_losses = np.zeros(self.n_bellman_iterations - int(freeze_first_head))
        self.cumulative_variance = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        # Update the network parameters 'update_to_data' times every step
        for _ in range(int(max(self.update_to_data, 1))):
            # if update_to_data < 1, only perform one update if step = 0 [1 / self.update_to_data]
            if self.update_to_data < 1 and step % (1 / self.update_to_data) != 0:
                return None
            batch_samples, (sample_keys, importance_weights) = replay_buffer.sample()

            self.params, self.optimizer_state, per_sample_q_losses, per_sample_h_losses, variance = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples, importance_weights
            )

            replay_buffer.update(sample_keys, per_sample_q_losses.mean(axis=1) + per_sample_h_losses.mean(axis=1))

            self.cumulative_q_losses += per_sample_q_losses.mean(axis=0)
            self.cumulative_h_losses += per_sample_h_losses.mean(axis=0)
            self.cumulative_variance += variance

    def update_target_params(self, step: int):
        # shift the network parameters every `target_update_period` steps. This starts the next Bellman iteration
        if step % self.target_update_period == 0:
            self.target_params = set_target_params(self.params)
            # Window shift
            self.params = shift_params(self.params)

            self.logs = {
                "n_training_steps": step,
                "loss": np.mean(self.cumulative_q_losses) / (self.target_update_period * self.update_to_data),
                "variance": self.cumulative_variance / (self.target_update_period * self.update_to_data),
                "h_loss": np.mean(self.cumulative_h_losses) / (self.target_update_period * self.update_to_data),
            }
            for idx_network in range(0, min(5, self.n_bellman_iterations)):
                self.logs[f"networks/{idx_network}_loss"] = self.cumulative_q_losses[idx_network] / (
                    self.target_update_period * self.update_to_data
                )
            for idx_network in range(min(5, self.n_bellman_iterations - int(self.freeze_first_head))):
                self.logs[f"h_networks/{idx_network}_loss"] = self.cumulative_h_losses[idx_network] / (
                    self.target_update_period * self.update_to_data
                )

            self.cumulative_q_losses = np.zeros(self.n_bellman_iterations)
            self.cumulative_h_losses = np.zeros(self.n_bellman_iterations - int(self.freeze_first_head))
            self.cumulative_variance = 0
            return True
        return False

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: FrozenDict, target_params: FrozenDict, optimizer_state, batch_samples, importance_weights
    ):
        grad_loss, (per_sample_q_losses, per_sample_h_losses, variance) = jax.grad(self.loss_on_batch, has_aux=True)(
            params, target_params, batch_samples, importance_weights
        )

        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, per_sample_q_losses, per_sample_h_losses, variance

    def loss_on_batch(self, params: FrozenDict, target_params: FrozenDict, samples, importance_weights):
        # vmap to compute the loss for all samples
        total_losses, q_losses, h_losses, variances = jax.vmap(self.loss, in_axes=(None, None, 0, 0))(
            params, target_params, samples, importance_weights
        )
        return total_losses.mean(axis=0).sum(), (q_losses, h_losses, variances.mean())

    def loss(self, params: FrozenDict, target_params: FrozenDict, sample: ReplayElement, importance_weight):
        # computes the loss for a single sample
        q_outputs, h_outputs = self.online_networks.apply(params, sample.state)
        q_values, h_values = q_outputs[:, sample.action], h_outputs[:, sample.action]

        next_q_values_first_target = self.root_network.apply(target_params, sample.next_state)
        next_q_values_remaining_targets = self.online_networks.apply(params, sample.next_state)[0][:-1]
        next_q_values = jnp.concatenate([next_q_values_first_target[None, :], next_q_values_remaining_targets], axis=0)
        targets = self.compute_target(next_q_values, sample)
        # (K)
        td_errors = targets - q_values

        # (K - 1) if freeze_first_head is True OR (K)
        h_loss = h_values * jax.lax.stop_gradient(h_values - td_errors[int(self.freeze_first_head) :])

        target_loss = targets[int(self.freeze_first_head) :] * jax.lax.stop_gradient(h_values)
        if self.freeze_first_head:
            h_loss = jnp.append(jnp.zeros(1), h_loss)
            target_loss = jnp.append(jnp.zeros(1), target_loss)

        td_loss = target_loss - q_values * jax.lax.stop_gradient(td_errors)

        return (
            importance_weight * (td_loss + h_loss),
            jnp.square(td_errors),
            jnp.square(h_values - td_errors[int(self.freeze_first_head) :]),
            targets**2 - targets * q_values,
        )

    def compute_target(self, next_q_values, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            next_q_values, axis=-1
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return jnp.argmax(self.online_networks.apply(params, state)[0].mean(axis=0))

    def get_model(self):
        return {"params": self.params, "root_params": self.target_params}
