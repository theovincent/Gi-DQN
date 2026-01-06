from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.algorithms.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@jax.jit
def shift_params(x):
    return jax.tree.map(lambda x: x.at[:-1].set(x[1:]), x)  # params[t] = params[t+1] for t in range(params)


class GiDQN:
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
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        freeze_first_head: bool,
        target_update_period: int,
        weight_decay: float,
        adam_eps: float = 1e-8,
    ):
        key_params, key_h_params = jax.random.split(key)

        self.n_bellman_iterations = n_bellman_iterations
        self.network = DQNNet(features, architecture_type, layer_norm, gap, False, n_actions, n_heads=1, n_h_heads=0)

        # initialize K+1 online networks
        self.params = jax.vmap(self.network.init, in_axes=(0, None))(
            jax.random.split(key_params, self.n_bellman_iterations + 1), jnp.zeros(observation_dim, dtype=jnp.float32)
        )
        # initialize K - 1 TD-error estimator networks OR K TD-error estimator networks if the first head is not frozen
        self.h_params = jax.vmap(self.network.init, in_axes=(0, None))(
            jax.random.split(key_h_params, self.n_bellman_iterations - int(freeze_first_head)),
            jnp.zeros(observation_dim, dtype=jnp.float32),
        )

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        # regularize the TD-error estimator networks
        self.h_optimizer = optax.adamw(
            learning_rate,
            eps=adam_eps,
            weight_decay=weight_decay,
        )

        self.optimizer_state = self.optimizer.init(self.params)
        self.h_optimizer_state = self.h_optimizer.init(self.h_params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.freeze_first_head = freeze_first_head
        self.target_update_period = target_update_period
        self.cumulative_q_losses = np.zeros(self.n_bellman_iterations)
        self.cumulative_h_losses = np.zeros(self.n_bellman_iterations - int(freeze_first_head))
        self.cumulative_variance = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        for _ in range(int(max(self.update_to_data, 1))):
            # if update_to_data < 1, only perform one update if step = 0 [1 / self.update_to_data]
            if self.update_to_data < 1 and step % (1 / self.update_to_data) != 0:
                return None
            batch_samples, _ = replay_buffer.sample()

            self.params, self.h_params, self.optimizer_state, self.h_optimizer_state, q_losses, h_losses, variance = (
                self.learn_on_batch(
                    self.params, self.h_params, self.optimizer_state, self.h_optimizer_state, batch_samples
                )
            )

            self.cumulative_q_losses += q_losses
            self.cumulative_h_losses += h_losses
            self.cumulative_variance += variance

    def update_target_params(self, step: int):
        # shift the network parameters every `target_update_period` steps. This starts the next Bellman iteration
        if step % self.target_update_period == 0:
            self.params = shift_params(self.params)
            self.h_params = shift_params(self.h_params)

            logs = {
                "loss": np.mean(self.cumulative_q_losses) / (self.target_update_period * self.update_to_data),
                "variance": self.cumulative_variance / (self.target_update_period * self.update_to_data),
                "h_loss": np.mean(self.cumulative_h_losses) / (self.target_update_period * self.update_to_data),
            }
            for idx_network in range(0, min(5, self.n_bellman_iterations)):
                logs[f"networks/{idx_network}_loss"] = self.cumulative_q_losses[idx_network] / (
                    self.target_update_period * self.update_to_data
                )
            for idx_network in range(min(5, self.n_bellman_iterations - int(self.freeze_first_head))):
                logs[f"h_networks/{idx_network}_loss"] = self.cumulative_h_losses[idx_network] / (
                    self.target_update_period * self.update_to_data
                )

            self.cumulative_q_losses = np.zeros(self.n_bellman_iterations)
            self.cumulative_h_losses = np.zeros(self.n_bellman_iterations - int(self.freeze_first_head))
            self.cumulative_variance = 0
            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: FrozenDict, h_params: FrozenDict, optimizer_state, h_optimizer_state, batch_samples
    ):
        (grad_loss, h_grad_loss), (q_losses, h_losses, variance) = jax.grad(
            self.loss_on_batch, has_aux=True, argnums=(0, 1)
        )(params, h_params, batch_samples)

        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)
        h_updates, h_optimizer_state = self.h_optimizer.update(h_grad_loss, h_optimizer_state, h_params)

        params = optax.apply_updates(params, updates)
        h_params = optax.apply_updates(h_params, h_updates)

        return params, h_params, optimizer_state, h_optimizer_state, q_losses, h_losses, variance

    def loss_on_batch(self, params: FrozenDict, h_params: FrozenDict, samples):
        # vmap to compute the loss for all samples (batch_size, K)
        total_losses, q_losses, h_losses, variances = jax.vmap(self.loss, in_axes=(None, None, 0))(
            params, h_params, samples
        )
        return total_losses.mean(axis=0).sum(), (q_losses.mean(axis=0), h_losses.mean(axis=0), variances.mean())

    def loss(
        self,
        params: FrozenDict,
        h_params: FrozenDict,
        sample: ReplayElement,
    ):
        # computes the loss for a single sample
        q_values = jax.vmap(self.network.apply, in_axes=(0, None))(jax.tree.map(lambda x: x[1:], params), sample.state)[
            :, sample.action
        ]  # use networks 1 to K to compute Q-values
        targets = jax.vmap(self.compute_target, in_axes=(0, None))(
            jax.tree.map(lambda x: x[:-1], params), sample
        )  # use networks 0 to K - 1 to compute targets
        # (K)
        td_errors = targets - q_values

        # (K - 1) if freeze_first_head is True OR (K)
        h_values = jax.vmap(self.network.apply, in_axes=(0, None))(h_params, sample.state)[:, sample.action]
        h_loss = h_values * jax.lax.stop_gradient(h_values - td_errors[int(self.freeze_first_head) :])

        target_loss = targets[int(self.freeze_first_head) :] * jax.lax.stop_gradient(h_values)
        if self.freeze_first_head:
            h_loss = jnp.append(jnp.zeros(1), h_loss)
            target_loss = jnp.append(jnp.zeros(1), target_loss)

        td_loss = target_loss - q_values * jax.lax.stop_gradient(td_errors)

        return (
            td_loss + h_loss,
            jnp.square(td_errors),
            jnp.square(h_values - td_errors[int(self.freeze_first_head) :]),
            targets**2 - targets * q_values,
        )

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            self.network.apply(params, sample.next_state)
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        q_predictions = jax.vmap(self.network.apply, in_axes=(0, None))(params, state)[1:]
        return jnp.argmax(jnp.mean(q_predictions, axis=0))

    def get_model(self):
        return {"params": self.params, "td_estimator_params": self.h_params}
