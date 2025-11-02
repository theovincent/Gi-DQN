from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class DQNRCShared:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        features: list,
        architecture_type: str,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
        weight_decay: float,
        mu: float,
        adam_eps: float = 1e-8,
    ):
        key, key_params = jax.random.split(key)
        self.n_actions = n_actions
        self.network = DQNNet(features, architecture_type, n_actions, n_heads=1, n_h_heads=1)

        # initialize online network
        self.params = self.network.init(key_params, jnp.zeros(observation_dim, dtype=jnp.float32))

        # regularize the TD-error estimator network
        self.optimizer = optax.adamw(
            learning_rate,
            eps=adam_eps,
            weight_decay=weight_decay,
            mask=jax.tree_util.tree_map_with_path(
                lambda path, leaf: (True if "Dense_final_h" in path[1].key else False), self.params
            ),
        )
        self.optimizer_state = self.optimizer.init(self.params)

        self.mu = mu
        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.cumulative_q_losses = 0
        self.cumulative_h_losses = 0
        self.cumulative_variance = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        # Update the network parameters every `update_to_data` steps
        for _ in range(int(self.update_to_data)):
            batch_samples, _ = replay_buffer.sample()

            (self.params, self.optimizer_state, q_losses, h_losses, variance) = self.learn_on_batch(
                self.params, self.optimizer_state, batch_samples
            )

            self.cumulative_q_losses += q_losses
            self.cumulative_h_losses += h_losses
            self.cumulative_variance += variance

    def update_target_params(self, step: int):
        # shift the network parameters every `target_update_frequency` steps. This starts the next Bellman iteration
        if step % self.target_update_frequency == 0:

            logs = {
                "loss": np.mean(self.cumulative_q_losses) / (self.target_update_frequency * self.update_to_data),
                "variance": np.mean(self.cumulative_variance) / (self.target_update_frequency * self.update_to_data),
                "h_loss": np.mean(self.cumulative_h_losses) / (self.target_update_frequency * self.update_to_data),
            }

            self.cumulative_q_losses = 0
            self.cumulative_h_losses = 0
            self.cumulative_variance = 0
            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples):
        (grad_loss), (q_losses, h_losses, variance) = jax.grad(self.loss_on_batch, has_aux=True)(params, batch_samples)

        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)

        params = optax.apply_updates(params, updates)

        return (params, optimizer_state, q_losses, h_losses, variance)

    def loss_on_batch(self, params: FrozenDict, samples):
        # vmap to compute the loss for all samples
        total_losses, q_losses, h_losses, variances = jax.vmap(self.loss, in_axes=(None, 0))(params, samples)
        return total_losses.mean(), (q_losses.mean(), h_losses.mean(), variances.mean())

    def loss(self, params: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        q_output, h_output = self.network.apply(params, sample.state)
        q_value, h_value = q_output[0, sample.action], h_output[0, sample.action]
        next_q_value = self.network.apply(params, sample.next_state)[0][0]

        target = self.compute_target(next_q_value, sample)
        td_error = target - q_value
        h_loss = h_value * jax.lax.stop_gradient(h_value - td_error)
        td_loss = target * jax.lax.stop_gradient(h_value) - q_value * jax.lax.stop_gradient(td_error)

        return (
            td_loss + self.mu * h_loss,
            jnp.square(td_error),
            jnp.square(h_value - td_error),
            target**2 - target * q_value,
        )

    def compute_target(self, next_q, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(next_q)

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array = None):
        # computes the best action for a single state
        return jnp.argmax(self.network.apply(params, state)[0][0])

    def get_model(self):
        return {"params": self.params}
