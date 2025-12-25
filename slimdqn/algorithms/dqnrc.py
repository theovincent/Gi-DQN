from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from slimdqn.algorithms.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class DQNRC:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        features: list,
        architecture_type: str,
        layer_norm: bool,
        gap: bool,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_period: int,
        weight_decay: float,
        adam_eps: float = 1e-8,
    ):
        key_params, key_h_params = jax.random.split(key, 2)

        self.network = DQNNet(features, architecture_type, layer_norm, gap, False, n_actions, n_heads=1, n_h_heads=0)

        # initialize online network
        self.params = self.network.init(key_params, jnp.zeros(observation_dim, dtype=jnp.float32))
        # initialize TD-error estimator networks
        self.h_params = self.network.init(key_h_params, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        # regularize the TD-error estimator network
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
        self.target_update_period = target_update_period
        self.cumulative_q_loss = 0
        self.cumulative_h_loss = 0
        self.cumulative_variance = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        for _ in range(int(max(self.update_to_data, 1))):
            # if update_to_data < 1, only perform one update if step = 0 [1 / self.update_to_data]
            if self.update_to_data < 1 and step % (1 / self.update_to_data) != 0:
                return None
            batch_samples, _ = replay_buffer.sample()

            self.params, self.h_params, self.optimizer_state, self.h_optimizer_state, q_loss, h_loss, variance = (
                self.learn_on_batch(
                    self.params, self.h_params, self.optimizer_state, self.h_optimizer_state, batch_samples
                )
            )

            self.cumulative_q_loss += q_loss
            self.cumulative_h_loss += h_loss
            self.cumulative_variance += variance

    def update_target_params(self, step: int):
        # shift the network parameters every `target_update_period` steps. This starts the next Bellman iteration
        if step % self.target_update_period == 0:
            logs = {
                "loss": self.cumulative_q_loss / (self.target_update_period * self.update_to_data),
                "variance": self.cumulative_variance / (self.target_update_period * self.update_to_data),
                "h_loss": self.cumulative_h_loss / (self.target_update_period * self.update_to_data),
            }

            self.cumulative_q_loss = 0
            self.cumulative_h_loss = 0
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
        # vmap to compute the loss for all samples
        total_losses, q_losses, h_losses, variances = jax.vmap(self.loss, in_axes=(None, None, 0))(
            params, h_params, samples
        )
        return total_losses.mean(), (q_losses.mean(), h_losses.mean(), variances.mean())

    def loss(self, params: FrozenDict, h_params: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        q_value = self.network.apply(params, sample.state)[sample.action]
        target = self.compute_target(params, sample)
        td_error = target - q_value

        h_value = self.network.apply(h_params, sample.state)[sample.action]
        h_loss = h_value * jax.lax.stop_gradient(h_value - td_error)

        td_loss = target * jax.lax.stop_gradient(h_value) - q_value * jax.lax.stop_gradient(td_error)

        return td_loss + h_loss, jnp.square(td_error), jnp.square(h_value - td_error), target**2 - target * q_value

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            self.network.apply(params, sample.next_state)
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return jnp.argmax(self.network.apply(params, state))

    def get_model(self):
        return {"params": self.params, "td_estimator_params": self.h_params}
