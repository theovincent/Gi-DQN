from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.algorithms.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class DQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        features: list,
        architecture_type: str,
        layer_norm: bool,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_period: int,
        adam_eps: float = 1e-8,
    ):
        self.network = DQNNet(features, architecture_type, layer_norm, n_actions)
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)
        self.target_params = self.params.copy()

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_period = target_update_period
        self.cumulative_loss = 0
        self.cumulative_variance = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        for _ in range(int(self.update_to_data)):
            batch_samples, _ = replay_buffer.sample()

            self.params, self.optimizer_state, loss, variance = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )
            self.cumulative_loss += loss
            self.cumulative_variance += variance

    def update_target_params(self, step: int):
        if step % self.target_update_period == 0:
            self.target_params = self.params.copy()

            logs = {
                "loss": self.cumulative_loss / (self.target_update_period * self.update_to_data),
                "variance": self.cumulative_variance / (self.target_update_period * self.update_to_data),
            }
            self.cumulative_loss = 0
            self.cumulative_variance = 0

            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        params_target: FrozenDict,
        optimizer_state,
        batch_samples,
    ):
        ((loss, variance), grad_loss) = jax.value_and_grad(self.loss_on_batch, has_aux=True)(
            params, params_target, batch_samples
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss, variance

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        losses, variances = jax.vmap(self.loss, in_axes=(None, None, 0))(params, params_target, samples)
        return losses.mean(), variances.mean()

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        target = self.compute_target(params_target, sample)
        q_value = self.network.apply(params, sample.state)[sample.action]
        return jnp.square(q_value - target), (target**2 - target * q_value)

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            self.network.apply(params, sample.next_state)
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key=None):
        # computes the best action for a single state
        return jnp.argmax(self.network.apply(params, state))

    def get_model(self):
        return {"params": self.params}
