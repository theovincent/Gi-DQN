from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class C51:
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
        min_value: float,
        max_value: float,
        adam_eps: float = 1e-8,
    ):
        self.network = DQNNet(features, architecture_type, n_actions, n_bins=51)
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)
        self.target_params = self.params

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.min_value = min_value
        self.max_value = max_value
        self.support = jnp.linspace(min_value, max_value, 51, dtype=jnp.float32)
        self.bin_size = (max_value - min_value) / 50
        self.clip_target = lambda target: jnp.clip(target, min_value, max_value)
        self.cumulative_loss = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        for _ in range(int(self.update_to_data)):
            batch_samples, _ = replay_buffer.sample()

            self.params, self.optimizer_state, loss = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )
            self.cumulative_loss += loss

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.target_params = self.params.copy()

            logs = {
                "loss": self.cumulative_loss / (self.target_update_frequency * self.update_to_data),
            }
            self.cumulative_loss = 0

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
        loss, grad_loss = jax.value_and_grad(self.loss_on_batch)(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        losses = jax.vmap(self.loss, in_axes=(None, None, 0))(params, params_target, samples)
        return losses.mean()

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        q_logits = self.network.apply(params, sample.state)
        best_action = self.best_action(params, sample.next_state)

        next_q_probs = jax.nn.softmax(self.network.apply(params_target, sample.next_state), axis=-1)
        projected_target = self.compute_target(next_q_probs[best_action], sample)
        return optax.softmax_cross_entropy(q_logits[sample.action], projected_target)

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
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key=None):
        # computes the best action for a single state
        return jnp.argmax(jax.nn.softmax(self.network.apply(params, state), axis=-1) @ self.support)

    def get_model(self):
        return {"params": self.params}
