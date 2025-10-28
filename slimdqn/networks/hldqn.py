from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class HLDQN:
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
        n_bins: int,
        min_value: float,
        max_value: float,
        sigma: float,
        adam_eps: float = 1e-8,
    ):
        self.network = DQNNet(features, architecture_type, n_actions, n_bins=n_bins)
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)
        self.target_params = self.params

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.n_bins = n_bins
        self.sigma = sigma
        self.support = jnp.linspace(min_value, max_value, self.n_bins + 1, dtype=jnp.float32)
        self.bin_centers = (self.support[:-1] + self.support[1:]) / 2
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
        q_logits = self.network.apply(params, sample.state)[sample.action]

        next_q_probs = jax.nn.softmax(self.network.apply(params_target, sample.next_state), axis=-1)
        target = self.compute_target(next_q_probs @ self.bin_centers, sample)
        projected_target = self.project_target(target)
        return optax.softmax_cross_entropy(q_logits, projected_target)

    def compute_target(self, next_q, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(next_q)

    def project_target(self, target):
        erf_support = jax.scipy.special.erf((self.support - self.clip_target(target)) / (jnp.sqrt(2) * self.sigma))
        return (erf_support[1:] - erf_support[:-1]) / (erf_support[-1] - erf_support[0] + 1e-9)

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key=None):
        # computes the best action for a single state
        return jnp.argmax(jax.nn.softmax(self.network.apply(params, state), axis=-1) @ self.bin_centers)

    def get_model(self):
        return {"params": self.params}
