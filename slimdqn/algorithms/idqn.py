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
    # Each online network is updated to the following online network
    # \theta_k <- \theta_{k + 1}, i.e., params[k] <- params[k + 1]
    return jax.tree.map(lambda x: x.at[:-1].set(x[1:]), x)  # params[t] = params[t+1] for t in range(params)


@jax.jit
def sync_target_params(params, target_params):
    # Each target network is synchronized to the online network it represents
    # \bar{\theta}_k <- \theta_k, i.e., target_params[k] <- params[k-1]
    return jax.tree.map(lambda param, target_param: target_param.at[1:].set(param[:-1]), params, target_params)


class iDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_bellman_iterations: int,
        features: list,
        architecture_type: str,
        layer_norm: bool,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_period: int,
        target_sync_frequency: int,
        adam_eps: float = 1e-8,
    ):
        self.n_bellman_iterations = n_bellman_iterations
        self.network = DQNNet(features, architecture_type, layer_norm, n_actions)

        # initialize K networks
        self.params = jax.vmap(self.network.init, in_axes=(0, None))(
            jax.random.split(key, self.n_bellman_iterations), jnp.zeros(observation_dim, dtype=jnp.float32)
        )
        self.target_params = self.params.copy()  # initialize target networks

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_period = target_update_period
        self.target_sync_frequency = target_sync_frequency
        self.cumulative_losses = np.zeros(self.n_bellman_iterations)
        self.cumulative_variance = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        # update online network parameters every update_to_data steps
        for _ in range(int(self.update_to_data)):
            batch_samples, _ = replay_buffer.sample()

            (self.params, self.optimizer_state, losses, variance) = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )

            self.cumulative_losses += losses
            self.cumulative_variance += variance

    def update_target_params(self, step: int):
        # update target network parameters every target_update_period steps. This starts the next Bellman iteration
        if step % self.target_update_period == 0:
            # Each target network is updated to its respective online network
            # \bar{\theta}_k <- \theta_{k + 1}, i.e., target_params[k] <- params[k]
            self.target_params = self.params.copy()
            # Window shift
            self.params = shift_params(self.params)

            logs = {
                "loss": np.mean(self.cumulative_losses) / (self.target_update_period * self.update_to_data),
                "variance": np.mean(self.cumulative_variance) / (self.target_update_period * self.update_to_data),
            }
            for idx_network in range(0, min(5, self.n_bellman_iterations)):
                logs[f"networks/{idx_network}_loss"] = self.cumulative_losses[idx_network] / (
                    self.target_update_period * self.update_to_data
                )

            self.cumulative_losses = np.zeros(self.n_bellman_iterations)
            self.cumulative_variance = 0
            return True, logs
        # sync target network parameters to previous online network every target_sync_frequency steps
        if step % self.target_sync_frequency == 0:
            self.target_params = sync_target_params(self.params, self.target_params)
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, target_params: FrozenDict, optimizer_state, batch_samples):
        grad_loss, (losses, variance) = jax.grad(self.loss_on_batch, has_aux=True)(params, target_params, batch_samples)

        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)

        params = optax.apply_updates(params, updates)

        return (params, optimizer_state, losses, variance)

    def loss_on_batch(self, params: FrozenDict, target_params: FrozenDict, samples):
        losses, variances = jax.vmap(self.loss, in_axes=(None, None, 0))(params, target_params, samples)

        return losses.sum(axis=-1).mean(), (
            losses.mean(axis=0),  # mean over the samples but keep networks separated
            variances.mean(),
        )

    def loss(self, params: FrozenDict, target_params: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        q_values = jax.vmap(self.network.apply, in_axes=(0, None))(params, sample.state)[:, sample.action]
        targets = jax.vmap(self.compute_target, in_axes=(0, None))(target_params, sample)
        td_errors = targets - q_values

        return (jnp.square(td_errors), (targets**2 - targets * q_values).mean())

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            self.network.apply(params, sample.next_state)
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        # computes the best action for a single state
        idx_params = jax.random.randint(key, (), 1, self.n_bellman_iterations + 1)
        return jnp.argmax(self.network.apply(jax.tree.map(lambda param: param[idx_params], params), state))

    def get_model(self):
        return {"params": self.params}
