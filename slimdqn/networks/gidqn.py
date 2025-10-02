from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
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
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
        adam_eps: float = 1e-8,
        weight_decay: float = 0.001,
    ):
        key_params, key_z_params = jax.random.split(key, 2)

        self.n_bellman_iterations = n_bellman_iterations
        self.network = DQNNet(features, architecture_type, n_actions)

        self.params = jax.vmap(self.network.init, in_axes=(0, None))(
            jax.random.split(key_params, self.n_bellman_iterations + 1),
            jnp.zeros(observation_dim, dtype=jnp.float32),
        )  # initialize  K+1 online networks
        self.zparams = jax.vmap(self.network.init, in_axes=(0, None))(
            jax.random.split(key_z_params, self.n_bellman_iterations),
            jnp.zeros(observation_dim, dtype=jnp.float32),
        )  # initialize K td-estimator networks

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.z_optimizer = optax.adamw(learning_rate, eps=adam_eps, weight_decay=weight_decay)

        self.optimizer_state = self.optimizer.init(self.params)
        self.z_optimizer_state = self.z_optimizer.init(self.zparams)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.cumulative_q_losses = np.zeros(self.n_bellman_iterations)
        self.cumulative_z_losses = np.zeros(self.n_bellman_iterations)
        self.cumulative_variance = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample()

            (self.params, self.zparams, self.optimizer_state, self.z_optimizer_state, q_losses, z_losses, variance) = (
                self.learn_on_batch(
                    self.params,
                    self.zparams,
                    self.optimizer_state,
                    self.z_optimizer_state,
                    batch_samples,
                )
            )

            self.cumulative_q_losses += q_losses
            self.cumulative_z_losses += z_losses
            self.cumulative_variance += variance

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.params = shift_params(self.params)
            self.zparams = shift_params(self.zparams)

            logs = {
                "loss": np.mean(self.cumulative_q_losses) / (self.target_update_frequency / self.update_to_data),
                "variance": np.mean(self.cumulative_variance) / (self.target_update_frequency / self.update_to_data),
                "z_loss": np.mean(self.cumulative_z_losses) / (self.target_update_frequency / self.update_to_data),
            }
            for idx_network in range(0, min(5, self.n_bellman_iterations + 1)):
                logs[f"networks/{idx_network}_loss"] = self.cumulative_q_losses[idx_network] / (
                    self.target_update_frequency / self.update_to_data
                )
            for idx_network in range(min(5, self.n_bellman_iterations)):
                logs[f"z_networks/{idx_network}_loss"] = self.cumulative_z_losses[idx_network] / (
                    self.target_update_frequency / self.update_to_data
                )

            self.cumulative_q_losses = np.zeros(self.n_bellman_iterations)
            self.cumulative_z_losses = np.zeros(self.n_bellman_iterations)
            self.cumulative_variance = 0
            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        zparams: FrozenDict,
        optimizer_state,
        z_optimizer_state,
        batch_samples,
    ):
        (grad_loss, z_grad_loss), (q_losses, z_losses, variance) = jax.grad(
            self.loss_on_batch, has_aux=True, argnums=(0, 1)
        )(params, zparams, batch_samples)

        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)
        z_updates, z_optimizer_state = self.z_optimizer.update(z_grad_loss, z_optimizer_state, zparams)

        params = optax.apply_updates(params, updates)
        zparams = optax.apply_updates(zparams, z_updates)

        return (params, zparams, optimizer_state, z_optimizer_state, q_losses, z_losses, variance)

    def loss_on_batch(self, params: FrozenDict, zparams: FrozenDict, samples):
        total_losses, q_losses, z_losses, variances = jax.vmap(self.loss, in_axes=(None, None, 0))(
            params, zparams, samples
        )
        return total_losses.sum(axis=-1).mean(), (
            q_losses.mean(axis=0),
            z_losses.mean(axis=0),
            variances.mean(),
        )

    def loss(
        self,
        params: FrozenDict,
        zparams: FrozenDict,
        sample: ReplayElement,
    ):
        # computes the loss for a single sample
        q_values = jax.vmap(self.network.apply, in_axes=(0, None))(jax.tree.map(lambda x: x[1:], params), sample.state)[
            :, sample.action
        ]  # from 1 to n_bellman_iterations
        targets = jax.vmap(self.compute_target, in_axes=(0, None))(
            jax.tree.map(lambda x: x[:-1], params), sample
        )  # from 0 to n_bellman_iterations - 1
        td_errors = targets - q_values

        z_values = jax.vmap(self.network.apply, in_axes=(0, None))(zparams, sample.state)[:, sample.action]
        z_loss = z_values * jax.lax.stop_gradient(z_values - td_errors)

        targets = targets.at[0].set(
            0.0
        )  # cut off the gradient flow to the first Q-Network Q_0 by overwriting the first target with a constant
        td_loss = targets * jax.lax.stop_gradient(z_values) - q_values * jax.lax.stop_gradient(td_errors)

        return (
            td_loss + z_loss,
            jnp.square(td_errors),
            jnp.square(z_values - td_errors),
            (targets**2 - targets * q_values).mean(),
        )

        # TDs * jnp.square(targets - TDs);

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
        return {"params": self.params, "td_estimator_params": self.zparams}
