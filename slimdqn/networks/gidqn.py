from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from wandb.jupyter import notebook_metadata

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@jax.jit
def shift_params(x):
    return jax.tree.map(
        lambda x: x.at[:-1].set(x[1:]), x
    )  # params[t] = params[t+1] for t in range(params)


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
    ):
        key_params, key_z_params = jax.random.split(key, 2)

        self.n_bellman_iterations = n_bellman_iterations
        self.network = DQNNet(features, architecture_type, n_actions)

        self.params = jax.vmap(self.network.init, in_axes=(0, None))(
            jax.random.split(key_params, self.n_bellman_iterations + 1),
            jnp.zeros(observation_dim, dtype=jnp.float32),
        )  # 0 to K
        self.zparams = jax.vmap(self.network.init, in_axes=(0, None))(
            jax.random.split(key_z_params, self.n_bellman_iterations),
            jnp.zeros(observation_dim, dtype=jnp.float32),
        )  # 1 to K

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.z_optimizer = optax.adamw(learning_rate, eps=adam_eps, weight_decay=0.001)

        self.optimizer_state = self.optimizer.init(self.params)
        self.z_optimizer_state = self.z_optimizer.init(self.zparams)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.cumulated_loss = np.zeros(self.n_bellman_iterations)
        self.z_means = np.zeros(self.n_bellman_iterations)
        self.variances = np.zeros(self.n_bellman_iterations)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample()

            (
                self.params,
                self.zparams,
                self.optimizer_state,
                self.z_optimizer_state,
                z_means_and_target_variances,
            ) = self.learn_on_batch(
                self.params,
                self.zparams,
                self.optimizer_state,
                self.z_optimizer_state,
                batch_samples,
            )
            self.z_means += z_means_and_target_variances[0]
            self.variances += z_means_and_target_variances[1]

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.params = shift_params(self.params)
            self.zparams = shift_params(self.zparams)
            logs = {
                "z_means": self.z_means
                / (self.target_update_frequency / self.update_to_data),
                "variance": self.variances
                / (self.target_update_frequency / self.update_to_data),
            }

            self.z_means = np.zeros(self.n_bellman_iterations)
            self.variances = np.zeros(self.n_bellman_iterations)
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
        (grad_loss, z_grad_loss), z_means_and_target_variances = jax.grad(
            self.loss_on_batch, has_aux=True, argnums=(0, 1)
        )(params, zparams, batch_samples)

        updates, optimizer_state = self.optimizer.update(
            grad_loss, optimizer_state, params
        )
        z_updates, z_optimizer_state = self.z_optimizer.update(
            z_grad_loss, z_optimizer_state, zparams
        )

        params = optax.apply_updates(params, updates)
        zparams = optax.apply_updates(zparams, z_updates)

        return (
            params,
            zparams,
            optimizer_state,
            z_optimizer_state,
            z_means_and_target_variances,
        )

    def loss_on_batch(self, params: FrozenDict, zparams: FrozenDict, samples):
        loss, z_values, variances = jax.vmap(self.loss, in_axes=(None, None, 0))(
            params, zparams, samples
        )
        return loss.mean(), (z_values.mean(), variances.mean())

    def loss(
        self,
        params: FrozenDict,
        zparams: FrozenDict,
        sample: ReplayElement,
        mu: float = 1,
    ):
        # computes the loss for a single sample

        q_values = jax.vmap(self.network.apply, in_axes=(0, None))(
            jax.tree.map(lambda x: x[1:], params), sample.state
        )[
            :, sample.action
        ]  # from 1 to n_bellman_iterations
        targets = jax.vmap(self.compute_target, in_axes=(0, None))(
            jax.tree.map(lambda x: x[:-1], params), sample
        )  # from 1 to n_bellman_iterations - 1
        TDs = targets - q_values

        z_values = jax.vmap(self.network.apply, in_axes=(0, None))(
            zparams, sample.state
        )[:, sample.action]
        z_loss = z_values * jax.lax.stop_gradient(z_values - TDs)

        targets = targets.at[0].set(
            0.0
        )  # cut off the gradient flow to the first Q-Network Q_0 by overwriting the first target with a constant.
        TD_loss = targets * jax.lax.stop_gradient(
            z_values
        ) - q_values * jax.lax.stop_gradient(TDs)

        return (
            (TD_loss + mu * z_loss).mean(),
            z_values,
            (targets**2 - targets * q_values).mean(),
        )  # TDs * jnp.square(targets - TDs); mu is weighting both loss terms

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (
            self.gamma**self.update_horizon
        ) * jnp.max(self.network.apply(params, sample.next_state))

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        # computes the best action for a single state
        idx_params = jax.random.randint(key, (), 1, self.n_bellman_iterations + 1)
        return jnp.argmax(
            self.network.apply(
                jax.tree.map(lambda param: param[idx_params], params), state
            )
        )

    def get_model(self):
        return {"params": self.params}
