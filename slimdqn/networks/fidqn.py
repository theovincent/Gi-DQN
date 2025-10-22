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


class FiDQN:
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
        self.n_bellman_iterations = n_bellman_iterations
        self.network = DQNNet(features, architecture_type, n_actions)

        # initialize K+1 networks
        self.params = jax.vmap(self.network.init, in_axes=(0, None))(
            jax.random.split(key, self.n_bellman_iterations + 1), jnp.zeros(observation_dim, dtype=jnp.float32)
        )

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.cumulative_losses = np.zeros(self.n_bellman_iterations)
        self.cumulative_variance = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        # update online network parameters every update_to_data steps
        for _ in range(int(self.update_to_data)):
            batch_samples = replay_buffer.sample()

            (self.params, self.optimizer_state, losses, variance) = self.learn_on_batch(
                self.params, self.optimizer_state, batch_samples
            )

            self.cumulative_losses += losses
            self.cumulative_variance += variance

    def update_target_params(self, step: int):
        # update target network parameters every target_update_frequency steps. This starts the next Bellman iteration
        if step % self.target_update_frequency == 0:
            self.params = shift_params(self.params)

            logs = {
                "loss": np.mean(self.cumulative_losses) / (self.target_update_frequency * self.update_to_data),
                "variance": self.cumulative_variance / (self.target_update_frequency * self.update_to_data),
            }
            for idx_network in range(0, min(5, self.n_bellman_iterations)):
                logs[f"networks/{idx_network}_loss"] = self.cumulative_losses[idx_network] / (
                    self.target_update_frequency * self.update_to_data
                )

            self.cumulative_losses = np.zeros(self.n_bellman_iterations)
            self.cumulative_variance = 0
            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples):
        grad_loss, (losses, variance) = jax.grad(self.loss_on_batch, has_aux=True)(params, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return (params, optimizer_state, losses, variance)

    def loss_on_batch(self, params: FrozenDict, samples):
        losses, mse_losses, variances = jax.vmap(self.loss, in_axes=(None, 0))(params, samples)

        return losses.sum(axis=-1).mean(), (mse_losses.mean(axis=0), variances.mean())

    def loss(self, params: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample

        # use networks 1 to K to compute the Q-values
        q_values = jax.vmap(self.network.apply, in_axes=(0, None))(jax.tree.map(lambda x: x[1:], params), sample.state)[
            :, sample.action
        ]
        # use networks 0 to K-1 to compute the targets
        targets = jax.vmap(self.compute_target, in_axes=(0, None))(jax.tree.map(lambda x: x[:-1], params), sample)
        td_errors = jax.lax.stop_gradient(targets - q_values)

        # cut off the gradient flow of the first Q-Network Q_0 by overwriting the first target with a constant
        targets = targets.at[0].set(0.0)
        td_loss = targets * td_errors - q_values * td_errors

        return (td_loss, jnp.square(td_errors), (targets**2 - targets * q_values).mean())

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
