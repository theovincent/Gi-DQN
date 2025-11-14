from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@partial(jax.jit, static_argnames="n_actions")
def shift_params(params, n_actions):
    kernel = params["params"]["Dense_final"]["kernel"]
    bias = params["params"]["Dense_final"]["bias"]
    root_params = optax.tree_utils.tree_set(
        params, Dense_final={"kernel": kernel[:, :n_actions], "bias": bias[:n_actions]}
    )
    params["params"]["Dense_final"]["kernel"] = kernel.at[:, :-n_actions].set(kernel[:, n_actions:])
    params["params"]["Dense_final"]["bias"] = bias.at[:-n_actions].set(bias[n_actions:])

    return root_params, params


class FiDQNShared:
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
        target_update_frequency: int,
        adam_eps: float = 1e-8,
    ):
        key, key_params = jax.random.split(key)
        self.n_actions = n_actions
        self.n_bellman_iterations = n_bellman_iterations

        self.root_network = DQNNet(features, architecture_type, layer_norm, n_actions)
        self.networks = DQNNet(features, architecture_type, layer_norm, n_actions, self.n_bellman_iterations)

        # initialize 1 root network and 1 network with K heads
        self.root_params = self.root_network.init(key_params, jnp.zeros(observation_dim, dtype=jnp.float32))
        self.params = self.networks.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

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
            batch_samples, _ = replay_buffer.sample()

            (self.params, self.optimizer_state, losses, variance) = self.learn_on_batch(
                self.params, self.root_params, self.optimizer_state, batch_samples
            )

            self.cumulative_losses += losses
            self.cumulative_variance += variance

    def update_target_params(self, step: int):
        # update target network parameters every target_update_frequency steps. This starts the next Bellman iteration
        if step % self.target_update_frequency == 0:
            self.root_params, self.params = shift_params(self.params, self.n_actions)

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
    def learn_on_batch(self, params: FrozenDict, root_params: FrozenDict, optimizer_state, batch_samples):
        grad_loss, (losses, variance) = jax.grad(self.loss_on_batch, has_aux=True)(params, root_params, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return (params, optimizer_state, losses, variance)

    def loss_on_batch(self, params: FrozenDict, root_params: FrozenDict, samples):
        losses, mse_losses, variances = jax.vmap(self.loss, in_axes=(None, None, 0))(params, root_params, samples)

        return losses.sum(axis=-1).mean(), (mse_losses.mean(axis=0), variances.mean())

    def loss(self, params: FrozenDict, root_params: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        q_values = self.networks.apply(params, sample.state)[:, sample.action]
        next_q_root = self.root_network.apply(root_params, sample.next_state)
        remaining_next_q_values = self.networks.apply(params, sample.next_state)[:-1]
        all_next_q_values = jnp.concatenate([next_q_root[None, :], remaining_next_q_values], axis=0)
        targets = self.compute_target(all_next_q_values, sample)
        td_errors = jax.lax.stop_gradient(targets - q_values)
        td_loss = targets * td_errors - q_values * td_errors

        return (td_loss, jnp.square(td_errors), (targets**2 - targets * q_values).mean())

    def compute_target(self, next_q, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(next_q, axis=-1)

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        # computes the best action for a single state
        idx_params = jax.random.randint(key, (), 0, self.n_bellman_iterations)
        return jnp.argmax(self.networks.apply(params, state)[idx_params])

    def get_model(self):
        return {"params": self.params, "root_params": self.root_params}
