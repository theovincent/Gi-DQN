from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict, freeze

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@partial(jax.jit, static_argnames="n_actions")
def copy_online_params_to_targets(params_online, n_actions):
    first_kernel, remaining_kernels = jnp.split(params_online["params"]["Dense_final"]["kernel"], [n_actions], axis=1)
    first_bias, remaining_biases = jnp.split(params_online["params"]["Dense_final"]["bias"], [n_actions], axis=0)

    first_params = params_online.copy(
        add_or_replace={
            "params": params_online["params"].copy(
                add_or_replace={"Dense_final": {"kernel": first_kernel, "bias": first_bias}}
            )
        }
    )
    remaining_params = params_online.copy(
        add_or_replace={
            "params": params_online["params"].copy(
                add_or_replace={"Dense_final": {"kernel": remaining_kernels, "bias": remaining_biases}}
            )
        }
    )

    return first_params, remaining_params


@partial(jax.jit, static_argnames="n_actions")
def shift_params(params, n_actions):
    # Each online network is updated to the following online network
    # \theta_k <- \theta_{k + 1}, i.e., params[k] <- params[k + 1]
    bias, kernel = params["params"]["Dense_final"]["bias"], params["params"]["Dense_final"]["kernel"]

    shifted_bias = bias.at[:-n_actions].set(bias[n_actions:])
    shifted_kernel = kernel.at[..., :-n_actions].set(kernel[..., n_actions:])
    shifted_params = params.copy(
        add_or_replace={
            "params": params["params"].copy(
                add_or_replace={"Dense_final": {"kernel": shifted_kernel, "bias": shifted_bias}}
            )
        }
    )

    return shifted_params


@partial(jax.jit, static_argnames="n_actions")
def sync_target_params(params, n_actions):
    # Each target network is synchronized to the online network it represents
    # \bar{\theta}_k <- \theta_k, i.e., target_params[k] <- params[k-1]
    kernel, bias = (
        params["params"]["Dense_final"]["kernel"][..., :-n_actions],
        params["params"]["Dense_final"]["bias"][:-n_actions],
    )

    return params.copy(
        add_or_replace={
            "params": params["params"].copy(add_or_replace={"Dense_final": {"kernel": kernel, "bias": bias}})
        }
    )


class iDQNShared:
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
        target_sync_frequency: int,
        adam_eps: float = 1e-8,
    ):
        key, key_params = jax.random.split(key)
        self.n_actions = n_actions
        self.n_bellman_iterations = n_bellman_iterations
        self.online_networks = DQNNet(features, architecture_type, n_actions, self.n_bellman_iterations)
        self.first_target_network = DQNNet(features, architecture_type, n_actions)
        self.remaining_target_networks = DQNNet(features, architecture_type, n_actions, self.n_bellman_iterations - 1)

        # initialize K networks
        self.params = freeze(self.online_networks.init(key_params, jnp.zeros(observation_dim, dtype=jnp.float32)))
        self.first_target_params, self.remaining_target_params = copy_online_params_to_targets(
            self.params, n_actions
        )  # initialize target networks

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.target_sync_frequency = target_sync_frequency
        self.cumulative_losses = np.zeros(self.n_bellman_iterations)
        self.cumulative_variance = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        # update online network parameters every update_to_data steps
        for _ in range(int(self.update_to_data)):
            batch_samples = replay_buffer.sample()

            (self.params, self.optimizer_state, losses, variance) = self.learn_on_batch(
                self.params, self.first_target_params, self.remaining_target_params, self.optimizer_state, batch_samples
            )

            self.cumulative_losses += losses
            self.cumulative_variance += variance

    def update_target_params(self, step: int):
        # update target network parameters every target_update_frequency steps. This starts the next Bellman iteration
        if step % self.target_update_frequency == 0:
            # Each target network is updated to its respective online network
            # \bar{\theta}_k <- \theta_{k + 1}, i.e., target_params[k] <- params[k]
            self.first_target_params, self.remaining_target_params = copy_online_params_to_targets(
                self.params, self.n_actions
            )
            # Window shift
            self.params = shift_params(self.params, self.n_actions)

            logs = {
                "loss": np.mean(self.cumulative_losses) / (self.target_update_frequency / self.update_to_data),
                "variance": np.mean(self.cumulative_variance) / (self.target_update_frequency / self.update_to_data),
            }
            for idx_network in range(0, min(5, self.n_bellman_iterations)):
                logs[f"networks/{idx_network}_loss"] = self.cumulative_losses[idx_network] / (
                    self.target_update_frequency / self.update_to_data
                )

            self.cumulative_losses = np.zeros(self.n_bellman_iterations)
            self.cumulative_variance = 0
            return True, logs
        # sync target network parameters to previous online network every target_sync_frequency steps
        if step % self.target_sync_frequency == 0:
            self.remaining_target_params = sync_target_params(self.params, self.n_actions)
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        first_target_params: FrozenDict,
        remaining_target_params: FrozenDict,
        optimizer_state,
        batch_samples,
    ):
        grad_loss, (losses, variance) = jax.grad(self.loss_on_batch, has_aux=True)(
            params, first_target_params, remaining_target_params, batch_samples
        )

        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)

        params = optax.apply_updates(params, updates)

        return (params, optimizer_state, losses, variance)

    def loss_on_batch(
        self, params: FrozenDict, first_target_params: FrozenDict, remaining_target_params: FrozenDict, samples
    ):
        losses, variances = jax.vmap(self.loss, in_axes=(None, None, None, 0))(
            params, first_target_params, remaining_target_params, samples
        )

        return losses.sum(axis=-1).mean(), (
            losses.mean(axis=0),  # mean over the samples but keep networks separated
            variances.mean(),
        )

    def loss(
        self,
        params: FrozenDict,
        first_target_params: FrozenDict,
        remaining_target_params: FrozenDict,
        sample: ReplayElement,
    ):
        # computes the loss for a single sample
        q_values = self.online_networks.apply(params, sample.state)[..., sample.action]
        next_q_value_first_target = self.first_target_network.apply(first_target_params, sample.next_state)
        next_q_values_remaining_targets = self.remaining_target_networks.apply(
            remaining_target_params, sample.next_state
        )
        next_q_values = jnp.concatenate([next_q_value_first_target[None, :], next_q_values_remaining_targets], axis=0)
        targets = self.compute_target(next_q_values, sample)
        td_errors = targets - q_values

        return (jnp.square(td_errors), (targets**2 - targets * q_values).mean())

    def compute_target(self, next_q_values, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            next_q_values, axis=-1
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        # computes the best action for a single state
        idx_params = jax.random.randint(key, (), 1, self.n_bellman_iterations + 1)
        return jnp.argmax(self.online_networks.apply(params, state)[idx_params])

    def get_model(self):
        return {"params": self.params}
