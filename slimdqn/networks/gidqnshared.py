from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict, freeze
from slimdqn.networks.architectures.dqn import DQNNet

from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@partial(jax.jit, static_argnames=["n_actions", "n_bellman_iterations"])
def roll(x, n_bellman_iterations, n_actions):
    td_estimator_start_idx = n_bellman_iterations * n_actions
    return (
        x.at[..., : td_estimator_start_idx - n_actions]
        .set(x[..., n_actions:td_estimator_start_idx])
        .at[..., td_estimator_start_idx:-n_actions]
        .set(x[..., td_estimator_start_idx + n_actions :])
    )


@partial(jax.jit, static_argnames=["n_actions", "n_bellman_iterations"])
def shift_params(params, n_bellman_iterations, n_actions):
    # Each online network is updated to the following online network
    # \theta_k <- \theta_{k + 1}, i.e., params[k] <- params[k + 1]
    kernel, bias = params["params"]["Dense_final"]["kernel"], params["params"]["Dense_final"]["bias"]

    shifted_bias = roll(bias, n_bellman_iterations, n_actions)
    shifted_kernel = roll(kernel, n_bellman_iterations, n_actions)

    shifted_root_params = params.copy(
        add_or_replace={
            "params": params["params"].copy(
                add_or_replace={
                    "Dense_final": {
                        "kernel": kernel[..., :n_actions],
                        "bias": bias[:n_actions],
                    }
                }
            )
        }
    )
    shifted_params = params.copy(
        add_or_replace={
            "params": params["params"].copy(
                add_or_replace={"Dense_final": {"kernel": shifted_kernel, "bias": shifted_bias}}
            )
        }
    )
    return shifted_root_params, shifted_params


class GiDQNShared:
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
        unfreeze_first_head: bool,
        target_update_frequency: int,
        weight_decay: float,
        adam_eps: float = 1e-8,
    ):
        key_params, key_root_params = jax.random.split(key, 2)

        self.n_bellman_iterations = n_bellman_iterations  # K
        self.n_actions = n_actions
        # One Root Network Q_0
        self.root_network = DQNNet(features, architecture_type, n_actions, 1)
        # 2K Networks: Q_1 to Q_K, TD-Surrogate_1 to TD-Surrogate_K-1
        self.networks = DQNNet(features, architecture_type, n_actions, n_bellman_iterations * 2 - 1)

        self.root_params = self.root_network.init(key_root_params, jnp.zeros(observation_dim, dtype=jnp.float32))
        self.params = freeze(self.networks.init(key_params, jnp.zeros(observation_dim, dtype=jnp.float32)))

        self.optimizer = optax.adamw(
            learning_rate,
            eps=adam_eps,
            weight_decay=weight_decay,
            mask=jax.tree_util.tree_map_with_path(
                lambda path, leaf: (jnp.ones_like(leaf) if "Dense_final" in path[1].key else jnp.zeros_like(leaf)),
                self.params,
            ),
        )
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.unfreeze_first_head = unfreeze_first_head
        self.target_update_frequency = target_update_frequency
        self.cumulative_q_losses = np.zeros(self.n_bellman_iterations)
        self.cumulative_z_losses = np.zeros(self.n_bellman_iterations)
        self.cumulative_variance = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        # Update the network parameters every `update_to_data` steps
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample()

            (
                self.params,
                self.optimizer_state,
                q_losses,
                z_losses,
                variance,
            ) = self.learn_on_batch(
                self.params,
                self.root_params,
                self.optimizer_state,
                batch_samples,
            )

            self.cumulative_q_losses += q_losses
            self.cumulative_z_losses += z_losses
            self.cumulative_variance += variance

    def update_target_params(self, step: int):
        # shift the network parameters every `target_update_frequency` steps. This starts the next Bellman iteration
        if step % self.target_update_frequency == 0:
            self.root_params, self.params = shift_params(self.params, self.n_bellman_iterations, self.n_actions)

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
        root_params: FrozenDict,
        optimizer_state,
        batch_samples,
    ):
        total_grad_loss, (q_losses, z_losses, variance) = jax.grad(self.loss_on_batch, has_aux=True, argnums=(0))(
            params, root_params, batch_samples
        )

        updates, optimizer_state = self.optimizer.update(total_grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return (
            params,
            optimizer_state,
            q_losses,
            z_losses,
            variance,
        )

    def loss_on_batch(self, params: FrozenDict, root_params: FrozenDict, samples):
        # vmap to compute the loss for all samples
        total_losses, q_losses, z_losses, variances = jax.vmap(self.loss, in_axes=(None, None, 0))(
            params, root_params, samples
        )
        return total_losses.sum(axis=-1).mean(), (  # sum over networks and mean over samples
            q_losses.mean(axis=0),  # mean over samples but keep networks seperated
            z_losses.mean(axis=0),  # mean over samples but keep networks seperated
            variances.mean(),  # mean over samples and networks
        )

    def loss(
        self,
        params: FrozenDict,
        root_params: FrozenDict,
        sample: ReplayElement,
    ):
        # computes the loss for a single sample
        predictions = self.networks.apply(params, sample.state)[..., sample.action]  # K + K - 1
        q_values, z_values = predictions[: self.n_bellman_iterations], predictions[self.n_bellman_iterations :]
        z_values = jnp.concatenate([jnp.array([0]), z_values])  # K, Add any constant Value to z_values to match shapes

        next_q_root = self.root_network.apply(root_params, sample.next_state)  # Shape (1,)
        remaining_next_q_values = self.networks.apply(params, sample.next_state)[
            : self.n_bellman_iterations - 1
        ]  # K - 1
        all_q_values_next = jnp.concatenate([next_q_root[None, :], remaining_next_q_values], axis=0)  # K

        targets = self.compute_target(all_q_values_next, sample)  # K
        td_errors = targets - q_values  # K
        z_loss = z_values * jax.lax.stop_gradient(z_values - td_errors)  # K
        td_loss = targets * jax.lax.stop_gradient(z_values) - q_values * jax.lax.stop_gradient(td_errors)  # K

        return (
            td_loss + z_loss,
            jnp.square(td_errors),
            jnp.square(z_values - td_errors),
            (targets**2 - targets * q_values).mean(),
        )

    def compute_target(self, q_values_next: jnp.ndarray, sample: ReplayElement):
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            q_values_next, axis=1
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        # computes the best action for a single state
        idx_params = jax.random.randint(key, (), 0, self.n_bellman_iterations)
        return jnp.argmax(self.networks.apply(params, state)[idx_params])

    def get_model(self):
        return {"params": self.params, "root_params": self.root_params}
