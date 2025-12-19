from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict, freeze
from slimdqn.networks.architectures.dqn import DQNNet

from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@partial(jax.jit)
def shift_params(params):
    q_heads = jax.tree.map(lambda p: p.at[:-1].set(p[1:]), params["params"]["q_heads"])
    h_heads = jax.tree.map(lambda p: p.at[:-1].set(p[1:]), params["params"]["h_heads"])
    first_q_head = jax.tree.map(lambda p: p[0], params["params"]["q_heads"])
    root_params = optax.tree_utils.tree_set(params, q_heads=first_q_head, h_heads=None)
    params = optax.tree_utils.tree_set(params, q_heads=q_heads, h_heads=h_heads)

    return root_params, params


class GiDQNShared:
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
        unfreeze_first_head: bool,
        target_update_frequency: int,
        weight_decay: float,
        mu: float,
        adam_eps: float = 1e-8,
    ):
        key_params, key_root_params = jax.random.split(key, 2)

        self.n_bellman_iterations = n_bellman_iterations
        self.n_actions = n_actions
        # One Root Network Q_0
        self.root_network = DQNNet(features, architecture_type, layer_norm, n_actions)
        # 2K Networks: Q_1 to Q_K, TD-Surrogate_1 to TD-Surrogate_K-1
        self.networks = DQNNet(
            features, architecture_type, layer_norm, n_actions, n_bellman_iterations, n_bellman_iterations - 1
        )

        self.root_params = self.root_network.init(key_root_params, jnp.zeros(observation_dim, dtype=jnp.float32))
        self.params = self.networks.init(key_params, jnp.zeros(observation_dim, dtype=jnp.float32))
        self.optimizer = optax.adamw(
            learning_rate,
            eps=adam_eps,
            weight_decay=weight_decay,
            mask=jax.tree_util.tree_map_with_path(
                lambda path, leaf: True if "h_heads" in path[1].key else False, self.params
            ),
        )
        self.optimizer_state = self.optimizer.init(self.params)

        self.mu = mu
        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.unfreeze_first_head = unfreeze_first_head
        self.target_update_frequency = target_update_frequency
        self.cumulative_q_losses = np.zeros(self.n_bellman_iterations)
        self.cumulative_h_losses = np.zeros(self.n_bellman_iterations - 1)
        self.cumulative_variance = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        # Update the network parameters 'update_to_data' times every step
        for _ in range(int(self.update_to_data)):
            batch_samples, _ = replay_buffer.sample()

            (
                self.params,
                self.optimizer_state,
                q_losses,
                h_losses,
                variance,
            ) = self.learn_on_batch(
                self.params,
                self.root_params,
                self.optimizer_state,
                batch_samples,
            )

            self.cumulative_q_losses += q_losses
            self.cumulative_h_losses += h_losses
            self.cumulative_variance += variance

    def update_target_params(self, step: int):
        # shift the network parameters every `target_update_frequency` steps. This starts the next Bellman iteration
        if step % self.target_update_frequency == 0:
            self.root_params, self.params = shift_params(self.params)

            logs = {
                "loss": np.mean(self.cumulative_q_losses) / (self.target_update_frequency * self.update_to_data),
                "variance": np.mean(self.cumulative_variance) / (self.target_update_frequency * self.update_to_data),
                "h_loss": np.mean(self.cumulative_h_losses) / (self.target_update_frequency * self.update_to_data),
            }
            for idx_network in range(0, min(5, self.n_bellman_iterations)):
                logs[f"networks/{idx_network}_loss"] = self.cumulative_q_losses[idx_network] / (
                    self.target_update_frequency * self.update_to_data
                )
            for idx_network in range(min(5, self.n_bellman_iterations - 1)):
                logs[f"h_networks/{idx_network}_loss"] = self.cumulative_h_losses[idx_network] / (
                    self.target_update_frequency * self.update_to_data
                )

            self.cumulative_q_losses = np.zeros(self.n_bellman_iterations)
            self.cumulative_h_losses = np.zeros(self.n_bellman_iterations - 1)
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
        total_grad_loss, (q_losses, h_losses, variance) = jax.grad(self.loss_on_batch, has_aux=True, argnums=(0))(
            params, root_params, batch_samples
        )

        updates, optimizer_state = self.optimizer.update(total_grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return (params, optimizer_state, q_losses, h_losses, variance)

    def loss_on_batch(self, params: FrozenDict, root_params: FrozenDict, samples):
        # vmap to compute the loss for all samples
        total_losses, q_losses, h_losses, variances = jax.vmap(self.loss, in_axes=(None, None, 0))(
            params, root_params, samples
        )
        return total_losses.sum(axis=-1).mean(), (  # sum over networks and mean over samples
            q_losses.mean(axis=0),  # mean over samples but keep networks seperated
            h_losses.mean(axis=0),  # mean over samples but keep networks seperated
            variances.mean(),  # mean over samples and networks
        )

    def loss(
        self,
        params: FrozenDict,
        root_params: FrozenDict,
        sample: ReplayElement,
    ):
        # computes the loss for a single sample
        q_outputs, h_outputs = self.networks.apply(params, sample.state)
        q_values, h_values = q_outputs[:, sample.action], h_outputs[:, sample.action]
        # Add a constant value to h_values to match shapes
        all_h_values = jnp.concatenate([jnp.array([0]), h_values])

        next_q_root = self.root_network.apply(root_params, sample.next_state)
        remaining_next_q_values = self.networks.apply(params, sample.next_state)[0][:-1]
        all_q_values_next = jnp.concatenate([next_q_root[None, :], remaining_next_q_values], axis=0)

        targets = self.compute_target(all_q_values_next, sample)
        td_errors = targets - q_values
        h_loss = all_h_values * jax.lax.stop_gradient(all_h_values - td_errors)
        td_loss = targets * jax.lax.stop_gradient(all_h_values) - q_values * jax.lax.stop_gradient(td_errors)

        return (
            td_loss + self.mu * h_loss,
            jnp.square(td_errors),
            jnp.square(all_h_values - td_errors)[1:],
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
        return jnp.argmax(self.networks.apply(params, state)[0][idx_params])

    def get_model(self):
        return {"params": self.params, "root_params": self.root_params}
