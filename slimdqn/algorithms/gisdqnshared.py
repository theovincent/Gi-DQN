from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from slimdqn.algorithms.architectures.dqn import DQNNet

from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@partial(jax.jit, static_argnames="n_actions")
def shift_params(params, n_actions):
    q_heads = jax.tree.map(lambda p: p.at[..., :-n_actions].set(p[..., n_actions:]), params["params"]["q_heads"])
    h_heads = jax.tree.map(lambda p: p.at[..., :-n_actions].set(p[..., n_actions:]), params["params"]["h_heads"])

    return optax.tree_utils.tree_set(params, q_heads=q_heads, h_heads=h_heads)


class GiSDQNShared:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_bellman_iterations: int,
        features: list,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_period: int,
        weight_decay: float,
        unfreeze_first_head: bool,
        adam_eps: float = 1e-8,
    ):
        self.n_bellman_iterations = n_bellman_iterations
        self.unfreeze_first_head = unfreeze_first_head
        self.n_actions = n_actions
        # 2K Networks: Q_0, Q_1 to Q_K, TD-Surrogate_1 to TD-Surrogate_K-1
        self.networks = DQNNet(
            features,
            n_actions,
            n_heads=1 + self.n_bellman_iterations,
            n_h_heads=self.n_bellman_iterations - 1 + int(unfreeze_first_head),
        )

        # initialize 1 network with K+1 q-heads and K-1 OR K h-heads
        self.params = self.networks.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adamw(
            learning_rate,
            eps=adam_eps,
            weight_decay=weight_decay,
            mask=jax.tree_util.tree_map_with_path(lambda path, _: "h_heads" in path[1].key, self.params),
        )
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_period = target_update_period
        self.cumulative_q_losses = np.zeros(self.n_bellman_iterations)
        self.cumulative_h_losses = np.zeros(self.n_bellman_iterations - 1 + int(unfreeze_first_head))

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        for _ in range(int(max(self.update_to_data, 1))):
            # if update_to_data < 1, only perform one update if step = 0 [1 / self.update_to_data]
            if self.update_to_data < 1 and step % (1 / self.update_to_data) != 0:
                return None
            batch_samples, (sample_keys, importance_weights) = replay_buffer.sample()

            self.params, self.optimizer_state, per_sample_q_losses, per_sample_h_losses = self.learn_on_batch(
                self.params, self.optimizer_state, batch_samples, importance_weights
            )

            replay_buffer.update(sample_keys, per_sample_q_losses.mean(axis=1) + per_sample_h_losses.mean(axis=1))

            self.cumulative_q_losses += per_sample_q_losses.mean(axis=0)
            self.cumulative_h_losses += per_sample_h_losses.mean(axis=0)

    def update_target_params(self, step: int):
        if step % self.target_update_period == 0:
            self.params = shift_params(self.params, self.n_actions)

            self.logs = {
                "n_training_steps": step,
                "loss": np.mean(self.cumulative_q_losses) / (self.target_update_period * self.update_to_data),
                "h_loss": np.mean(self.cumulative_h_losses) / (self.target_update_period * self.update_to_data),
            }
            # for idx_network in range(0, min(5, self.n_bellman_iterations)):
            #     self.logs[f"networks/{idx_network}_loss"] = self.cumulative_q_losses[idx_network] / (
            #         self.target_update_period * self.update_to_data
            #     )
            # for idx_network in range(min(5, self.n_bellman_iterations - 1 + int(self.unfreeze_first_head))):
            #     self.logs[f"h_networks/{idx_network}_loss"] = self.cumulative_h_losses[idx_network] / (
            #         self.target_update_period * self.update_to_data
            #     )

            self.cumulative_q_losses = np.zeros(self.n_bellman_iterations)
            self.cumulative_h_losses = np.zeros(self.n_bellman_iterations - 1 + int(self.unfreeze_first_head))

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples, importance_weights):
        grad_loss, (per_sample_q_losses, per_sample_h_losses) = jax.grad(self.loss_on_batch, has_aux=True)(
            params, batch_samples, importance_weights
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, per_sample_q_losses, per_sample_h_losses

    def loss_on_batch(self, params: FrozenDict, samples, importance_weights):
        total_losses, q_losses, h_losses = jax.vmap(self.loss, in_axes=(None, 0, 0))(
            params, samples, importance_weights
        )

        return total_losses.mean(axis=0).sum(), (q_losses, h_losses)

    def loss(self, params: FrozenDict, sample: ReplayElement, importance_weight):
        q_outputs, h_outputs = self.networks.apply(params, sample.state)
        q_values, h_values = q_outputs[1:, sample.action], h_outputs[:, sample.action]

        next_q_values = self.networks.apply(params, sample.next_state)[0][:-1]
        targets = self.compute_target(next_q_values, sample)

        td_errors = targets - q_values

        h_loss = h_values * jax.lax.stop_gradient(h_values - td_errors[1 - int(self.unfreeze_first_head) :])
        target_loss = targets[1 - int(self.unfreeze_first_head) :] * jax.lax.stop_gradient(h_values)

        if not self.unfreeze_first_head:
            h_loss = jnp.append(jnp.zeros(1), h_loss)
            target_loss = jnp.append(jnp.zeros(1), target_loss)

        td_loss = target_loss - q_values * jax.lax.stop_gradient(td_errors)

        return (
            importance_weight * (td_loss + h_loss),
            jnp.square(td_errors),
            jnp.square(h_values - td_errors[1 - int(self.unfreeze_first_head) :]),
        )

    def compute_target(self, next_q_values, sample: ReplayElement):
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            next_q_values, axis=-1
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        return jnp.argmax(self.networks.apply(params, state)[0][1:].mean(axis=0))

    def get_model(self):
        return {"params": self.params}
