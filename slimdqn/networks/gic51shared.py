from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict, freeze
from slimdqn.networks.architectures.dqn import DQNNet

from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@partial(jax.jit, static_argnames=["n_actions", "n_bins"])
def shift_params(params, n_actions, n_bins):
    kernel = params["params"]["Dense_final"]["kernel"]
    bias = params["params"]["Dense_final"]["bias"]
    root_params = optax.tree_utils.tree_set(
        params,
        Dense_final={"kernel": kernel[:, : n_actions * n_bins], "bias": bias[: n_actions * n_bins]},
        Dense_final_h=None,
    )
    h_kernel = params["params"]["Dense_final_h"]["kernel"]
    h_bias = params["params"]["Dense_final_h"]["bias"]

    params["params"]["Dense_final"]["kernel"] = kernel.at[:, : -n_actions * n_bins].set(kernel[:, n_actions * n_bins :])
    params["params"]["Dense_final"]["bias"] = bias.at[: -n_actions * n_bins].set(bias[n_actions * n_bins :])
    params["params"]["Dense_final_h"]["kernel"] = h_kernel.at[:, : -n_actions * n_bins].set(
        h_kernel[:, n_actions * n_bins :]
    )
    params["params"]["Dense_final_h"]["bias"] = h_bias.at[: -n_actions * n_bins].set(h_bias[n_actions * n_bins :])

    return root_params, params


class GiC51Shared:
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
        weight_decay: float,
        min_value: float,
        max_value: float,
        adam_eps: float = 1e-8,
    ):
        key_params, key_root_params = jax.random.split(key, 2)

        self.n_bellman_iterations = n_bellman_iterations
        self.n_actions = n_actions
        # One Root Network Q_0
        self.root_network = DQNNet(features, architecture_type, n_actions, n_bins=51)
        # 2K Networks: Q_1 to Q_K, TD-Surrogate_1 to TD-Surrogate_K-1
        self.networks = DQNNet(
            features, architecture_type, n_actions, n_bellman_iterations, n_bellman_iterations - 1, n_bins=51
        )

        self.root_params = self.root_network.init(key_root_params, jnp.zeros(observation_dim, dtype=jnp.float32))
        self.params = self.networks.init(key_params, jnp.zeros(observation_dim, dtype=jnp.float32))
        self.optimizer = optax.adamw(
            learning_rate,
            eps=adam_eps,
            weight_decay=weight_decay,
            mask=jax.tree_util.tree_map_with_path(
                lambda path, leaf: True if "Dense_final_h" in path[1].key else False, self.params
            ),
        )
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.n_bins = 51
        self.support = jnp.linspace(min_value, max_value, 51, dtype=jnp.float32)
        self.bin_size = (max_value - min_value) / 50
        self.min_value = min_value
        self.max_value = max_value
        self.clip_target = lambda target: jnp.clip(target, min_value, max_value)
        self.cumulative_q_losses = np.zeros(self.n_bellman_iterations)
        self.cumulative_h_losses = np.zeros(self.n_bellman_iterations - 1)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        # Update the network parameters 'update_to_data' times every step
        for _ in range(int(self.update_to_data)):
            batch_samples, _ = replay_buffer.sample()

            (
                self.params,
                self.optimizer_state,
                q_losses,
                h_losses,
            ) = self.learn_on_batch(
                self.params,
                self.root_params,
                self.optimizer_state,
                batch_samples,
            )

            self.cumulative_q_losses += q_losses
            self.cumulative_h_losses += h_losses

    def update_target_params(self, step: int):
        # shift the network parameters every `target_update_frequency` steps. This starts the next Bellman iteration
        if step % self.target_update_frequency == 0:
            self.root_params, self.params = shift_params(self.params, self.n_actions, self.n_bins)

            logs = {
                "loss": np.mean(self.cumulative_q_losses) / (self.target_update_frequency * self.update_to_data),
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
        total_grad_loss, (q_losses, h_losses) = jax.grad(self.loss_on_batch, has_aux=True, argnums=(0))(
            params, root_params, batch_samples
        )
        updates, optimizer_state = self.optimizer.update(total_grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return (params, optimizer_state, q_losses, h_losses)

    def loss_on_batch(self, params: FrozenDict, root_params: FrozenDict, samples):
        # vmap to compute the loss for all samples
        total_losses, q_losses, h_losses = jax.vmap(self.loss, in_axes=(None, None, 0))(params, root_params, samples)
        return total_losses.sum(axis=-1).mean(), (  # sum over networks and mean over samples
            q_losses.mean(axis=0),  # mean over samples but keep networks seperated
            h_losses.mean(axis=0),  # mean over samples but keep networks seperated # mean over samples and networks
        )

    def loss(
        self,
        params: FrozenDict,
        root_params: FrozenDict,
        sample: ReplayElement,
    ):
        # computes the loss for a single sample
        q_logits, h_logits = self.networks.apply(params, sample.state)
        q_logits_a, h_logits_a = q_logits[:, sample.action], h_logits[:, sample.action]
        q_value_probs, h_value_probs = jax.nn.softmax(q_logits_a, axis=-1), jax.nn.softmax(h_logits_a, axis=-1)
        best_action = self.best_action(params, sample.next_state)

        first_target_next_q_probs = jax.nn.softmax(self.root_network.apply(root_params, sample.next_state), axis=-1)
        remaining_next_q_probs = jax.nn.softmax(self.networks.apply(params, sample.next_state)[0][:-1], axis=-1)
        next_q_values = jnp.concatenate([first_target_next_q_probs[None, :], remaining_next_q_probs], axis=0)
        projected_targets = self.compute_target(next_q_values[:, best_action], sample)

        baseline = jax.scipy.special.logsumexp(a=h_logits_a, b=jax.lax.stop_gradient(q_value_probs[1:]), axis=-1)[
            :, None
        ]
        centered_h_logits = h_logits_a - baseline
        target_loss = jnp.sum(jax.lax.stop_gradient(centered_h_logits) * projected_targets[1:], axis=-1)
        target_loss = jnp.append(jnp.zeros(1), target_loss)
        ce_loss = optax.softmax_cross_entropy(q_logits_a, jax.lax.stop_gradient(projected_targets), axis=-1)

        kl = target_loss + ce_loss
        h_loss = -jnp.sum(centered_h_logits * jax.lax.stop_gradient(projected_targets[1:]), axis=-1)
        h_loss = jnp.append(jnp.zeros(1), h_loss)

        return kl + h_loss, kl, h_loss[1:]

    def compute_target(self, next_q, sample: ReplayElement):
        # computes the target value for single sample
        target_atoms = sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * self.support
        clipped_target_atoms = self.clip_target(target_atoms)
        b = ((clipped_target_atoms - self.min_value) / self.bin_size)[None, :]
        l = jnp.clip(jnp.floor(b).astype(jnp.int32), 0, 50)
        u = jnp.clip(jnp.ceil(b).astype(jnp.int32), 0, 50)

        m = jnp.zeros((self.n_bellman_iterations, self.support.shape[0]))
        rows = jnp.arange(self.n_bellman_iterations)[:, None]
        m = m.at[rows, l].add(next_q * (u.astype(b.dtype) - b))
        m = m.at[rows, u].add(next_q * (b - l.astype(b.dtype)))
        m = m.at[rows, l].add(next_q * (l == u))

        return m

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key=None):
        # computes the best action for a single state
        return jnp.argmax((jax.nn.softmax(self.networks.apply(params, state)[0], axis=-1) @ self.support).mean(axis=0))

    def get_model(self):
        return {"params": self.params, "root_params": self.root_params}
