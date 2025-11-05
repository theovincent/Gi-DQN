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


class HLGiDQNShared:
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
        mu: float,
        n_bins: int,
        min_value: float,
        max_value: float,
        sigma: float,
        adam_eps: float = 1e-8,
    ):
        key_params, key_root_params = jax.random.split(key, 2)

        self.n_bellman_iterations = n_bellman_iterations
        self.n_actions = n_actions
        # One Root Network Q_0
        self.root_network = DQNNet(features, architecture_type, n_actions, n_bins=n_bins)
        # 2K Networks: Q_1 to Q_K, TD-Surrogate_1 to TD-Surrogate_K-1
        self.networks = DQNNet(
            features, architecture_type, n_actions, n_bellman_iterations, n_bellman_iterations - 1, n_bins=n_bins
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

        self.mu = mu
        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.unfreeze_first_head = unfreeze_first_head
        self.target_update_frequency = target_update_frequency
        self.n_bins = n_bins
        self.sigma = sigma
        self.support = jnp.linspace(min_value, max_value, self.n_bins + 1, dtype=jnp.float32)
        self.bin_centers = (self.support[:-1] + self.support[1:]) / 2
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
        q_logits_a, h_logits_a = q_logits[:, sample.action, :], h_logits[:, sample.action, :]
        h_logits_a = jnp.concatenate([jnp.zeros((1, self.n_bins)), h_logits_a], axis=0)
        q_value_probs = jax.nn.softmax(q_logits_a, axis=-1)

        first_target_next_q = self.root_network.apply(root_params, sample.next_state)
        remaining_next_q = self.networks.apply(params, sample.next_state)[0][:-1]
        next_q_values = (
            jax.nn.softmax(jnp.concatenate([first_target_next_q[None, :], remaining_next_q], axis=0), axis=-1)
            @ self.bin_centers
        )
        targets = self.compute_target(next_q_values, sample)
        projected_targets = self.project_target(targets)
        kl = jnp.sum(jax.lax.stop_gradient(h_logits_a) * projected_targets, axis=-1) + optax.softmax_cross_entropy(
            q_logits_a, jax.lax.stop_gradient(projected_targets), axis=-1
        )
        h_loss = -jnp.sum(h_logits_a * jax.lax.stop_gradient(projected_targets), axis=-1) + jax.scipy.special.logsumexp(
            a=h_logits_a, b=jax.lax.stop_gradient(q_value_probs), axis=-1
        )

        return kl + self.mu * h_loss, kl, h_loss[1:]

    def compute_target(self, q_values_next: jnp.ndarray, sample: ReplayElement):
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            q_values_next, axis=-1
        )

    def project_target(self, target):
        erf_support = jax.scipy.special.erf(
            (self.support - self.clip_target(target)[:, None]) / (jnp.sqrt(2) * self.sigma)
        )
        return (erf_support[:, 1:] - erf_support[:, :-1]) / (erf_support[:, -1] - erf_support[:, 0] + 1e-9)[:, None]

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        # computes the best action for a single state
        idx_params = jax.random.randint(key, (), 0, self.n_bellman_iterations)
        return jnp.argmax(jax.nn.softmax(self.networks.apply(params, state)[0][idx_params], axis=-1) @ self.bin_centers)

    def get_model(self):
        return {"params": self.params, "root_params": self.root_params}
