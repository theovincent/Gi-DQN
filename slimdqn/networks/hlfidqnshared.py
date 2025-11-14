from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@partial(jax.jit, static_argnames=["n_actions", "n_bins"])
def shift_params(params, n_actions, n_bins):
    kernel = params["params"]["Dense_final"]["kernel"]
    bias = params["params"]["Dense_final"]["bias"]
    root_params = optax.tree_utils.tree_set(
        params, Dense_final={"kernel": kernel[:, : n_actions * n_bins], "bias": bias[: n_actions * n_bins]}
    )
    params["params"]["Dense_final"]["kernel"] = kernel.at[:, : -n_actions * n_bins].set(kernel[:, n_actions * n_bins :])
    params["params"]["Dense_final"]["bias"] = bias.at[: -n_actions * n_bins].set(bias[n_actions * n_bins :])

    return root_params, params


class HLFiDQNShared:
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
        n_bins: int,
        min_value: float,
        max_value: float,
        sigma: float,
        adam_eps: float = 1e-8,
    ):
        key, key_params = jax.random.split(key)
        self.n_actions = n_actions
        self.n_bellman_iterations = n_bellman_iterations

        self.root_network = DQNNet(features, architecture_type, layer_norm, n_actions, n_bins=n_bins)
        self.networks = DQNNet(features, architecture_type, layer_norm, n_actions, self.n_bellman_iterations, n_bins=n_bins)

        # initialize 1 root network and 1 network with K heads
        self.root_params = self.root_network.init(key_params, jnp.zeros(observation_dim, dtype=jnp.float32))
        self.params = self.networks.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.n_bins = n_bins
        self.sigma = sigma
        self.support = jnp.linspace(min_value, max_value, self.n_bins + 1, dtype=jnp.float32)
        self.bin_centers = (self.support[:-1] + self.support[1:]) / 2
        self.clip_target = lambda target: jnp.clip(target, min_value, max_value)
        self.cumulative_losses = np.zeros(self.n_bellman_iterations)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        # update online network parameters every update_to_data steps
        for _ in range(int(self.update_to_data)):
            batch_samples, _ = replay_buffer.sample()

            (self.params, self.optimizer_state, losses) = self.learn_on_batch(
                self.params, self.root_params, self.optimizer_state, batch_samples
            )

            self.cumulative_losses += losses

    def update_target_params(self, step: int):
        # update target network parameters every target_update_frequency steps. This starts the next Bellman iteration
        if step % self.target_update_frequency == 0:
            self.root_params, self.params = shift_params(self.params, self.n_actions, self.n_bins)

            logs = {
                "loss": np.mean(self.cumulative_losses) / (self.target_update_frequency * self.update_to_data),
            }
            for idx_network in range(0, min(5, self.n_bellman_iterations)):
                logs[f"networks/{idx_network}_loss"] = self.cumulative_losses[idx_network] / (
                    self.target_update_frequency * self.update_to_data
                )

            self.cumulative_losses = np.zeros(self.n_bellman_iterations)
            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, root_params: FrozenDict, optimizer_state, batch_samples):
        grad_loss, losses = jax.grad(self.loss_on_batch, has_aux=True)(params, root_params, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return (params, optimizer_state, losses)

    def loss_on_batch(self, params: FrozenDict, root_params: FrozenDict, samples):
        losses = jax.vmap(self.loss, in_axes=(None, None, 0))(params, root_params, samples)

        return losses.sum(axis=-1).mean(), losses.mean(axis=0)

    def loss(self, params: FrozenDict, root_params: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        # computes the loss for a single sample
        q_logits = self.networks.apply(params, sample.state)[:, sample.action]

        first_target_next_q = self.root_network.apply(root_params, sample.next_state)
        remaining_next_q = self.networks.apply(params, sample.next_state)[:-1]
        next_q_values = (
            jax.nn.softmax(jnp.concatenate([first_target_next_q[None, :], remaining_next_q], axis=0), axis=-1)
            @ self.bin_centers
        )
        targets = self.compute_target(next_q_values, sample)
        projected_targets = self.project_target(targets)

        loss = optax.softmax_cross_entropy(q_logits, projected_targets, axis=-1)

        return loss

    def project_target(self, target):
        erf_support = jax.scipy.special.erf(
            (self.support - self.clip_target(target)[:, None]) / (jnp.sqrt(2) * self.sigma)
        )
        return (erf_support[:, 1:] - erf_support[:, :-1]) / (erf_support[:, -1] - erf_support[:, 0] + 1e-9)[:, None]

    def compute_target(self, next_q, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(next_q, axis=-1)

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        # computes the best action for a single state
        idx_params = jax.random.randint(key, (), 0, self.n_bellman_iterations)
        return jnp.argmax(jax.nn.softmax(self.networks.apply(params, state)[idx_params], axis=-1) @ self.bin_centers)

    def get_model(self):
        return {"params": self.params, "root_params": self.root_params}
