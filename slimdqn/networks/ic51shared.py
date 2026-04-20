from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@partial(jax.jit, static_argnames=["n_actions", "n_bins"])
def set_target_params(params, n_actions, n_bins):
    first_params = optax.tree_utils.tree_set(
        params,
        Dense_final={
            "kernel": params["params"]["Dense_final"]["kernel"][:, : n_actions * n_bins],
            "bias": params["params"]["Dense_final"]["bias"][: n_actions * n_bins],
        },
    )
    remaining_params = optax.tree_utils.tree_set(
        params,
        Dense_final={
            "kernel": params["params"]["Dense_final"]["kernel"][:, n_actions * n_bins :],
            "bias": params["params"]["Dense_final"]["bias"][n_actions * n_bins :],
        },
    )

    return first_params, remaining_params


@partial(jax.jit, static_argnames=["n_actions", "n_bins"])
def shift_params(params, n_actions, n_bins):
    # Each online network is updated to the following online network
    # \theta_k <- \theta_{k + 1}, i.e., params[k] <- params[k + 1]
    kernel = params["params"]["Dense_final"]["kernel"]
    bias = params["params"]["Dense_final"]["bias"]
    params["params"]["Dense_final"]["kernel"] = kernel.at[:, : -n_actions * n_bins].set(kernel[:, n_actions * n_bins :])
    params["params"]["Dense_final"]["bias"] = bias.at[: -n_actions * n_bins].set(bias[n_actions * n_bins :])

    return params


@partial(jax.jit, static_argnames=["n_actions", "n_bins"])
def sync_target_params(params, n_actions, n_bins):
    # Each target network is synchronized to the online network it represents
    # \bar{\theta}_k <- \theta_k, i.e., target_params[k] <- params[k-1]
    return optax.tree_utils.tree_set(
        params,
        Dense_final={
            "kernel": params["params"]["Dense_final"]["kernel"][:, : -n_actions * n_bins],
            "bias": params["params"]["Dense_final"]["bias"][: -n_actions * n_bins],
        },
    )


class iC51Shared:
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
        min_value: float,
        max_value: float,
        adam_eps: float = 1e-8,
    ):
        key, key_params = jax.random.split(key)
        self.n_actions = n_actions
        self.n_bellman_iterations = n_bellman_iterations
        self.online_networks = DQNNet(features, architecture_type, n_actions, self.n_bellman_iterations, n_bins=51)
        self.root_network = DQNNet(features, architecture_type, n_actions, n_bins=51)
        self.remaining_target_networks = DQNNet(
            features, architecture_type, n_actions, self.n_bellman_iterations - 1, n_bins=51
        )

        # initialize 1 network with K heads
        self.params = self.online_networks.init(key_params, jnp.zeros(observation_dim, dtype=jnp.float32))
        # initialize the target networks
        self.first_target_params, self.remaining_target_params = set_target_params(self.params, n_actions, 51)

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.target_sync_frequency = target_sync_frequency
        self.min_value = min_value
        self.max_value = max_value
        self.n_bins = 51
        self.support = jnp.linspace(min_value, max_value, 51, dtype=jnp.float32)
        self.bin_size = (max_value - min_value) / 50
        self.clip_target = lambda target: jnp.clip(target, min_value, max_value)
        self.cumulative_losses = np.zeros(self.n_bellman_iterations)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        # update online network parameters every update_to_data steps
        for _ in range(int(self.update_to_data)):
            batch_samples, _ = replay_buffer.sample()

            (self.params, self.optimizer_state, losses) = self.learn_on_batch(
                self.params, self.first_target_params, self.remaining_target_params, self.optimizer_state, batch_samples
            )

            self.cumulative_losses += losses

    def update_target_params(self, step: int):
        # update target network parameters every target_update_frequency steps. This starts the next Bellman iteration
        if step % self.target_update_frequency == 0:
            # Each target network is updated to its respective online network
            # \bar{\theta}_k <- \theta_{k + 1}, i.e., target_params[k] <- params[k]
            self.first_target_params, self.remaining_target_params = set_target_params(
                self.params, self.n_actions, self.n_bins
            )
            # Window shift
            self.params = shift_params(self.params, self.n_actions, self.n_bins)

            logs = {"loss": np.mean(self.cumulative_losses) / (self.target_update_frequency * self.update_to_data)}
            for idx_network in range(0, min(5, self.n_bellman_iterations)):
                logs[f"networks/{idx_network}_loss"] = self.cumulative_losses[idx_network] / (
                    self.target_update_frequency * self.update_to_data
                )

            self.cumulative_losses = np.zeros(self.n_bellman_iterations)
            self.cumulative_variance = 0
            return True, logs
        # sync target network parameters to previous online network every target_sync_frequency steps
        if step % self.target_sync_frequency == 0:
            self.remaining_target_params = sync_target_params(self.params, self.n_actions, self.n_bins)
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
        grad_loss, losses = jax.grad(self.loss_on_batch, has_aux=True)(
            params, first_target_params, remaining_target_params, batch_samples
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return (params, optimizer_state, losses)

    def loss_on_batch(
        self, params: FrozenDict, first_target_params: FrozenDict, remaining_target_params: FrozenDict, samples
    ):
        losses = jax.vmap(self.loss, in_axes=(None, None, None, 0))(
            params, first_target_params, remaining_target_params, samples
        )

        return losses.sum(axis=-1).mean(), losses.mean(axis=0)  # mean over the samples but keep networks separated

    def loss(
        self,
        params: FrozenDict,
        first_target_params: FrozenDict,
        remaining_target_params: FrozenDict,
        sample: ReplayElement,
    ):
        # computes the loss for a single sample
        q_logits = self.online_networks.apply(params, sample.state)[:, sample.action]
        best_action = self.best_action(params, sample.next_state)

        first_target_next_q_probs = jax.nn.softmax(
            self.root_network.apply(first_target_params, sample.next_state), axis=-1
        )
        remaining_next_q_probs = jax.nn.softmax(
            self.remaining_target_networks.apply(remaining_target_params, sample.next_state), axis=-1
        )
        next_q_values = jnp.concatenate([first_target_next_q_probs[None, :], remaining_next_q_probs], axis=0)
        projected_target = self.compute_target(next_q_values[:, best_action], sample)
        return optax.softmax_cross_entropy(q_logits, projected_target, axis=-1)

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
        return jnp.argmax(
            (jax.nn.softmax(self.online_networks.apply(params, state), axis=-1) @ self.support).mean(axis=0)
        )

    def get_model(self):
        return {"params": self.params}
