from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.algorithms.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@partial(jax.jit, static_argnames=("linear_heads", "n_actions", "n_bins"))
def set_target_params(params, linear_heads, n_actions, n_bins):
    if not linear_heads:
        q_heads = jax.tree.map(lambda p: p[0], params["params"]["q_heads"])
    else:
        q_heads = jax.tree.map(lambda p: p[..., : n_actions * n_bins], params["params"]["q_heads"])
    return optax.tree_utils.tree_set(params, q_heads=q_heads)


@partial(jax.jit, static_argnames=("linear_heads", "n_actions", "n_bins"))
def shift_params(params, linear_heads, n_actions, n_bins):
    if not linear_heads:
        q_heads = jax.tree.map(lambda p: p.at[:-1].set(p[1:]), params["params"]["q_heads"])
    else:
        q_heads = jax.tree.map(
            lambda p: p.at[..., : -n_actions * n_bins].set(p[..., n_actions * n_bins :]), params["params"]["q_heads"]
        )

    return optax.tree_utils.tree_set(params, q_heads=q_heads)


class iC51Shared:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_bellman_iterations: int,
        features: list,
        architecture_type: str,
        layer_norm: bool,
        gap: bool,
        linear_heads: bool,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_period: int,
        min_value: float,
        max_value: float,
        adam_eps: float = 1e-8,
    ):
        key, key_params = jax.random.split(key)
        self.n_actions = n_actions
        self.n_bellman_iterations = n_bellman_iterations
        self.n_bins = 51
        self.online_networks = DQNNet(
            features,
            architecture_type,
            layer_norm,
            gap,
            linear_heads,
            n_actions,
            n_heads=self.n_bellman_iterations,
            n_h_heads=0,
            n_bins=51,
        )
        self.root_network = DQNNet(
            features, architecture_type, layer_norm, gap, linear_heads, n_actions, n_heads=1, n_h_heads=0, n_bins=51
        )

        # initialize 1 network with K heads
        self.params = self.online_networks.init(key_params, jnp.zeros(observation_dim, dtype=jnp.float32))
        # initialize the target networks
        self.target_params = set_target_params(self.params, linear_heads, self.n_actions, n_bins=self.n_bins)
        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_period = target_update_period
        self.min_value = min_value
        self.max_value = max_value
        self.support = jnp.linspace(min_value, max_value, 51, dtype=jnp.float32)
        self.bin_size = (max_value - min_value) / 50
        self.clip_target = lambda target: jnp.clip(target, min_value, max_value)
        self.cumulative_losses = np.zeros(self.n_bellman_iterations)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        # update online network parameters every update_to_data steps
        for _ in range(int(max(self.update_to_data, 1))):
            # if update_to_data < 1, only perform one update if step = 0 [1 / self.update_to_data]
            if self.update_to_data < 1 and step % (1 / self.update_to_data) != 0:
                return None
            batch_samples, (sample_keys, importance_weights) = replay_buffer.sample()

            (self.params, self.optimizer_state, per_sample_q_losses) = self.learn_on_batch(
                self.params,
                self.target_params,
                self.optimizer_state,
                batch_samples,
                importance_weights,
            )

            replay_buffer.update(sample_keys, per_sample_q_losses.mean(axis=1))

            self.cumulative_losses += per_sample_q_losses.mean()

    def update_target_params(self, step: int):
        # update target network parameters every target_update_period steps. This starts the next Bellman iteration
        if step % self.target_update_period == 0:
            # Each target network is updated to its respective online network
            # \bar{\theta}_k <- \theta_{k + 1}, i.e., target_params[k] <- params[k]
            self.target_params = set_target_params(
                self.params, self.online_networks.linear_heads, self.n_actions, n_bins=self.n_bins
            )
            # Window shift
            self.params = shift_params(self.params, self.online_networks.linear_heads, self.n_actions, self.n_bins)
            self.logs = {
                "n_training_steps": step,
                "loss": np.mean(self.cumulative_losses) / (self.target_update_period * self.update_to_data),
            }
            for idx_network in range(0, min(5, self.n_bellman_iterations)):
                self.logs[f"networks/{idx_network}_loss"] = self.cumulative_losses[idx_network] / (
                    self.target_update_period * self.update_to_data
                )

            self.cumulative_losses = np.zeros(self.n_bellman_iterations)
            self.cumulative_variance = 0

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: FrozenDict, target_params: FrozenDict, optimizer_state, batch_samples, importance_weights
    ):
        grad_loss, per_sample_loss = jax.grad(self.loss_on_batch, has_aux=True)(
            params, target_params, batch_samples, importance_weights
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return (params, optimizer_state, per_sample_loss)

    def loss_on_batch(self, params: FrozenDict, target_params: FrozenDict, samples, importance_weights):
        losses, per_sample_q_loss = jax.vmap(self.loss, in_axes=(None, None, 0, 0))(
            params, target_params, samples, importance_weights
        )

        return losses.sum(axis=-1).mean(), per_sample_q_loss

    def loss(self, params: FrozenDict, target_params: FrozenDict, sample: ReplayElement, importance_weight):
        # computes the loss for a single sample
        q_logits = self.online_networks.apply(params, sample.state)[:, sample.action]
        best_action = self.best_action(params, sample.next_state)

        first_target_next_q_probs = jax.nn.softmax(self.root_network.apply(target_params, sample.next_state), axis=-1)
        remaining_next_q_probs = jax.nn.softmax(self.online_networks.apply(params, sample.next_state), axis=-1)[:-1]
        next_q_values = jnp.concatenate([first_target_next_q_probs[None, :], remaining_next_q_probs], axis=0)
        projected_target = self.compute_target(next_q_values[:, best_action], sample)
        ce = optax.softmax_cross_entropy(q_logits, jax.lax.stop_gradient(projected_target), axis=-1)

        return importance_weight * ce, ce

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
