from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.algorithms.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@jax.jit
def shift_params(x):
    return jax.tree.map(lambda x: x.at[:-1].set(x[1:]), x)


class iC51:
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
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_period: int,
        min_value: float,
        max_value: float,
        adam_eps: float = 1e-8,
    ):
        self.n_actions = n_actions
        self.n_bellman_iterations = n_bellman_iterations
        self.network = DQNNet(
            features,
            architecture_type,
            layer_norm,
            gap,
            False,
            n_actions,
            n_heads=1,
            n_h_heads=0,
            n_bins=51,
        )

        self.n_bins = 51

        # initialize 1 network with K heads
        self.params = jax.vmap(self.network.init, in_axes=(0, None))(
            jax.random.split(key, self.n_bellman_iterations + 1), jnp.zeros(observation_dim, dtype=jnp.float32)
        )
        # initialize the target networks
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
        for _ in range(int(max(self.update_to_data, 1))):
            # if update_to_data < 1, only perform one update if step = 0 [1 / self.update_to_data]
            if self.update_to_data < 1 and step % (1 / self.update_to_data) != 0:
                return None
            batch_samples, (sample_keys, importance_weights) = replay_buffer.sample()

            self.params, self.optimizer_state, per_sample_q_losses = self.learn_on_batch(
                self.params, self.optimizer_state, batch_samples, importance_weights
            )

            replay_buffer.update(sample_keys, per_sample_q_losses.mean(axis=1))

            self.cumulative_losses += per_sample_q_losses.mean(axis=0)

    def update_target_params(self, step: int):
        # update target network parameters every target_update_period steps. This starts the next Bellman iteration
        if step % self.target_update_period == 0:
            # Window shift
            self.params = shift_params(self.params)
            self.logs = {
                "n_training_steps": step,
                "loss": np.mean(self.cumulative_losses) / (self.target_update_period * self.update_to_data),
            }
            for idx_network in range(0, min(5, self.n_bellman_iterations)):
                self.logs[f"networks/{idx_network}_loss"] = self.cumulative_losses[idx_network] / (
                    self.target_update_period * self.update_to_data
                )

            self.cumulative_losses = np.zeros(self.n_bellman_iterations)

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples, importance_weights):
        grad_loss, (per_sample_q_losses) = jax.grad(self.loss_on_batch, has_aux=True)(
            params, batch_samples, importance_weights
        )

        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)

        params = optax.apply_updates(params, updates)

        return params, optimizer_state, per_sample_q_losses

    def loss_on_batch(self, params: FrozenDict, samples, importance_weights):
        losses, per_sample_q_loss = jax.vmap(self.loss, in_axes=(None, 0, 0))(params, samples, importance_weights)

        return losses.sum(axis=-1).mean(), per_sample_q_loss

    def loss(self, params: FrozenDict, sample: ReplayElement, importance_weight):
        # computes the loss for a single sample
        q_logits = jax.vmap(self.network.apply, in_axes=(0, None))(jax.tree.map(lambda x: x[1:], params), sample.state)[
            :, sample.action
        ]
        best_action = self.best_action(params, sample.next_state)

        target_next_q_probs = jax.nn.softmax(
            jax.vmap(self.network.apply, in_axes=(0, None))(jax.tree.map(lambda x: x[:-1], params), sample.next_state),
            axis=-1,
        )
        projected_target = self.compute_target(target_next_q_probs[:, best_action], sample)
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
            (
                jax.nn.softmax(
                    jax.vmap(self.network.apply, in_axes=(0, None))(jax.tree.map(lambda x: x[1:], params), state),
                    axis=-1,
                )
                @ self.support
            ).mean(axis=0)
        )

    def get_model(self):
        return {"params": self.params}
