from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.algorithms.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@jax.jit
def set_target_params(params):
    return optax.tree_utils.tree_set(params, q_heads=jax.tree.map(lambda p: p[0], params["params"]["q_heads"]))


@jax.jit
def shift_params(params):
    return optax.tree_utils.tree_set(
        params, q_heads=jax.tree.map(lambda p: p.at[:-1].set(p[1:]), params["params"]["q_heads"])
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
        layer_norm: bool,
        gap: bool,
        linear_heads: bool,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_period: int,
        adam_eps: float = 1e-8,
    ):
        self.n_actions = n_actions
        self.n_bellman_iterations = n_bellman_iterations
        self.online_networks = DQNNet(
            features,
            architecture_type,
            layer_norm,
            gap,
            linear_heads,
            n_actions,
            n_heads=self.n_bellman_iterations,
            n_h_heads=0,
        )
        self.root_network = DQNNet(
            features, architecture_type, layer_norm, gap, linear_heads, n_actions, n_heads=1, n_h_heads=0
        )

        # initialize 1 network with K heads
        self.params = self.online_networks.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))
        # initialize the target networks
        self.target_params = set_target_params(self.params)

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_period = target_update_period
        self.cumulative_losses = np.zeros(self.n_bellman_iterations)
        self.cumulative_variance = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        for _ in range(int(max(self.update_to_data, 1))):
            # if update_to_data < 1, only perform one update if step = 0 [1 / self.update_to_data]
            if self.update_to_data < 1 and step % (1 / self.update_to_data) != 0:
                return None
            batch_samples, _ = replay_buffer.sample()

            self.params, self.optimizer_state, losses, variance = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )

            self.cumulative_losses += losses
            self.cumulative_variance += variance

    def update_target_params(self, step: int):
        # update target network parameters every target_update_period steps. This starts the next Bellman iteration
        if step % self.target_update_period == 0:
            self.target_params = set_target_params(self.params)
            # Window shift
            self.params = shift_params(self.params)

            logs = {
                "loss": np.mean(self.cumulative_losses) / (self.target_update_period * self.update_to_data),
                "variance": self.cumulative_variance / (self.target_update_period * self.update_to_data),
            }
            for idx_network in range(0, min(5, self.n_bellman_iterations)):
                logs[f"networks/{idx_network}_loss"] = self.cumulative_losses[idx_network] / (
                    self.target_update_period * self.update_to_data
                )

            self.cumulative_losses = np.zeros(self.n_bellman_iterations)
            self.cumulative_variance = 0
            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, target_params: FrozenDict, optimizer_state, batch_samples):
        grad_loss, (losses, variance) = jax.grad(self.loss_on_batch, has_aux=True)(params, target_params, batch_samples)

        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)

        params = optax.apply_updates(params, updates)

        return params, optimizer_state, losses, variance

    def loss_on_batch(self, params: FrozenDict, target_params: FrozenDict, samples):
        losses, variances = jax.vmap(self.loss, in_axes=(None, None, 0))(params, target_params, samples)

        return losses.mean(axis=0).sum(), (losses.mean(axis=0), variances.mean())

    def loss(self, params: FrozenDict, target_params: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        q_values = self.online_networks.apply(params, sample.state)[:, sample.action]
        next_q_values_first_target = self.root_network.apply(target_params, sample.next_state)
        next_q_values_remaining_targets = self.online_networks.apply(params, sample.next_state)[:-1]
        next_q_values = jnp.concatenate([next_q_values_first_target[None, :], next_q_values_remaining_targets], axis=0)
        targets = self.compute_target(next_q_values, sample)
        td_errors = targets - q_values

        return jnp.square(td_errors), targets**2 - targets * q_values

    def compute_target(self, next_q_values, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            next_q_values, axis=-1
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return jnp.argmax(self.online_networks.apply(params, state).mean(axis=0))

    def get_model(self):
        return {"params": self.params}
