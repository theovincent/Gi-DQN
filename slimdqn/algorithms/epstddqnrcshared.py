from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from slimdqn.algorithms.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class EPSTDDQNRCShared:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        features: list,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_period: int,  # for logging only
        weight_decay: float,
        adam_eps: float = 1e-8,
    ):
        self.network = DQNNet(features, n_actions, n_heads=1, n_h_heads=1)
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

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
        self.cumulative_q_loss = 0
        self.cumulative_h_loss = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        for _ in range(int(max(self.update_to_data, 1))):
            # if update_to_data < 1, only perform one update if step = 0 [1 / self.update_to_data]
            if self.update_to_data < 1 and step % (1 / self.update_to_data) != 0:
                return None
            batch_samples, (sample_keys, importance_weights) = replay_buffer.sample()

            self.params, self.optimizer_state, per_sample_q_loss, per_sample_h_loss = self.learn_on_batch(
                self.params, self.optimizer_state, batch_samples, importance_weights
            )

            replay_buffer.update(sample_keys, per_sample_q_loss + per_sample_h_loss)

            self.cumulative_q_loss += per_sample_q_loss.mean()
            self.cumulative_h_loss += per_sample_h_loss.mean()

    def update_target_params(self, step: int):
        if step % self.target_update_period == 0:
            self.logs = {
                "n_training_steps": step,
                "loss": self.cumulative_q_loss / (self.target_update_period * self.update_to_data),
                "h_loss": self.cumulative_h_loss / (self.target_update_period * self.update_to_data),
            }
            self.cumulative_q_loss = 0
            self.cumulative_h_loss = 0

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples, importance_weights):
        grad_loss, (per_sample_q_loss, per_sample_h_loss) = jax.grad(self.loss_on_batch, has_aux=True)(
            params, batch_samples, importance_weights
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, per_sample_q_loss, per_sample_h_loss

    def loss_on_batch(self, params: FrozenDict, samples, importance_weights):
        total_loss, q_losses, h_losses = jax.vmap(self.loss, in_axes=(None, 0, 0))(params, samples, importance_weights)
        return total_loss.mean(), (q_losses, h_losses)

    def loss(self, params: FrozenDict, sample: ReplayElement, importance_weight):
        q_values, h_values = self.network.apply(params, sample.state)
        q_value, h_value = q_values[sample.action], h_values[sample.action]

        target = self.compute_target(params, sample)
        td_error = target - q_value
        h_loss = h_value * jax.lax.stop_gradient(h_value - td_error)
        td_loss = target * jax.lax.stop_gradient(h_value) - q_value * jax.lax.stop_gradient(td_error)

        return importance_weight * (td_loss + h_loss), jnp.square(td_error), jnp.square(h_value - td_error)

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            self.network.apply(params, sample.next_state)[0]
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # H-predictions are used for exploration
        q_values, h_values = self.network.apply(params, state)
        return jnp.argmax(q_values), jnp.argmax(jnp.absolute(h_values))

    def get_model(self):
        return {"params": self.params}
