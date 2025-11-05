from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class HLDQNRCShared:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        features: list,
        architecture_type: str,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
        weight_decay: float,
        mu: float,
        n_bins: int,
        min_value: float,
        max_value: float,
        sigma: float,
        adam_eps: float = 1e-8,
    ):
        key, key_params = jax.random.split(key)
        self.n_actions = n_actions
        self.network = DQNNet(features, architecture_type, n_actions, n_heads=1, n_h_heads=1, n_bins=n_bins)
        # initialize online network
        self.params = self.network.init(key_params, jnp.zeros(observation_dim, dtype=jnp.float32))

        # regularize the TD-error estimator network
        self.optimizer = optax.adamw(
            learning_rate,
            eps=adam_eps,
            weight_decay=weight_decay,
            mask=jax.tree_util.tree_map_with_path(
                lambda path, leaf: (True if "Dense_final_h" in path[1].key else False), self.params
            ),
        )
        self.optimizer_state = self.optimizer.init(self.params)

        self.mu = mu
        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.n_bins = n_bins
        self.sigma = sigma
        self.support = jnp.linspace(min_value, max_value, self.n_bins + 1, dtype=jnp.float32)
        self.bin_centers = (self.support[:-1] + self.support[1:]) / 2
        self.clip_target = lambda target: jnp.clip(target, min_value, max_value)
        self.cumulative_q_losses = 0
        self.cumulative_h_losses = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        # Update the network parameters every `update_to_data` steps
        for _ in range(int(self.update_to_data)):
            batch_samples, _ = replay_buffer.sample()

            (self.params, self.optimizer_state, q_losses, h_losses) = self.learn_on_batch(
                self.params, self.optimizer_state, batch_samples
            )

            self.cumulative_q_losses += q_losses
            self.cumulative_h_losses += h_losses

    def update_target_params(self, step: int):
        # shift the network parameters every `target_update_frequency` steps. This starts the next Bellman iteration
        if step % self.target_update_frequency == 0:

            logs = {
                "loss": np.mean(self.cumulative_q_losses) / (self.target_update_frequency * self.update_to_data),
                "h_loss": np.mean(self.cumulative_h_losses) / (self.target_update_frequency * self.update_to_data),
            }

            self.cumulative_q_losses = 0
            self.cumulative_h_losses = 0
            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples):
        (grad_loss), (q_losses, h_losses) = jax.grad(self.loss_on_batch, has_aux=True)(params, batch_samples)

        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state, params)

        params = optax.apply_updates(params, updates)

        return (params, optimizer_state, q_losses, h_losses)

    def loss_on_batch(self, params: FrozenDict, samples):
        # vmap to compute the loss for all samples
        total_losses, q_losses, h_losses = jax.vmap(self.loss, in_axes=(None, 0))(params, samples)

        return total_losses.mean(), (q_losses.mean(), h_losses.mean())

    def loss(self, params: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        q_logits, h_logits = self.network.apply(params, sample.state)
        h_logits_a, q_logits_a = h_logits[0, sample.action], q_logits[0, sample.action]
        q_value_probs = jax.nn.softmax(q_logits_a)

        next_q_value = jax.nn.softmax(self.network.apply(params, sample.next_state)[0][0], axis=-1) @ self.bin_centers
        projected_target = self.project_target(self.compute_target(next_q_value, sample))
        baseline = jax.scipy.special.logsumexp(a=h_logits_a, b=jax.lax.stop_gradient(q_value_probs), axis=-1)
        centered_h_logits = h_logits_a - baseline
        kl = jnp.sum(
            jax.lax.stop_gradient(h_logits_a) * projected_target, axis=-1
        ) +  optax.softmax_cross_entropy(q_logits_a, jax.lax.stop_gradient(projected_target), axis=-1)
        h_loss = -jnp.sum(
            centered_h_logits * jax.lax.stop_gradient(projected_target), axis=-1
        ) 

        return kl + self.mu * h_loss, kl, h_loss

    def compute_target(self, next_q, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(next_q)

    def project_target(self, target):
        erf_support = jax.scipy.special.erf((self.support - self.clip_target(target)) / (jnp.sqrt(2) * self.sigma))
        return (erf_support[1:] - erf_support[:-1]) / (erf_support[-1] - erf_support[0] + 1e-9)

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array = None):
        # computes the best action for a single state
        return jnp.argmax(jax.nn.softmax(self.network.apply(params, state)[0][0], axis=-1) @ self.bin_centers)

    def get_model(self):
        return {"params": self.params}
