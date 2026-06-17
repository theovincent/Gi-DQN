import unittest
import numpy as np
import jax
import jax.numpy as jnp

from slimdqn.algorithms.c51 import C51
from tests.utils import Generator


class TestC51(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        key_actions, key_feature_1, key_feature_2, key_feature_3, key_feature_4 = jax.random.split(self.key, 5)
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.n_bins = np.random.randint(1, 100)
        self.q = C51(
            self.key,
            self.observation_dim,
            self.n_actions,
            [
                jax.random.randint(key_feature_1, (), minval=1, maxval=10),
                jax.random.randint(key_feature_2, (), minval=1, maxval=10),
                jax.random.randint(key_feature_3, (), minval=1, maxval=10),
                jax.random.randint(key_feature_4, (), minval=1, maxval=10),
            ],
            "impala",
            True,
            False,
            0.001,
            0.94,
            1,
            self.n_bins,
            np.random.randint(-100, 0),
            np.random.randint(0, 100),
        )

        self.generator = Generator(None, self.observation_dim, self.n_actions)

    def test_compute_target(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)

        computed_target = self.q.compute_target(self.q.params, sample)

        next_logits = self.q.network.apply(self.q.params, sample.next_state).reshape(self.n_actions, self.n_bins)
        non_aligned_target_atoms = (
            sample.reward + (1 - sample.is_terminal) * (self.q.gamma**self.q.update_horizon) * self.q.support
        )
        next_probabilities = jax.nn.softmax(next_logits, axis=-1)  # (n_actions, n_bins)
        next_q_values = next_probabilities @ self.q.support  # (n_actions,)
        best_action_index = jnp.argmax(next_q_values)  # (1,)
        next_probabilities_target = next_probabilities[best_action_index, :]  # (n_bins,)
        clipped_non_aligned_target_atoms = jnp.clip(non_aligned_target_atoms, self.q.vmin, self.q.vmax)  # (n_bins,)

        fractional_coordinates = (clipped_non_aligned_target_atoms - self.q.support[0]) / (
            (self.q.vmax - self.q.vmin) / (self.n_bins - 1)
        )  # (n_bins,)
        lower, upper = jnp.floor(fractional_coordinates).astype(jnp.int32), jnp.ceil(fractional_coordinates).astype(
            jnp.int32
        )  # (n_bins,), (n_bins,)

        target_distribution = jnp.zeros(self.n_bins)  # (n_bins,)
        target_distribution = target_distribution.at[lower].add(
            next_probabilities_target * (upper - fractional_coordinates)
        )  # (n_bins,)
        target_distribution = target_distribution.at[upper].add(
            next_probabilities_target * (fractional_coordinates - lower)
        )  # (n_bins,)
        target_distribution = target_distribution.at[lower].add(
            jnp.where(lower == upper, next_probabilities_target, 0.0)
        )  # (n_bins,)

        self.assertEqual(next_logits.shape, (self.n_actions, self.n_bins))
        self.assertEqual(target_distribution, computed_target)

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)

        computed_loss = self.q.loss(self.q.params, self.q.params, sample, jnp.ones(1))[0]

        next_logits = self.q.network.apply(self.q.params, sample.next_state).reshape(
            self.n_actions, self.n_bins
        )  # (n_actions, n_bins)
        next_probabilities = jax.nn.softmax(next_logits, axis=-1)  # (n_actions, n_bins)
        next_q_values = next_probabilities @ self.q.support  # (n_actions,)
        best_action_index = jnp.argmax(next_q_values)  # (1,)
        next_probabilities_target = next_probabilities[best_action_index, :]  # (n_bins,)

        non_aligned_target_atoms = (
            sample.reward + (1 - sample.is_terminal) * (self.q.gamma**self.q.update_horizon) * self.q.support
        )  # (n_bins,)
        clipped_non_aligned_target_atoms = jnp.clip(non_aligned_target_atoms, self.q.vmin, self.q.vmax)  # (n_bins,)

        fractional_coordinates = (clipped_non_aligned_target_atoms - self.q.support[0]) / (
            (self.q.vmax - self.q.vmin) / (self.n_bins - 1)
        )  # (n_bins,)
        lower, upper = jnp.floor(fractional_coordinates).astype(jnp.int32), jnp.ceil(fractional_coordinates).astype(
            jnp.int32
        )  # (n_bins,), (n_bins,)

        target_distribution = jnp.zeros(self.n_bins)  # (n_bins,)
        target_distribution = target_distribution.at[lower].add(
            next_probabilities_target * (upper - fractional_coordinates)
        )  # (n_bins,)
        target_distribution = target_distribution.at[upper].add(
            next_probabilities_target * (fractional_coordinates - lower)
        )  # (n_bins,)
        target_distribution = target_distribution.at[lower].add(
            jnp.where(lower == upper, next_probabilities_target, 0.0)
        )  # (n_bins,)

        logits = self.q.network.apply(self.q.params, sample.state).reshape(
            self.n_actions, self.n_bins
        )  # (n_actions, n_bins)
        log_distribution = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(target_distribution * log_distribution[sample.action, :]) * jnp.ones(1)

        self.assertEqual(loss, computed_loss)

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state)

        logits = self.q.network.apply(self.q.params, state).reshape(self.n_actions, self.n_bins)
        probabilities = jax.nn.softmax(logits, axis=-1)
        q_values = probabilities @ self.q.support
        best_action = jnp.argmax(q_values)

        self.assertEqual(q_values.shape, (self.n_actions,))
        self.assertEqual(best_action, computed_best_action)
