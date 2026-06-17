import unittest
import numpy as np
import jax
import jax.numpy as jnp

from slimdqn.algorithms.sc51 import SC51
from tests.utils import Generator


def categorical_projection(next_probs, reward, is_terminal, gamma_h, vmin, vmax, n_bins):
    support = np.linspace(vmin, vmax, n_bins)
    dz = (vmax - vmin) / (n_bins - 1)
    tz = np.clip(reward + (1.0 - is_terminal) * gamma_h * support, vmin, vmax)
    b = (tz - vmin) / dz
    lower = np.floor(b).astype(np.int32)
    upper = np.ceil(b).astype(np.int32)
    m = np.zeros(n_bins)
    for j in range(n_bins):
        m[lower[j]] += next_probs[j] * (upper[j] - b[j])
        m[upper[j]] += next_probs[j] * (b[j] - lower[j])
        m[lower[j]] += next_probs[j] * (lower[j] == upper[j])
    return m


class TestSC51(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        key_actions, key_feature_1, key_feature_2, key_feature_3, key_feature_4 = jax.random.split(self.key, 5)
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.q = SC51(
            self.key,
            self.observation_dim,
            self.n_actions,
            [
                jax.random.randint(key_feature_1, (), minval=1, maxval=10),
                jax.random.randint(key_feature_2, (), minval=1, maxval=10),
                jax.random.randint(key_feature_3, (), minval=1, maxval=10),
                jax.random.randint(key_feature_4, (), minval=1, maxval=10),
            ],
            0.001,
            0.94,
            1,
            1,
            1,
        )
        self.n_bins = self.q.n_bins

        self.generator = Generator(None, self.observation_dim, self.n_actions)

    def test_compute_target(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)

        computed_target = self.q.compute_target(self.q.params, sample)

        next_logits = self.q.network.apply(self.q.params, sample.next_state)[0].reshape(self.n_actions, self.n_bins)
        next_probabilities = jax.nn.softmax(next_logits, axis=-1)
        next_q_values = next_probabilities @ self.q.support
        best_action_index = jnp.argmax(next_q_values)
        next_probabilities_target = np.array(next_probabilities[best_action_index])

        target = categorical_projection(
            next_probabilities_target,
            float(sample.reward),
            float(sample.is_terminal),
            self.q.gamma**self.q.update_horizon,
            self.q.vmin,
            self.q.vmax,
            self.n_bins,
        )

        self.assertEqual(computed_target.shape, (self.n_bins,))
        self.assertAlmostEqual(float(jnp.sum(computed_target)), 1.0, places=4)
        np.testing.assert_allclose(np.array(computed_target), target, atol=1e-4)

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)

        computed_loss = self.q.loss(self.q.params, sample, jnp.ones(1))[0]
        target_distribution = self.q.compute_target(self.q.params, sample)
        logits = self.q.network.apply(self.q.params, sample.state)[1].reshape(self.n_actions, self.n_bins)
        log_distribution = jax.nn.log_softmax(logits, axis=-1)
        cross_entropy = -jnp.sum(target_distribution * log_distribution[sample.action])
        self.assertAlmostEqual(cross_entropy.sum().item(), computed_loss.item(), places=3)

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state)

        logits = self.q.network.apply(self.q.params, state)[1].reshape(self.n_actions, self.n_bins)
        probabilities = jax.nn.softmax(logits, axis=-1)
        q_values = probabilities @ self.q.support
        best_action = jnp.argmax(q_values)

        self.assertEqual(q_values.shape, (self.n_actions,))
        self.assertEqual(best_action, computed_best_action)
