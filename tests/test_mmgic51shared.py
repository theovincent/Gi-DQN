import unittest
import numpy as np
import jax
import jax.numpy as jnp

from slimdqn.algorithms.mmgic51shared import MMGiC51Shared, set_target_params, shift_params
from tests.utils import Generator


def categorical_projection(next_probs, reward, is_terminal, gamma_h, vmin, vmax, n_bins, shift=0.0):
    support = np.linspace(vmin, vmax, n_bins)
    dz = (vmax - vmin) / (n_bins - 1)
    tz = np.clip(reward + (1.0 - is_terminal) * gamma_h * (support + shift), vmin, vmax)
    b = (tz - vmin) / dz
    lower = np.floor(b).astype(np.int32)
    upper = np.ceil(b).astype(np.int32)
    m = np.zeros(n_bins)
    for j in range(n_bins):
        m[lower[j]] += next_probs[j] * (upper[j] - b[j])
        m[upper[j]] += next_probs[j] * (b[j] - lower[j])
        m[lower[j]] += next_probs[j] * (lower[j] == upper[j])
    return m


class TestMMGiC51Shared(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        key_actions, key_feature_1, key_feature_2, key_feature_3, key_feature_4 = jax.random.split(self.key, 5)
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.q = MMGiC51Shared(
            self.key,
            self.observation_dim,
            self.n_actions,
            5,
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
            1,
            1,
        )
        self.n_bins = self.q.n_bins

        self.generator = Generator(None, self.observation_dim, self.n_actions)

    def test_compute_target(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)

        next_distribution = jax.nn.softmax(
            self.q.online_networks.apply(self.q.params, sample.next_state)[0].reshape(-1, self.n_actions, self.n_bins),
            axis=-1,
        )  # (K, n_actions, n_bins)
        computed_target = self.q.compute_target(next_distribution, sample)  # (K, n_bins)

        next_q_values = np.array(next_distribution @ self.q.support)  # (K, n_actions)
        scaled_q_values = self.q.omega * np.array(next_q_values)
        scaled_q_values -= np.max(scaled_q_values, axis=-1, keepdims=True)
        weights = np.exp(scaled_q_values) / np.sum(np.exp(scaled_q_values), axis=-1, keepdims=True)
        next_probabilities_target = np.einsum("ka,kan->kn", weights, np.array(next_distribution))
        logsumexp_q = np.max(self.q.omega * next_q_values, axis=-1) + np.log(np.exp(scaled_q_values).sum(axis=-1))
        mellowmax = (logsumexp_q - np.log(self.n_actions)) / self.q.omega  # (K,)
        boltzmann_mean = np.einsum("ka,ka->k", weights, next_q_values)  # (K,)
        shift = mellowmax - boltzmann_mean  # (K,)
        target = np.stack(
            [
                categorical_projection(
                    next_probabilities_target[k],
                    float(sample.reward),
                    float(sample.is_terminal),
                    self.q.gamma**self.q.update_horizon,
                    self.q.vmin,
                    self.q.vmax,
                    self.n_bins,
                    shift[k],
                )
                for k in range(self.q.n_bellman_iterations)
            ]
        )

        self.assertAlmostEqual(computed_target.shape, (self.q.n_bellman_iterations, self.n_bins))
        np.testing.assert_allclose(np.array(computed_target), target, atol=1e-4)
        np.testing.assert_allclose(
            np.array(computed_target).sum(axis=-1), np.ones(self.q.n_bellman_iterations), atol=1e-4
        )

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)

        computed_loss, computed_cross_entropy, computed_suboptimality = self.q.loss(
            self.q.params, self.q.target_params, sample, jnp.ones(1)
        )
        computed_loss = computed_loss.sum()

        next_distribution_first = jax.nn.softmax(
            self.q.root_network.apply(self.q.target_params, sample.next_state).reshape(self.n_actions, self.n_bins),
            axis=-1,
        )
        next_distribution_remaining = jax.nn.softmax(
            self.q.online_networks.apply(self.q.params, sample.next_state)[0].reshape(-1, self.n_actions, self.n_bins)[
                :-1
            ],
            axis=-1,
        )
        next_distribution = jnp.concatenate([next_distribution_first[None], next_distribution_remaining], axis=0)
        target = self.q.compute_target(next_distribution, sample)  # (K, n_bins)

        q_logits, h_logits = self.q.online_networks.apply(self.q.params, sample.state)
        q_log_distribution = jax.nn.log_softmax(q_logits.reshape(-1, self.n_actions, self.n_bins), axis=-1)[
            :, sample.action, :
        ]
        h_logits = h_logits.reshape(-1, self.n_actions, self.n_bins)[:, sample.action, :]  # (K-1, n_bins)

        target_loss = jnp.append(jnp.zeros(1), jnp.sum(target[1:] * h_logits, axis=-1))  # (K,)
        h_loss = jnp.append(
            jnp.zeros(1),
            jnp.sum(target[1:] * h_logits, axis=-1)
            - jax.scipy.special.logsumexp(h_logits + q_log_distribution[1:], axis=-1),
        )
        td_loss = target_loss - jnp.sum(target * q_log_distribution, axis=-1)  # (K,)
        cross_entropy = -jnp.sum(target * q_log_distribution, axis=-1)  # (K,)
        suboptimality = cross_entropy + jnp.sum(jax.scipy.special.xlogy(target, target), axis=-1) - h_loss  # (K,)

        self.assertAlmostEqual(float((td_loss - h_loss).sum()), float(computed_loss), places=4)
        np.testing.assert_allclose(np.array(cross_entropy), np.array(computed_cross_entropy), atol=1e-4)
        np.testing.assert_allclose(np.array(suboptimality[1:]), np.array(computed_suboptimality), atol=1e-4)

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state)

        logits = self.q.online_networks.apply(self.q.params, state)[0].reshape(-1, self.n_actions, self.n_bins)
        probabilities = jax.nn.softmax(logits, axis=-1)
        q_values = (probabilities @ self.q.support).mean(axis=0)
        best_action = jnp.argmax(q_values)

        self.assertAlmostEqual(q_values.shape, (self.n_actions,))
        self.assertAlmostEqual(best_action, computed_best_action)

    def test_target_update(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        first_q_values = self.q.online_networks.apply(self.q.params, state)[0][0]
        target_params = set_target_params(self.q.params, self.n_actions * self.n_bins)
        target_q_values = self.q.root_network.apply(target_params, state)

        self.assertAlmostEqual(np.linalg.norm(first_q_values - target_q_values), 0)

        q_values, h_values = self.q.online_networks.apply(self.q.params, state)
        shifted_params = shift_params(self.q.params, self.n_actions * self.n_bins)
        shifted_q_values, shifted_h_values = self.q.online_networks.apply(shifted_params, state)

        self.assertAlmostEqual(np.linalg.norm(shifted_q_values[:-1] - q_values[1:]), 0)
        self.assertAlmostEqual(np.linalg.norm(shifted_h_values[:-1] - h_values[1:]), 0)
