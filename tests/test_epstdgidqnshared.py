import unittest
import numpy as np
import jax
import jax.numpy as jnp

from slimdqn.algorithms.epstdgidqnshared import EPSTDGiDQNShared, shift_params
from tests.utils import Generator


class TestEPSTDGiDQNShared(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        key_actions, key_feature_1, key_feature_2, key_feature_3, key_feature_4 = jax.random.split(self.key, 5)
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.q = EPSTDGiDQNShared(
            self.key,
            self.observation_dim,
            self.n_actions,
            5,  # n_bellman_iterations
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
        )

        self.generator = Generator(None, self.observation_dim, self.n_actions)

    def _next_q_values(self, sample):
        next_q_first = self.q.root_network.apply(self.q.target_params, sample.next_state)  # (n_actions,)
        next_q_remaining = self.q.online_networks.apply(self.q.params, sample.next_state)[0][:-1]  # (K-1, n_actions)
        return jnp.concatenate([next_q_first[None, :], next_q_remaining], axis=0)  # (K, n_actions)

    def test_compute_target(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)

        next_q_values = self._next_q_values(sample)
        computed_target = self.q.compute_target(next_q_values, sample)  # (K,)

        expected = float(sample.reward) + (1 - float(sample.is_terminal)) * (
            self.q.gamma**self.q.update_horizon
        ) * np.max(np.array(next_q_values), axis=-1)

        self.assertEqual(computed_target.shape, (self.q.n_bellman_iterations,))
        np.testing.assert_allclose(np.array(computed_target), expected, rtol=1e-5, atol=1e-5)

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)

        computed_loss = self.q.loss(self.q.params, self.q.target_params, sample, jnp.ones(1))[0].sum()

        q_outputs, h_outputs = self.q.online_networks.apply(self.q.params, sample.state)
        q_values, h_values = q_outputs[:, sample.action], h_outputs[:, sample.action]  # (K,), (K-1,)

        targets = self.q.compute_target(self._next_q_values(sample), sample)  # (K,)
        td_errors = targets - q_values

        h_loss = jnp.append(jnp.zeros(1), h_values * (h_values - td_errors[1:]))  # (K,)
        target_loss = jnp.append(jnp.zeros(1), targets[1:] * h_values)  # (K,)
        td_loss = target_loss - q_values * td_errors  # (K,)
        expected = (td_loss + h_loss).sum()

        np.testing.assert_allclose(float(expected), float(computed_loss), rtol=1e-4, atol=1e-4)

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_q_action, computed_h_action = self.q.best_action(self.q.params, state)

        q_outputs, h_outputs = self.q.online_networks.apply(self.q.params, state)  # (K, n_actions), (K-1, n_actions)
        q_action = jnp.argmax(q_outputs.mean(axis=0))
        h_action = jnp.argmax(jnp.abs(h_outputs.mean(axis=0)))  # eps-TD: mean over heads, then |H|

        self.assertEqual(q_outputs.mean(axis=0).shape, (self.n_actions,))
        self.assertEqual(q_action, computed_q_action)
        self.assertEqual(h_action, computed_h_action)

    def test_target_update(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        q_outputs, h_outputs = self.q.online_networks.apply(self.q.params, state)
        shifted_params = shift_params(self.q.params, self.n_actions)
        shifted_q_outputs, shifted_h_outputs = self.q.online_networks.apply(shifted_params, state)

        # after shifting, head k of the shifted net equals head k+1 of the original
        self.assertAlmostEqual(float(np.linalg.norm(shifted_q_outputs[:-1] - q_outputs[1:])), 0.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(shifted_h_outputs[:-1] - h_outputs[1:])), 0.0, places=5)
