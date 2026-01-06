import unittest
import numpy as np
import jax
import jax.numpy as jnp

from slimdqn.algorithms.gidqnshared import GiDQNShared, set_target_params, shift_params
from tests.utils import Generator


class TestGiDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        key_actions, key_feature_1, key_feature_2, key_feature_3, key_feature_4 = jax.random.split(self.key, 5)
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.q = GiDQNShared(
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
            "cnn",
            True,
            False,
            False,
            0.001,
            0.94,
            1,
            1,
            True,
            1,
            1,
        )

        self.generator = Generator(None, self.observation_dim, self.n_actions)

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)

        computed_loss = self.q.loss(self.q.params, self.q.target_params, sample)[0].sum()

        first_next_q_values = self.q.root_network.apply(self.q.target_params, sample.next_state)
        next_q_values = self.q.online_networks.apply(self.q.params, sample.next_state)[0]
        targets = self.q.compute_target(
            jnp.concatenate([first_next_q_values[None], next_q_values[:-1]], axis=0), sample
        )
        q_values, h_values = self.q.online_networks.apply(self.q.params, sample.state)
        q_predictions, h_predictions = q_values[:, sample.action], h_values[:, sample.action]

        td_loss = (targets[1:] * h_predictions).sum() - (q_predictions * (targets - q_predictions)).sum()
        h_loss = -(h_predictions * (targets[1:] - q_predictions[1:] - h_predictions)).sum()

        self.assertAlmostEqual(td_loss + h_loss, computed_loss, places=4)

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state)

        q_values = self.q.online_networks.apply(self.q.params, state)[0].mean(axis=0)
        best_action = jnp.argmax(q_values)

        self.assertEqual(q_values.shape, (self.n_actions,))
        self.assertEqual(best_action, computed_best_action)

    def test_target_update(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        first_q_values = self.q.online_networks.apply(self.q.params, state)[0][0]
        target_params = set_target_params(self.q.params)
        target_q_values = self.q.root_network.apply(target_params, state)

        self.assertEqual(np.linalg.norm(first_q_values - target_q_values), 0)

        q_values, h_values = self.q.online_networks.apply(self.q.params, state)
        shifted_params = shift_params(self.q.params)
        shifted_q_values, shifted_h_values = self.q.online_networks.apply(shifted_params, state)

        self.assertEqual(np.linalg.norm(shifted_q_values[:-1] - q_values[1:]), 0)
        self.assertEqual(np.linalg.norm(shifted_h_values[:-1] - h_values[1:]), 0)
