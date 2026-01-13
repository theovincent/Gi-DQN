import unittest
import numpy as np
import jax
import jax.numpy as jnp

from slimdqn.algorithms.gidqn import GiDQN, shift_params
from tests.utils import Generator


class TestGiDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        key_actions, key_feature_1, key_feature_2, key_feature_3, key_feature_4 = jax.random.split(self.key, 5)
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.q = GiDQN(
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

        computed_loss = self.q.loss(self.q.params, self.q.h_params, sample, jnp.ones(1))[0].sum()

        targets = jax.vmap(self.q.compute_target, in_axes=(0, None))(self.q.params, sample)[:-1]
        q_predictions = jax.vmap(self.q.network.apply, in_axes=(0, None))(self.q.params, sample.state)[
            1:, sample.action
        ]
        h_predictions = jax.vmap(self.q.network.apply, in_axes=(0, None))(self.q.h_params, sample.state)[
            :, sample.action
        ]
        td_loss = (targets[1:] * h_predictions).sum() - (q_predictions * (targets - q_predictions)).sum()
        h_loss = -(h_predictions * (targets[1:] - q_predictions[1:] - h_predictions)).sum()

        self.assertAlmostEqual(td_loss + h_loss, computed_loss, places=4)

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state)

        q_values = jax.vmap(self.q.network.apply, in_axes=(0, None))(self.q.params, state)[1:].mean(axis=0)
        best_action = jnp.argmax(q_values)

        self.assertEqual(q_values.shape, (self.n_actions,))
        self.assertEqual(best_action, computed_best_action)

    def test_target_update(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        q_values = jax.vmap(self.q.network.apply, in_axes=(0, None))(self.q.params, state)
        shifted_params = shift_params(self.q.params)
        shifted_q_values = jax.vmap(self.q.network.apply, in_axes=(0, None))(shifted_params, state)

        self.assertAlmostEqual(np.linalg.norm(shifted_q_values[:-1] - q_values[1:]), 0, places=5)
