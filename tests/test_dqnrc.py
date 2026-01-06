import unittest
import numpy as np
import jax
import jax.numpy as jnp

from slimdqn.algorithms.dqnrc import DQNRC
from tests.utils import Generator


class TestDQNRC(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        key_actions, key_feature_1, key_feature_2, key_feature_3, key_feature_4 = jax.random.split(self.key, 5)
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.q = DQNRC(
            self.key,
            self.observation_dim,
            self.n_actions,
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
            1,
            1,
        )

        self.generator = Generator(None, self.observation_dim, self.n_actions)

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)

        computed_loss = self.q.loss(self.q.params, self.q.h_params, sample)[0]

        target = self.q.compute_target(self.q.params, sample)
        q_prediction = self.q.network.apply(self.q.params, sample.state)[sample.action]
        h_prediction = self.q.network.apply(self.q.h_params, sample.state)[sample.action]
        td_loss = target * h_prediction - q_prediction * (target - q_prediction)
        h_loss = -h_prediction * (target - q_prediction - h_prediction)

        self.assertEqual(td_loss + h_loss, computed_loss)

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state)

        q_values = self.q.network.apply(self.q.params, state)
        best_action = jnp.argmax(q_values)

        self.assertEqual(q_values.shape, (self.n_actions,))
        self.assertEqual(best_action, computed_best_action)
