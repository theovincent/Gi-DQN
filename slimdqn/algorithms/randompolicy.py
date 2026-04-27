from functools import partial

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from slimdqn.sample_collection.replay_buffer import ReplayBuffer


class RandomPolicy:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions: int,
        **kwargs,
    ):
        self.n_actions = int(n_actions)
        self.key = key

        self.params: FrozenDict = FrozenDict({})
        self.target_params: FrozenDict = FrozenDict({})

        self.cumulative_loss = 0.0
        self.logs = {"n_training_steps": 0, "loss": 0.0}

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        return None

    def update_target_params(self, step: int):
        self.logs = {"n_training_steps": step, "loss": 0.0}
        return None

    def best_action(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        self.key, subkey = jax.random.split(self.key)
        return jax.random.randint(subkey, shape=(), minval=0, maxval=self.n_actions)

    def get_model(self):
        return {"params": self.params}

