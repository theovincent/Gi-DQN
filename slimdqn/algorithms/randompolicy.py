import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from slimdqn.sample_collection.replay_buffer import ReplayBuffer


class RandomPolicy:
    def __init__(self, key: jax.random.PRNGKey, n_actions: int):
        self.n_actions = int(n_actions)
        self.key = key

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        return None

    def update_target_params(self, step: int):
        self.logs = {"n_training_steps": step, "loss": 0.0}

    def best_action(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        self.key, subkey = jax.random.split(self.key)
        return jax.random.randint(subkey, shape=(), minval=0, maxval=self.n_actions)

    def get_model(self):
        return {"params": []}
