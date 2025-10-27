import jax
import time
import numpy as np

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.environments.atari import AtariEnv

env = AtariEnv("Breakout")

key = jax.random.key(0)
key1, key2, key3 = jax.random.split(key, 3)
env.reset()
x = env.state_


model_one = DQNNet([32, 64, 64, 512], "cnn", 4, 6)
params_one = model_one.init(key1, x)
f_one = jax.jit(lambda x: model_one.apply(params_one, x))

model_tuple = DQNNet([32, 64, 64, 512], "gi", 4, 6)
params_tuple = model_tuple.init(key2, x)
f_tuple = jax.jit(lambda x: model_tuple.apply(params_tuple, x))


def measure_time(key, name, fn, x, n=100):

    out = fn(x)
    jax.tree_util.tree_map(lambda o: o.block_until_ready(), out)

    times = []
    for i in range(n):
        key, key_used = jax.random.split(key)
        obs = jax.random.normal(key_used, x.shape)
        t0 = time.time()
        y = jax.block_until_ready(fn(obs))
        t1 = time.time()
        times.append(t1 - t0)
    print(f"{name:25s}: {np.sum(times)*1000:.3f} ms", flush=True)


print("=== Forward-pass benchmark ===", flush=True)
measure_time(key3, "One big head", f_one, x)
measure_time(key3, "Two heads (tuple)", f_tuple, x)
