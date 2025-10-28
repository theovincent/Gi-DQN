import os
import sys

import jax

from experiments.base.dqn import train
from experiments.base.utils import prepare_logs
from slimdqn.environments.mountain_car import MountainCar
from slimdqn.networks.hlgidqnshared import HLGiDQNShared
from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.sample_collection.samplers import Uniform


def run(argvs=sys.argv[1:]):
    env_name, algo_name = os.path.abspath(__file__).split("/")[-2], os.path.abspath(__file__).split("/")[-1][:-3]
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, train_key, env_key = jax.random.split(jax.random.PRNGKey(p["seed"]), 3)

    env = MountainCar(env_key)
    rb = ReplayBuffer(
        sampling_distribution=Uniform(p["seed"]),
        batch_size=p["batch_size"],
        max_capacity=p["replay_buffer_capacity"],
        stack_size=1,
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
        clipping=None,
    )
    agent = HLGiDQNShared(
        q_key,
        env.observation_shape[0],
        env.n_actions,
        n_bellman_iterations=p["n_bellman_iterations"],
        features=p["features"],
        architecture_type=p["architecture_type"],
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        update_to_data=p["update_to_data"],
        unfreeze_first_head=p["unfreeze_first_head"],
        target_update_frequency=p["target_update_frequency"],
        weight_decay=p["weight_decay"],
        mu=p["mu"],
        n_bins=p["n_bins"],
        min_value=p["min_value"],
        max_value=p["max_value"],
        sigma=p["sigma"],
    )
    train(train_key, p, agent, env, rb)


if __name__ == "__main__":
    run()
