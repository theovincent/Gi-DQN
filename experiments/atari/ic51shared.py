import os
import sys

import jax
import numpy as np

from experiments.base.dqn import train
from experiments.base.utils import prepare_logs
from slimdqn.environments.atari import AtariEnv
from slimdqn.algorithms.ic51shared import ic51Shared
from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.sample_collection.samplers import Uniform, Prioritized


def run(argvs=sys.argv[1:]):
    env_name, algo_name = os.path.abspath(__file__).split("/")[-2], os.path.abspath(__file__).split("/")[-1][:-3]
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = AtariEnv(
        name=p["experiment_name"].split("_")[-1],
        state_height_width=(p["pixels"], p["pixels"]),
        n_stacked_frames=p["n_frame_stack"],
        n_skipped_frames=p["n_frame_skip"],
    )
    rb = ReplayBuffer(
        sampling_distribution=Prioritized(p["seed"], p["replay_buffer_capacity"]) if p["per"] else Uniform(p["seed"]),
        max_capacity=p["replay_buffer_capacity"],
        batch_size=p["batch_size"],
        stack_size=env.n_stacked_frames,
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
        clipping=lambda x: np.clip(x, -1, 1),
    )
    agent = ic51Shared(
        q_key,
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        n_bellman_iterations=p["n_bellman_iterations"],
        features=p["features"],
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        update_to_data=p["update_to_data"],
        target_update_period=p["target_update_period"],
        adam_eps=1.5e-4,
        n_bins=p["number_of_bins"],
        vmin=-10,
        vmax=10,
    )
    train(train_key, p, agent, env, rb)


if __name__ == "__main__":
    run()
