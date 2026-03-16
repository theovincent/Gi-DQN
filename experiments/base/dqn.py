import jax
import numpy as np
import optax
from tqdm import tqdm

from experiments.base.utils import save_data
from slimdqn.algorithms.dqn import DQN
from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.sample_collection.utils import collect_single_sample

import time, resource


def train(key: jax.random.PRNGKey, p: dict, agent: DQN, env, rb: ReplayBuffer):
    epsilon_schedule = optax.linear_schedule(1.0, p["epsilon_end"], p["epsilon_duration"])

    n_training_steps = 0
    env.reset()
    episode_returns_per_epoch = [[0]]
    episode_lengths_per_epoch = [[0]]

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(agent.params))
    print(f"Agent param count: {param_count:,}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
    peak_ram = 0
    for idx_epoch in tqdm(range(p["n_epochs"])):
        n_training_steps_epoch = 0
        has_reset = False
        start_epoch = time.time()
        while n_training_steps_epoch < p["n_training_steps_per_epoch"] or not has_reset:
            key, exploration_key = jax.random.split(key)
            reward, has_reset = collect_single_sample(
                exploration_key, env, agent, rb, p, epsilon_schedule, n_training_steps
            )

            n_training_steps_epoch += 1
            n_training_steps += 1

            episode_returns_per_epoch[idx_epoch][-1] += reward
            episode_lengths_per_epoch[idx_epoch][-1] += 1
            if has_reset and n_training_steps_epoch < p["n_training_steps_per_epoch"]:
                episode_returns_per_epoch[idx_epoch].append(0)
                episode_lengths_per_epoch[idx_epoch].append(0)

            if n_training_steps > p["n_initial_samples"]:
                agent.update_online_params(n_training_steps, rb)
                agent.update_target_params(n_training_steps)

                if n_training_steps % 64_000 == 0:
                    p["wandb"].log(agent.logs)

        end_epoch = time.time()
        print(f"Runtime Epoch {idx_epoch}: {(end_epoch - start_epoch) / 60:.4f} minutes")
        peak_ram_episode = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        peak_ram = peak_ram_episode if peak_ram_episode > peak_ram else peak_ram

        avg_return = np.mean(episode_returns_per_epoch[idx_epoch])
        avg_length_episode = np.mean(episode_lengths_per_epoch[idx_epoch])
        n_episodes = len(episode_lengths_per_epoch[idx_epoch])
        print(f"\nEpoch {idx_epoch}: Return {avg_return} averaged on {n_episodes} episodes.\n", flush=True)
        p["wandb"].log(
            {
                "epoch": idx_epoch,
                "n_training_steps": n_training_steps,
                "avg_return": avg_return,
                "avg_length_episode": avg_length_episode,
            }
        )

        if idx_epoch < p["n_epochs"] - 1:
            episode_returns_per_epoch.append([0])
            episode_lengths_per_epoch.append([0])
        print(f"Peak RAM: {peak_ram:.2f} MB")
        save_data(p, episode_returns_per_epoch, episode_lengths_per_epoch, agent.get_model())
