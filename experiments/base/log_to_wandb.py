import wandb
import json
import os
import numpy as np

ENV_NAME = "lunar_lander"
EXPERIMENT_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../{ENV_NAME}/exp_output")
ALGO_NAMES = ["dqn" , "dqnrc", "idqn", "fidqn", "gidqn", "gidqn_unfrozen"]
SEEDS = [1, 2, 3]
NUM_EPOCHS = 25
LRS = [
    "1e-4",
    "1e-1"
]
TUFS =["25", "1_000"] # np.int32(np.round(np.logspace(1, np.log10(5000), num=10)))


def get_experiment_name(algo_name, seed, tuf, lr):
    if algo_name == "gidqn_unfrozen":
        return f"unfrozen_test_new_pipeline3_tuf{tuf}_lr{lr}"
    else:
        return f"test_new_pipeline3_tuf{tuf}_lr{lr}"


def init_wandb(algo_name, seed):
    return wandb.init(
        project=f"Gi-DQN_grid",
        mode="online",
        config={"algo": algo_name},
        # TODO: possibly add good config
        name=str(seed),
        group=f"test_new_pipeline3_{algo_name}",
        settings=wandb.Settings(_disable_stats=True),
    )


def write_data_to_wandb():
    keys = ["epoch"] + [f"avg_return_{tuf}_{lr}" for tuf in TUFS for lr in LRS]
    for algo in ALGO_NAMES:
        for seed in SEEDS:
            wb = init_wandb(algo, seed)
            print(f"algo: {algo}, seed: {seed}")
            values = collect_experiment_results(algo, seed)
            for i in range(NUM_EPOCHS):
                entry = dict(zip(keys, values[i]))
                wb.log(entry)
            wb.finish()

def collect_experiment_results(algo_name, seed):
    data = [np.arange(NUM_EPOCHS)]
    for tuf in TUFS:
        for lr in LRS:
            returns_json = load_json_data(algo_name, seed, tuf, lr)["episode_returns"]
            avg_returns = [np.mean(episodic_returns) for episodic_returns in returns_json]
            data.append(avg_returns)
    data = np.stack(data)
    print(data)

    return data.T


def load_json_data(algo_name: str, seed, tuf, lr):
    experiment_name = get_experiment_name(algo_name, seed, tuf, lr)
    if algo_name == "gidqn_unfrozen":
        algo_name = "gidqn"
    file_path = os.path.join(
        EXPERIMENT_RESULTS_PATH, f"{experiment_name}/{algo_name}/episode_returns_and_lengths/{seed}.json"
    )
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data for {algo_name} with seed {seed}: {e}")

    return data


def check_experiments():
    for algo in ALGO_NAMES:
        for seed in SEEDS:
            for tuf in TUFS:
                for lr in LRS:
                    experiment_name = get_experiment_name(algo, seed, tuf, lr)
                    if algo == "gidqn_unfrozen":
                        algo = "gidqn"
                    file_path = os.path.join(
                        EXPERIMENT_RESULTS_PATH, f"{experiment_name}/{algo}/episode_returns_and_lengths/{seed}.json"
                    )
                    assert os.path.exists(file_path), f"File {file_path} does not exist."


if __name__ == "__main__":
    print("Checking experiments...")
    check_experiments()
    print("All experiments found. Writing to wandb...")
    write_data_to_wandb()
