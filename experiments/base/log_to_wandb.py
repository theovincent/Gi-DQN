import wandb
import json
import os
import numpy as np


def collect_data(get_experiment_name, env_name, algorithms, seeds, var1_options, var2_options, n_epochs):
    data = {}
    for algo in algorithms:
        data[algo] = {}
        for seed in seeds:
            data[algo][seed] = [{} for _ in range(n_epochs)]
            for v1 in var1_options:
                for v2 in var2_options:
                    returns = load_json_data(get_experiment_name, env_name, algo, seed, v1, v2)["episode_returns"]
                    for epoch in range(n_epochs):
                        data[algo][seed][epoch][f"avg_return_{v1}_{v2}"] = np.mean(returns[epoch])
    return data


def load_json_data(get_experiment_name, env_name, algo, seed, v1, v2):
    experiment_name = get_experiment_name(v1, v2)
    if algo == "gidqn_unfrozen":
        experiment_name = "unfrozen_" + experiment_name
        algo = "gidqn"

    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../{env_name}/exp_output/{experiment_name}/{algo}/episode_returns_and_lengths/{seed}.json",
    )

    return json.load(open(file_path, "r"))


def write_data_to_wandb(base_experiment_name, data, algorithms, seeds, n_epochs):
    for algo in algorithms:
        print(f"Algorithm: {algo}")
        for seed in seeds:
            print(f"seed: {seed}")
            wb = wandb.init(
                project="Gi-DQN_grid",
                mode="online",
                config={"algo": algo, "seed": seed},
                name=str(seed),
                group=f"{base_experiment_name}_{algo}",
                settings=wandb.Settings(_disable_stats=True),
            )

            for epoch in range(n_epochs):
                wb.log({"epoch": epoch, **data[algo][seed][epoch]})
            wb.finish()


if __name__ == "__main__":
    base_experiment_name = "test_new_pipeline3"
    get_experiment_name = lambda var1, var2: f"{base_experiment_name}_tuf{var1}_lr{var2}"
    env_name = "lunar_lander"
    algorithms = ["dqn", "dqnrc", "idqn", "fidqn", "gidqn", "gidqn_unfrozen"]
    seeds = range(1, 4)
    var1_options = ["25", "1_000"]
    var2_options = ["1e-4", "1e-1"]
    n_epochs = 25

    print("Checking experiments...")
    data = collect_data(get_experiment_name, env_name, algorithms, seeds, var1_options, var2_options, n_epochs)
    print("All experiments found. Writing to wandb...")
    write_data_to_wandb(base_experiment_name, data, algorithms, seeds, n_epochs)
