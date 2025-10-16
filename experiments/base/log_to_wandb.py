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
                    if len(returns) < n_epochs:
                        print(
                            get_experiment_name(v1, v2),
                            f"{algo} seed {seed} was not finished. {n_epochs - len(returns)} epochs are missing.",
                        )

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
        print(f"Algorithm: {algo}", flush=True)
        for seed in seeds:
            print(f"seed: {seed}", flush=True)
            config = get_config(algo, seed)
            wb = wandb.init(
                project="Gi-DQN_grid",
                mode="online",
                config=config,
                name=str(seed),
                group=f"{algo}_{base_experiment_name}",
                settings=wandb.Settings(_disable_stats=True),
            )

            for epoch in range(n_epochs):
                wb.log({"epoch": epoch, **data[algo][seed][epoch]})
            wb.finish()


def get_config(algo, seed):
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../{env_name}/exp_output/{get_experiment_name(var1_options[0], var2_options[0])}/parameters.json",
    )
    config = json.load(open(config_path, "r"))

    if algo == "gidqn_unfrozen":
        algo = "gidqn"

    return config["shared_parameters"] | config[algo] | {"seed": seed, "algo_name": algo}


if __name__ == "__main__":
    # get_experiment_name = lambda var1, var2: f"sa25_utd1_f32_wd1_tuf{var1}_lr{var2}"
    get_experiment_name = lambda var1, var2: f"sa25_utd1_f{var1}_wd{var2}_tuf300_lr3e-03"
    env_name = "mountain_car"
    algorithms = ["dqnrc", "gidqn"] # ["dqn", "dqnrc", "idqn", "fidqn", "gidqn", "gidqn_unfrozen"]
    seeds = range(1, 11)
    # var1_options = ["10", "20", "40", "79", "158", "316", "630", "1257", "2507", "5000"]
    var1_options = ["5", "8", "12", "20", "31", "49", "77", "121", "190", "300"]
    # var2_options = ["5e-05", "1.39e-04", "3.87e-04", "1.08e-03", "3e-03", "8.34e-03", "2.32e-02", "6.46e-02", "1.8e-01", "5e-01"]
    var2_options = ["1e-05", "4.6e-05", "2.15e-04", "1e-03", "4.64e-03", "2.15e-02", "1e-01", "4.64e-01", "2.15e+00", "1e+01"]
    n_epochs = 10

    print("Checking experiments...", flush=True)
    data = collect_data(get_experiment_name, env_name, algorithms, seeds, var1_options, var2_options, n_epochs)
    print("All experiments found. Writing to wandb...", flush=True)
    write_data_to_wandb(get_experiment_name("X", "X"), data, algorithms, seeds, n_epochs)
