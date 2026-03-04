import json
import os
import shutil
import subprocess
import time

import numpy as np


def run_algorithm(algo_name, algo_args):
    print(f"\n\n\n--------------- Time {algo_name} ---------------", flush=True)
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"../../experiments/atari/exp_output/_time_{algo_name}_Breakout"
    )
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    time_begin = time.time()
    returncode = subprocess.run(
        f"python3 experiments/atari/{algo_name}.py --experiment_name _time_{algo_name}_Breakout {algo_args}".split(" ")
    ).returncode
    time_end = time.time()

    if returncode != 0:
        print(
            f"Training {algo_name} should not have raised an error. The training time has to be recomputed.", flush=True
        )
    else:
        print(f"{algo_name} trained in {np.around(time_end - time_begin)} seconds.", flush=True)

    shutil.rmtree(save_path)

    return time_end - time_begin if returncode == 0 else None


if __name__ == "__main__":
    base_args_cnn = (
        "--seed 1 --disable_wandb --replay_buffer_capacity 1_000_000 --batch_size 32 --gamma 0.99 --horizon 27_000 "
        + "--n_epochs 1 "  # reduce to 1
        + "--n_training_steps_per_epoch 250_000 "
        + "--n_initial_samples 32 "  # reduce to 32
        + "--epsilon_end 0.01 "
        + "--epsilon_duration 1 "  # reduce to 1
        + "--learning_rate 6.25e-5 --features 32 64 64 512 --target_update_period 8000 --architecture_type cnn --update_to_data 0.25 --update_horizon 1"
    )

    time_cnn_q = run_algorithm("dqn", base_args_cnn)
    time_cnn_qrc = run_algorithm("dqnrcshared", base_args_cnn + " --linear_heads --weight_decay 1")
    time_cnn_iq = run_algorithm("idqnshared", base_args_cnn + " --linear_heads --n_bellman_iterations 5")
    time_cnn_giq = run_algorithm(
        "gidqnshared", base_args_cnn + " --linear_heads --n_bellman_iterations 5 --weight_decay 1 --freeze_first_head"
    )

    results = {"cnn+dqn": {"q": time_cnn_q, "qrc": time_cnn_qrc, "iq": time_cnn_iq, "giq": time_cnn_giq}}

    base_args_impala = (
        "--seed 1 --disable_wandb --replay_buffer_capacity 1_000_000 --batch_size 32 --gamma 0.99 --horizon 27_000 "
        + "--n_epochs 1 "  # reduce to 1
        + "--n_training_steps_per_epoch 250_000 "
        + "--n_initial_samples 32 "  # reduce to 32
        + "--epsilon_end 0.01 "
        + "--epsilon_duration 1 "  # reduce to 1
        + "--learning_rate 6.25e-5 --features 16 32 32 512 --gap --per --target_update_period 8000 --architecture_type impala --update_to_data 0.25 --update_horizon 3"
    )

    time_impala_q = run_algorithm("dqn", base_args_impala)
    time_impala_qrc = run_algorithm("dqnrcshared", base_args_impala + " --linear_heads --weight_decay 1")
    time_impala_iq = run_algorithm("idqnshared", base_args_impala + " --linear_heads --n_bellman_iterations 5")
    time_impala_giq = run_algorithm(
        "gidqnshared",
        base_args_impala + " --linear_heads --n_bellman_iterations 5 --weight_decay 1 --freeze_first_head",
    )

    results.update(
        {"impala+dqn": {"q": time_impala_q, "qrc": time_impala_qrc, "iq": time_impala_iq, "giq": time_impala_giq}}
    )

    json.dump(results, open("tests/flop_comparison/time_algorithms.json", "w"), indent=4)
