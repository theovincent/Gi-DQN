#!/bin/bash


EXPERIMENT_PATH=$1
ALGO_NAME=$2
N_EVAL_EPISODES=$3
FPS=$4

source env/bin/activate

python3 -m experiments.atari.evaluate \
    --experiment_path "${EXPERIMENT_PATH}" \
    --algo_name "${ALGO_NAME}" \
    --seed "${SLURM_ARRAY_TASK_ID}" \
    --n_eval_episodes "${N_EVAL_EPISODES}" \
    --fps "${FPS}"

