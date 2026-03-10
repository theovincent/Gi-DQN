#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

echo "launch train $ALGO_NAME"

sbatch --job-name $EXPERIMENT_NAME-$ALGO_NAME --array=$FIRST_SEED-$LAST_SEED:$N_PARALLEL_SEEDS --cpus-per-task=4 --mem-per-cpu=$((N_PARALLEL_SEEDS * 1000))M--time=72:00:00 --gres=gpu:1 --prefer="rtx3090|a5000" --exclude="dgx-station" --partition gpu \
--output=experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_$FIRST_SEED-$LAST_SEED.out \
launch_job/$ENV_NAME/train.sh --algo_name $ALGO_NAME --env_name $ENV_NAME --experiment_name $EXPERIMENT_NAME $ARGS --n_parallel_seeds $N_PARALLEL_SEEDS

