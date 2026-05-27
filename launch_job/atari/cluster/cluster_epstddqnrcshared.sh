#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

echo "launch train $ALGO_NAME"

sbatch --job-name $EXPERIMENT_NAME-$ALGO_NAME --array=$FIRST_SEED-$LAST_SEED:$N_PARALLEL_SEEDS --cpus-per-task=1 --mem-per-cpu=$((N_PARALLEL_SEEDS * 1000))M --time=24:00:00 --partition stud \
--output=experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_$FIRST_SEED-$LAST_SEED.out \
launch_job/$ENV_NAME/train.sh --algo_name $ALGO_NAME --env_name $ENV_NAME --experiment_name $EXPERIMENT_NAME $ARGS --n_parallel_seeds $N_PARALLEL_SEEDS
