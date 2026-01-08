#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

echo "launch train $ALGO_NAME"

sbatch --job-name $EXPERIMENT_NAME-$ALGO_NAME --array=$FIRST_SEED-$LAST_SEED:$N_PARALLEL_SEEDS --cpus-per-task=4 --mem-per-cpu=$((N_PARALLEL_SEEDS * 4500 + 5500))M --time=48:00:00 --gres=gpu:1 --partition A100-40GB,A100-80GB,A100-PCI,B200,H100,H100-PCI,H200,H200-PCI,L40S,RTX3090,RTXA6000,V100-32GB,batch \
--output=experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_$FIRST_SEED-$LAST_SEED.out \
--container-mounts=/netscratch/$USER/Gi-DQN:/netscratch/$USER/Gi-DQN,/home/$USER/.netrc:/root/.netrc --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh --container-workdir=/netscratch/$USER/Gi-DQN \
launch_job/$ENV_NAME/pegasus/train.sh --algo_name $ALGO_NAME --env_name $ENV_NAME --experiment_name $EXPERIMENT_NAME $ARGS --n_parallel_seeds $N_PARALLEL_SEEDS