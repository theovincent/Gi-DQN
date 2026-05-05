#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@ --first_seed dummy --last_seed dummy
FIRST_SEED=$SLURM_ARRAY_TASK_ID 
LAST_SEED=$((N_PARALLEL_SEEDS + SLURM_ARRAY_TASK_ID - 1))

source env/bin/activate
#export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(echo "scale=2 ; 0.98 / ($LAST_SEED - $FIRST_SEED + 1)" | bc)

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    python3 experiments/$ENV_NAME/$ALGO_NAME.py --experiment_name $EXPERIMENT_NAME --seed $seed $ARGS &> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_$seed.out & 
done
wait
#timemout 900 wandb sync --sync-all --include-offline || true  #only sync per-task after all seeds finished
