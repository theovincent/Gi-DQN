SHARED_ARGS="--features 100 100 --replay_buffer_capacity 10000 --batch_size 32 --update_horizon 1 --gamma 0.99 \
    --horizon 1_000 --n_epochs 10 --n_training_steps_per_epoch 10_000 --update_to_data 1 --n_initial_samples 1_000 \
    --epsilon_end 0.01 --epsilon_duration 1_000 -at fc"

N_BELLMAN_ITERATIONS=5
TARGET_SYNC_FREQ=5
WEIGHT_DECAY=0.01

PLATFORM="cluster/cluster"  # stud/cluster local/local

for lr in 1e-4 1e-1
do
  for tuf in 25 1_000
  do
    SHARED_NAME="tuf${tuf}_lr${lr}"
    SHARED_ARGS="$SHARED_ARGS --first_seed 1 --last_seed 1 --n_parallel_seeds 1 --learning_rate $lr"

    launch_job/mountain_car/cluster_dqn.sh --experiment_name $SHARED_NAME $SHARED_ARGS
    launch_job/mountain_car/cluster_dqnrc.sh --experiment_name $SHARED_NAME $SHARED_ARGS --weight_decay $WEIGHT_DECAY
    launch_job/mountain_car/cluster_idqn.sh --experiment_name $SHARED_NAME $SHARED_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --target_sync_frequency $TARGET_SYNC_FREQ
    launch_job/mountain_car/cluster_fidqn.sh --experiment_name $SHARED_NAME $SHARED_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS
    launch_job/mountain_car/cluster_gidqn.sh --experiment_name $SHARED_NAME $SHARED_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY
    launch_job/mountain_car/cluster_gidqn.sh --experiment_name unfrozen_$SHARED_NAME $SHARED_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY --unfreeze_first_head
  done
done