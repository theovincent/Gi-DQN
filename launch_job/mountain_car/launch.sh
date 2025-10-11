SHARED_ARGS="--features 32 32 --replay_buffer_capacity 4_000 --batch_size 32 --update_horizon 1 --gamma 0.99 \
  --horizon 1_000 --n_epochs 10 --n_training_steps_per_epoch 10_000 --update_to_data 1 --n_initial_samples 1_000 \
  --epsilon_end 0.01 --epsilon_duration 1_000 -at fc"

N_BELLMAN_ITERATIONS=5
TARGET_SYNC_FREQ=5
WEIGHT_DECAY=1

# np.logspace(np.log10(start), np.log10(stop), 10)
LEARNING_RATES=(1e-06 4.30e-06 1.85e-05 7.94e-05 3.41e-04 1.47e-03 6.30e-03 2.71e-02 1.16e-01 5e-01)
TARGET_UPDATE_FREQUENCIES=(10 20 40 79 158 316 630 1257 2507 5000)

PLATFORM="cluster/cluster"  # stud/cluster local/local

for lr in "${LEARNING_RATES[@]}"
do
  for tuf in "${TARGET_UPDATE_FREQUENCIES[@]}"
  do
    SHARED_NAME="sa25_wd${WEIGHT_DECAY}_tuf${tuf}_lr${lr}"
    SHARED_ARGS="$SHARED_ARGS --first_seed 1 --last_seed 1 --n_parallel_seeds 1 --learning_rate $lr"

    launch_job/mountain_car/${PLATFORM}_dqn.sh --experiment_name $SHARED_NAME $SHARED_ARGS
    launch_job/mountain_car/${PLATFORM}_dqnrc.sh --experiment_name $SHARED_NAME $SHARED_ARGS --weight_decay $WEIGHT_DECAY
    launch_job/mountain_car/${PLATFORM}_idqn.sh --experiment_name $SHARED_NAME $SHARED_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --target_sync_frequency $TARGET_SYNC_FREQ
    launch_job/mountain_car/${PLATFORM}_fidqn.sh --experiment_name $SHARED_NAME $SHARED_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS
    launch_job/mountain_car/${PLATFORM}_gidqn.sh --experiment_name $SHARED_NAME $SHARED_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY
    launch_job/mountain_car/${PLATFORM}_gidqn.sh --experiment_name unfrozen_$SHARED_NAME $SHARED_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY --unfreeze_first_head
  done
done