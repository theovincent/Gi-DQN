SHARED_ARGS="--replay_buffer_capacity 10_000 --batch_size 32 --update_horizon 1 --gamma 0.99 --horizon 1_000 \
  --n_epochs 10 --n_training_steps_per_epoch 5_000 --n_initial_samples 1_000 --epsilon_end 0.01 \
  --epsilon_duration 1_000 --architecture_type fc"

TARGET_SYNC_FREQ=5
TARGET_UPDATE_FREQUENCIES=300
FEATURE_SIZE=100
DISABLE_WANDB=true
N_BELLMAN_ITERATIONS=5
WEIGHT_DECAY=1
MU=1

# np.logspace(np.log10(start), np.log10(stop), 5)
UPDATE_TO_DATA=(1 3 5 8 10)
LEARNING_RATES=(5e-05 1.39e-04 3.87e-04 1.08e-03 3e-03 8.34e-03 2.32e-02 6.46e-02 1.8e-01 5e-01)


PLATFORM="stud/cluster"  # stud/cluster local/local

if [[ $DISABLE_WANDB = true ]]
then
    SHARED_ARGS="$SHARED_ARGS --disable_wandb"
fi

for utd in "${UPDATE_TO_DATA[@]}" 
do
  for lr in "${LEARNING_RATES[@]}"
  do
    SHARED_NAME="sa25_utd${utd}_lr${lr}_f${FEATURE_SIZE}_wd${WEIGHT_DECAY}_tuf${TARGET_UPDATE_FREQUENCIES}_nbi${N_BELLMAN_ITERATIONS}"
    EXPERIMENT_ARGS="$SHARED_ARGS --first_seed 1 --last_seed 10 --n_parallel_seeds 1 --features $FEATURE_SIZE $FEATURE_SIZE \
      --learning_rate $lr --target_update_frequency $TARGET_UPDATE_FREQUENCIES --update_to_data $utd"

    launch_job/mountain_car/${PLATFORM}_dqn.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS
    sleep 2
    launch_job/mountain_car/${PLATFORM}_dqnrc.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --weight_decay $WEIGHT_DECAY 
    sleep 2
    launch_job/mountain_car/${PLATFORM}_idqn.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --target_sync_frequency $TARGET_SYNC_FREQ
    sleep 2
    launch_job/mountain_car/${PLATFORM}_fidqn.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS
    sleep 2
    launch_job/mountain_car/${PLATFORM}_gidqn.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY 
    sleep 5m
  done
done
