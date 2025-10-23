SHARED_ARGS="--replay_buffer_capacity 10_000 --batch_size 32 --update_horizon 1 --gamma 0.99 --horizon 1_000 \
  --n_epochs 10 --n_training_steps_per_epoch 10_000 --n_initial_samples 1_000 --epsilon_end 0.01 \
  --epsilon_duration 1_000 --architecture_type fc"

N_BELLMAN_ITERATIONS=5
TARGET_SYNC_FREQ=5
UPDATE_TO_DATA=1
FEATURES=32
WEIGHT_DECAY=1
DISABLE_WANDB=true

# np.logspace(np.log10(start), np.log10(stop), 10)
LEARNING_RATES=(5e-05 1.39e-04 3.87e-04 1.08e-03 3e-03 8.34e-03 2.32e-02 6.46e-02 1.8e-01 5e-01)
TARGET_UPDATE_FREQUENCIES=(10 20 40 79 158 316 630 1257 2507 5000)

PLATFORM="cluster/cluster"  # stud/cluster local/local

if [[ $DISABLE_WANDB = true ]]
then
    SHARED_ARGS="$SHARED_ARGS --disable_wandb"
fi

for lr in "${LEARNING_RATES[@]}"
do
  for tuf in "${TARGET_UPDATE_FREQUENCIES[@]}"
  do
    SHARED_NAME="sa25_utd${UPDATE_TO_DATA}_f${FEATURES}_wd${WEIGHT_DECAY}_tuf${tuf}_lr${lr}"
    EXPERIMENT_ARGS="$SHARED_ARGS --first_seed 11 --last_seed 40 --n_parallel_seeds 1  --features $FEATURES $FEATURES \
      --learning_rate $lr --target_update_frequency $tuf"
      
    launch_job/mountain_car/${PLATFORM}_dqn.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS
    sleep 10
    launch_job/mountain_car/${PLATFORM}_dqnrc.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --weight_decay $WEIGHT_DECAY
    sleep 10
    launch_job/mountain_car/${PLATFORM}_idqn.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --target_sync_frequency $TARGET_SYNC_FREQ
    sleep 10
    launch_job/mountain_car/${PLATFORM}_fidqn.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS
    sleep 10
    launch_job/mountain_car/${PLATFORM}_gidqn.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY
    sleep 10
    launch_job/mountain_car/${PLATFORM}_gidqn.sh --experiment_name unfrozen_$SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY --unfreeze_first_head
    sleep 10m
  done
done
