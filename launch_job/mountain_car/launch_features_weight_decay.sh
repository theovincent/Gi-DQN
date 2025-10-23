SHARED_ARGS="--replay_buffer_capacity 10_000 --batch_size 32 --update_horizon 1 --gamma 0.99 --horizon 1_000 \
  --n_epochs 10 --n_training_steps_per_epoch 10_000 --n_initial_samples 1_000 --epsilon_end 0.01 \
  --epsilon_duration 1_000 --architecture_type fc"

N_BELLMAN_ITERATIONS=5
TARGET_SYNC_FREQ=5
UPDATE_TO_DATA=1
TARGET_UPDATE_FREQUENCIES=300
LEARNING_RATES=3e-03
DISABLE_WANDB=true

# np.logspace(np.log10(start), np.log10(stop), 10)
FEATURES=(5 8 12 20 31 49 77 121 190 300)
WEIGHT_DECAY=(1e-05 4.6e-05 2.15e-04 1e-03 4.64e-03 2.15e-02 1e-01 4.64e-01 2.15e+00 1e+01)


PLATFORM="cluster/cluster"  # stud/cluster local/local

if [[ $DISABLE_WANDB = true ]]
then
    SHARED_ARGS="$SHARED_ARGS --disable_wandb"
fi

for features in "${FEATURES[@]}"
do
  for weight_decay in "${WEIGHT_DECAY[@]}"
  do
    SHARED_NAME="sa25_utd${UPDATE_TO_DATA}_f${features}_wd${weight_decay}_tuf${TARGET_UPDATE_FREQUENCIES}_lr${LEARNING_RATES}"
    EXPERIMENT_ARGS="$SHARED_ARGS --first_seed 1 --last_seed 10 --n_parallel_seeds 1 --features $features $features \
      --learning_rate $LEARNING_RATES --target_update_frequency $TARGET_UPDATE_FREQUENCIES"

    launch_job/mountain_car/${PLATFORM}_dqn.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS
    sleep 2
    launch_job/mountain_car/${PLATFORM}_dqnrc.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --weight_decay $weight_decay
    sleep 2
    launch_job/mountain_car/${PLATFORM}_idqn.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --target_sync_frequency $TARGET_SYNC_FREQ
    sleep 2
    launch_job/mountain_car/${PLATFORM}_fidqn.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS
    sleep 2
    launch_job/mountain_car/${PLATFORM}_gidqn.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $weight_decay
    sleep 2
    launch_job/mountain_car/${PLATFORM}_gidqn.sh --experiment_name unfrozen_$SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $weight_decay --unfreeze_first_head
    sleep 5m
  done
done