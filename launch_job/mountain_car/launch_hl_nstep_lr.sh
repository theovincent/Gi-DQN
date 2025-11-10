SHARED_ARGS="--replay_buffer_capacity 10_000 --batch_size 32 --gamma 0.99 --horizon 1_000 \
  --n_epochs 10 --n_training_steps_per_epoch 5_000 --n_initial_samples 1_000 --epsilon_end 0.01 \
  --epsilon_duration 1_000 --architecture_type fc"

TARGET_SYNC_FREQ=5
TARGET_UPDATE_FREQUENCIES=300
FEATURE_SIZE=100
DISABLE_WANDB=true
N_BELLMAN_ITERATIONS=5
WEIGHT_DECAY=1
MU=1
UPDATE_TO_DATA=1
SIGMA=5
MIN_VALUE=-150
MAX_VALUE=50
N_BINS=51

# np.logspace(np.log10(start), np.log10(stop), 5)
UPDATE_HORIZON=(1 2 3 4 6 9 14 21 32 50)
LEARNING_RATES=(5e-05 1.39e-04 3.87e-04 1.08e-03 3e-03 8.34e-03 2.32e-02 6.46e-02 1.8e-01 5e-01)


PLATFORM="stud/cluster"  # stud/cluster local/local

if [[ $DISABLE_WANDB = true ]]
then
    SHARED_ARGS="$SHARED_ARGS --disable_wandb"
fi

for horizon in "${UPDATE_HORIZON[@]}"
do
  for lr in "${LEARNING_RATES[@]}"
  do
    SHARED_NAME="sa25_nstep${horizon}_lr${lr}_utd${UPDATE_TO_DATA}_f${FEATURE_SIZE}_wd${WEIGHT_DECAY}_tuf${TARGET_UPDATE_FREQUENCIES}_nbi${N_BELLMAN_ITERATIONS}_distributional"
    EXPERIMENT_ARGS="$SHARED_ARGS --first_seed 1 --last_seed 10 --n_parallel_seeds 1 --features $FEATURE_SIZE $FEATURE_SIZE \
      --learning_rate $lr --update_horizon $horizon --target_update_frequency $TARGET_UPDATE_FREQUENCIES --update_to_data $UPDATE_TO_DATA --sigma $SIGMA --n_bins $N_BINS --min_value $MIN_VALUE --max_value $MAX_VALUE"

    launch_job/mountain_car/${PLATFORM}_hldqn.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS
    sleep 2
    launch_job/mountain_car/${PLATFORM}_hldqnrcshared.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --weight_decay $WEIGHT_DECAY --mu $MU
    sleep 2
    launch_job/mountain_car/${PLATFORM}_hlidqnshared.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --target_sync_frequency $TARGET_SYNC_FREQ
    sleep 2
    launch_job/mountain_car/${PLATFORM}_hlfidqnshared.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS
    sleep 2
    launch_job/mountain_car/${PLATFORM}_hlgidqnshared.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY --mu $MU
    sleep 5m
  done
done

