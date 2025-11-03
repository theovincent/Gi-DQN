SHARED_ARGS="--replay_buffer_capacity 10_000 --batch_size 32 --gamma 0.99 --horizon 1_000 \
  --n_epochs 10 --n_training_steps_per_epoch 5_000 --n_initial_samples 1_000 --epsilon_end 0.01 \
  --epsilon_duration 1_000 --architecture_type fc"

TARGET_SYNC_FREQ=5
TARGET_UPDATE_FREQUENCIES=300
LEARNING_RATES=3e-03
DISABLE_WANDB=true
WEIGHT_DECAY=1
UPDATE_TO_DATA=1
FEATURES=100
MU=1

# np.logspace(np.log10(start), np.log10(stop), 5)
UPDATE_HORIZON=(6) #(1 2 3 4 6 9 14 21 32 50)
N_BELLMAN_ITERATIONS=(2 3 4 6 8 12 17 24 35 50)


PLATFORM="stud/cluster"  # stud/cluster local/local

if [[ $DISABLE_WANDB = true ]]
then
    SHARED_ARGS="$SHARED_ARGS --disable_wandb"
fi

for update_horizon in "${UPDATE_HORIZON[@]}"
do
  SHARED_NAME="sa25_nstep${update_horizon}_utd${UPDATE_TO_DATA}_f${FEATURES}_wd${WEIGHT_DECAY}_tuf${TARGET_UPDATE_FREQUENCIES}_lr${LEARNING_RATES}"
  EXPERIMENT_ARGS="$SHARED_ARGS --first_seed 1 --last_seed 10 --n_parallel_seeds 1 --features $FEATURES $FEATURES \
      --learning_rate $LEARNING_RATES --update_horizon $update_horizon --target_update_frequency $TARGET_UPDATE_FREQUENCIES --update_to_data $UPDATE_TO_DATA"
  #launch_job/mountain_car/${PLATFORM}_dqn.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS
  #sleep 2
  #launch_job/mountain_car/${PLATFORM}_dqnrcshared.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --weight_decay $WEIGHT_DECAY --mu $MU
  #sleep 2

  for n_bellman_iterations in "${N_BELLMAN_ITERATIONS[@]}"
  do
    SHARED_NAME="sa25_nstep${update_horizon}_utd${UPDATE_TO_DATA}_f${FEATURES}_wd${WEIGHT_DECAY}_tuf${TARGET_UPDATE_FREQUENCIES}_lr${LEARNING_RATES}_nbi${n_bellman_iterations}_sanity_check"
    launch_job/mountain_car/${PLATFORM}_idqnshared.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $n_bellman_iterations --target_sync_frequency $TARGET_SYNC_FREQ
    sleep 2
    #launch_job/mountain_car/${PLATFORM}_fidqnshared.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $n_bellman_iterations
    #sleep 2
    launch_job/mountain_car/${PLATFORM}_gidqnshared.sh --experiment_name $SHARED_NAME $EXPERIMENT_ARGS --n_bellman_iterations $n_bellman_iterations --weight_decay $WEIGHT_DECAY --mu $MU
    sleep 5m
  done
done
