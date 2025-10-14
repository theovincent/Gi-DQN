SHARED_ARGS="--features 100 100 --replay_buffer_capacity 10000 --batch_size 32 --update_horizon 1 --gamma 0.99 \
    --horizon 1_000 --n_epochs 25 --n_training_steps_per_epoch 10_000 --update_to_data 1 --n_initial_samples 1_000 \
    --epsilon_end 0.01 --epsilon_duration 1_000 -at fc"

WIND_POWER=15.0
TURB_POWER=1.5
N_BELLMAN_ITERATIONS=5
TARGET_SYNC_FREQ=5
WEIGHT_DECAY=0.001
DISABLE_WANDB=true

PLATFORM="stud/cluster"  # stud/cluster local/local


if [[ $DISABLE_WANDB = true ]]
then
    SHARED_ARGS="$SHARED_ARGS --disable_wandb"
fi

for lr in 1e-4  1e-1
do
  for tuf in 25 1_000
  do
    SHARED_NAME="test_new_pipeline3_tuf${tuf}_lr${lr}"
    SHARED_ARGS="$SHARED_ARGS --first_seed 1 --last_seed 3 --n_parallel_seeds 1 --learning_rate $lr \
      --target_update_frequency $tuf --wind_and_turbulence_power $WIND_POWER $TURB_POWER"

    launch_job/lunar_lander/${PLATFORM}_dqn.sh --experiment_name $SHARED_NAME $SHARED_ARGS
    sleep 2
    launch_job/lunar_lander/${PLATFORM}_dqnrc.sh --experiment_name $SHARED_NAME $SHARED_ARGS --weight_decay $WEIGHT_DECAY
    sleep 2
    launch_job/lunar_lander/${PLATFORM}_idqn.sh --experiment_name $SHARED_NAME $SHARED_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --target_sync_frequency $TARGET_SYNC_FREQ
    sleep 2
    launch_job/lunar_lander/${PLATFORM}_fidqn.sh --experiment_name $SHARED_NAME $SHARED_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS
    sleep 2
    launch_job/lunar_lander/${PLATFORM}_gidqn.sh --experiment_name $SHARED_NAME $SHARED_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY
    sleep 2
    launch_job/lunar_lander/${PLATFORM}_gidqn.sh --experiment_name unfrozen_$SHARED_NAME $SHARED_ARGS --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY --unfreeze_first_head
    sleep 5m
  done
done



