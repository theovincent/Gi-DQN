SHARED_ARGS="--features 16 256 --replay_buffer_capacity 50_000 --batch_size 16 --update_horizon 1 \
    --gamma 0.99 --horizon 10_000 --n_epochs 30 --n_training_steps_per_epoch 50_000 --update_to_data 1 \
    --n_initial_samples 10_000 --epsilon_end 0.01 --epsilon_duration 100_000 --pixels 16 --n_frame_stack 2 \
    --n_frame_skip 4"

# SHARED_ARGS="$SHARED_ARGS --disable_wandb"
N_BELLMAN_ITERATIONS=5
WEIGHT_DECAY=1

PLATFORM="local/local"  # cluster/cluster lichtenberg/cluster local/local


TARGET_UPDATE_PERIOD_DQN=3000
LEARNING_RATE_DQN=10e-5

TARGET_UPDATE_PERIOD_IDQN=10000
LEARNING_RATE_IDQN=25e-5

TARGET_UPDATE_PERIOD_GIDQN=10000
LEARNING_RATE_GIDQN=10e-5

for GAME in Gopher YarsRevenge Enduro Asterix Alien
do
    for OMEGA in 0.01 0.1 1 10 100
    do
          launch_job/atari/${PLATFORM}_mmdqn.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS \
          --learning_rate $LEARNING_RATE_DQN --target_update_period $TARGET_UPDATE_PERIOD_DQN --update_to_data $UTD --omega $OMEGA \
          --experiment_name "LR${LEARNING_RATE_DQN}_T${TARGET_UPDATE_PERIOD_DQN}_UTD${UTD}_OMG${OMEGA}_${GAME}"

          launch_job/atari/${PLATFORM}_mmidqnshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS  \
          --learning_rate $LEARNING_RATE_IDQN --target_update_period $TARGET_UPDATE_PERIOD_IDQN \
          --n_bellman_iterations $N_BELLMAN_ITERATIONS --omega $OMEGA \
          --experiment_name "LR${LEARNING_RATE_IDQN}_T${TARGET_UPDATE_PERIOD_IDQN}_K${N_BELLMAN_ITERATIONS}_OMG${OMEGA}_${GAME}"

          launch_job/atari/${PLATFORM}_mmgidqnshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS  \
              --learning_rate $LEARNING_RATE_GIDQN --target_update_period $TARGET_UPDATE_PERIOD_GIDQN \
              --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY  --omega $OMEGA \
              --experiment_name "LR${LEARNING_RATE_GIDQN}_T${TARGET_UPDATE_PERIOD_GIDQN}_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_OMG${OMEGA}_${GAME}"
    done
done