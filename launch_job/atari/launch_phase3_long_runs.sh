SHARED_ARGS="--features 16 256 --replay_buffer_capacity 50_000 --batch_size 16 --update_horizon 1 --update_to_data 1 \
    --gamma 0.99 --horizon 10_000 --n_epochs 250 --n_training_steps_per_epoch 50_000 \
    --n_initial_samples 10_000 --epsilon_end 0.01 --epsilon_duration 100_000 --pixels 16 --n_frame_stack 2 \
    --n_frame_skip 4"

# SHARED_ARGS="$SHARED_ARGS --disable_wandb"
N_BELLMAN_ITERATIONS=5
WEIGHT_DECAY=1

PLATFORM="lichtenberg/cluster"  # cluster/cluster lichtenberg/cluster local/local


TARGET_UPDATE_PERIOD_DQN=3000
LEARNING_RATE_DQN=10e-5

LEARNING_RATE_QRC=10e-5

TARGET_UPDATE_PERIOD_IDQN=10000
LEARNING_RATE_IDQN=25e-5

TARGET_UPDATE_PERIOD_GIDQN=10000
LEARNING_RATE_GIDQN=10e-5

for GAME in Atlantis Boxing Breakout Assault SpaceInvaders Asterix Enduro Phoenix Pong KungFuMaster NameThisGame Qbert MsPAcman VideoPinball Krull RoadRunner StarGunner \
            Gopher CrazyClimber Jamesbond Frostbite YarsRevenge Riverraid Amidar Alien
do
          launch_job/atari/${PLATFORM}_dqn.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS \
              --learning_rate $LEARNING_RATE_DQN --target_update_period $TARGET_UPDATE_PERIOD_DQN \
              --experiment_name "LR${LEARNING_RATE_DQN}_T${TARGET_UPDATE_PERIOD_DQN}_${GAME}"

          launch_job/atari/${PLATFORM}_dqnrcshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS \
              --learning_rate $LEARNING_RATE_QRC \
              --weight_decay $WEIGHT_DECAY --experiment_name "LR${LEARNING_RATE_QRC}_WD${WEIGHT_DECAY}_${GAME}"

          launch_job/atari/${PLATFORM}_idqnshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS \
              --learning_rate $LEARNING_RATE_IDQN --target_update_period $TARGET_UPDATE_PERIOD_IDQN \
              --n_bellman_iterations $N_BELLMAN_ITERATIONS --experiment_name "LR${LEARNING_RATE_IDQN}_T${TARGET_UPDATE_PERIOD_IDQN}_K${N_BELLMAN_ITERATIONS}_${GAME}"

          launch_job/atari/${PLATFORM}_gidqnshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS \
              --learning_rate $LEARNING_RATE_GIDQN --target_update_period $TARGET_UPDATE_PERIOD_GIDQN \
              --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY \
              --experiment_name "LR${LEARNING_RATE_GIDQN}_T${TARGET_UPDATE_PERIOD_GIDQN}_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${GAME}"
done