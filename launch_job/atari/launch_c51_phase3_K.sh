SHARED_ARGS="--features 16 256 --replay_buffer_capacity 50_000 --batch_size 16 --update_horizon 1 \
    --gamma 0.99 --horizon 10_000 --n_epochs 30 --n_training_steps_per_epoch 50_000 --update_to_data 1 \
    --n_initial_samples 10_000 --epsilon_end 0.01 --epsilon_duration 100_000 --pixels 16 --n_frame_stack 2 \
    --n_frame_skip 4"

# SHARED_ARGS="$SHARED_ARGS --disable_wandb"
WEIGHT_DECAY=1

PLATFORM="local/local"  # cluster/cluster lichtenberg/cluster local/local

TARGET_UPDATE_PERIOD_IC51=10000
LEARNING_RATE_IC51=25e-5

TARGET_UPDATE_PERIOD_GIC51=10000
LEARNING_RATE_GIC51=10e-5

for GAME in Gopher YarsRevenge Enduro Asterix Alien
do
    for N_BELLMAN_ITERATIONS in 2 20 50 100
    do
          launch_job/atari/${PLATFORM}_ic51shared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS  \
          --learning_rate $LEARNING_RATE_IC51 --target_update_period $TARGET_UPDATE_PERIOD_IC51 --n_bellman_iterations $N_BELLMAN_ITERATIONS \
          --experiment_name "LR${LEARNING_RATE_IC51}_T${TARGET_UPDATE_PERIOD_IC51}_K${N_BELLMAN_ITERATIONS}_${GAME}"

          launch_job/atari/${PLATFORM}_gic51shared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS  \
              --learning_rate $LEARNING_RATE_GIC51 --target_update_period $TARGET_UPDATE_PERIOD_GIC51 \
              --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY \
              --experiment_name "LR${LEARNING_RATE_GIC51}_T${TARGET_UPDATE_PERIOD_GIC51}_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${GAME}"
    done
done
