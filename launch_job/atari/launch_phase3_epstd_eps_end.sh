SHARED_ARGS="--features 16 256 --replay_buffer_capacity 50_000 --batch_size 16 --update_horizon 1 \
    --gamma 0.99 --horizon 10_000 --n_epochs 30 --n_training_steps_per_epoch 50_000 --update_to_data 1 \
    --n_initial_samples 10_000 --epsilon_duration 100_000 --pixels 16 --n_frame_stack 2 \
    --n_frame_skip 4"

# SHARED_ARGS="$SHARED_ARGS --disable_wandb"
N_BELLMAN_ITERATIONS=5
WEIGHT_DECAY=1

PLATFORM="lichtenberg/cluster"  # cluster/cluster lichtenberg/cluster local/local


LEARNING_RATE_QRC=10e-5

TARGET_UPDATE_PERIOD_GIDQN=10000
LEARNING_RATE_GIDQN=10e-5

for GAME in Gopher YarsRevenge Enduro Asterix Alien
do
    for EPS_END in 0.05 0.15 0.2 0.01 0.1
    do
    launch_job/atari/${PLATFORM}_epstddqnrcshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS \
        --learning_rate $LEARNING_RATE_QRC --epsilon_end $EPS_END \
        --weight_decay $WEIGHT_DECAY --experiment_name "LR${LEARNING_RATE_QRC}_WD${WEIGHT_DECAY}_END${EPS_END}_${GAME}"

    launch_job/atari/${PLATFORM}_epstdgidqnshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS \
        --learning_rate $LEARNING_RATE_GIDQN --target_update_period $TARGET_UPDATE_PERIOD_GIDQN  \
        --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY --epsilon_end $EPS_END \
        --experiment_name "LR${LEARNING_RATE_GIDQN}_T${TARGET_UPDATE_PERIOD_GIDQN}_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_END${EPS_END}_${GAME}"
    done
done
