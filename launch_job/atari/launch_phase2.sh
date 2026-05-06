SHARED_ARGS="--features 16 256 --replay_buffer_capacity 50_000 --batch_size 16 --update_horizon 1 \
    --gamma 0.99 --horizon 10_000 --n_epochs 30 --n_training_steps_per_epoch 50_000 --update_to_data 1 \
    --n_initial_samples 10_000 --epsilon_end 0.01 --epsilon_duration 100_000 --pixels 16 --n_frame_stack 2 \
    --n_frame_skip 4"

# SHARED_ARGS="$SHARED_ARGS --disable_wandb"
N_BELLMAN_ITERATIONS=5
WEIGHT_DECAY=1

PLATFORM="local/local"  # cluster/cluster lichtenberg/cluster local/local

for GAME in Gopher YarsRevenge Enduro Asterix Alien
do 
    for LEARNING_RATE in 1e-5 10e-5 25e-5 50e-5 100e-5
    do
        for TARGET_UPDATE_PERIOD in 100 300 3000 6000 10000
        do
            SHARED_ARGS="$SHARED_ARGS --learning_rate $LEARNING_RATE --target_update_period $TARGET_UPDATE_PERIOD"
            SHARED_NAME="LR${LEARNING_RATE}_T${TARGET_UPDATE_PERIOD}"

            launch_job/atari/${PLATFORM}_dqn.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS \
                --experiment_name ${SHARED_NAME}_${GAME}

            launch_job/atari/${PLATFORM}_dqnrcshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS \
                --weight_decay $WEIGHT_DECAY --experiment_name ${SHARED_NAME}_WD${WEIGHT_DECAY}_${GAME}

            launch_job/atari/${PLATFORM}_idqnshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS \
                --n_bellman_iterations $N_BELLMAN_ITERATIONS --experiment_name ${SHARED_NAME}_K${N_BELLMAN_ITERATIONS}_${GAME}

            launch_job/atari/${PLATFORM}_gidqnshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS \
                --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY \
                --experiment_name ${SHARED_NAME}_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${GAME}
        done
    done
done