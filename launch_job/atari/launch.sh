SHARED_ARGS="--features 32 64 64 512 --replay_buffer_capacity 1_000_000 --batch_size 32 --update_horizon 1 --gamma 0.99 \
    --horizon 27_000 --n_epochs 40 --n_training_steps_per_epoch 250_000 --n_initial_samples 20_000 \
    --epsilon_end 0.01 --epsilon_duration 250_000 --learning_rate 6.25e-5"

GAME="Qbert"
N_BELLMAN_ITERATIONS=50  # 1 3 5 10
UPDATE_TO_DATA=0.25  # 0.25 1
LAYER_NORM=0  # 0 1
ARCHITECTURE_TYPE="cnn"  # cnn impala
WEIGHT_DECAY=1
TARGET_UPDATE_PERIOD=8000
FREEZE_FIRST_HEAD=1 # 0 1
GAP=0 # 0 1 
LINEAR_HEADS=1 # 0 1
DISABLE_WANDB=0 # 0 1

PLATFORM="cluster/cluster"  # cluster/cluster local/local

if [ $GAP == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --gap"
fi
if [ $LAYER_NORM == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --layer_norm"
fi
if [ $DISABLE_WANDB == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --disable_wandb"
fi

SHARED_ARGS="$SHARED_ARGS --target_update_period $TARGET_UPDATE_PERIOD --architecture_type $ARCHITECTURE_TYPE --update_to_data $UPDATE_TO_DATA"
SHARED_NAME="LN${LAYER_NORM}_${ARCHITECTURE_TYPE}_T${TARGET_UPDATE_PERIOD}_UTD${UPDATE_TO_DATA}"

DQN_ARGS="--experiment_name L2_${SHARED_NAME}_${GAME}"
# launch_job/atari/${PLATFORM}_dqn.sh --first_seed 3 --last_seed 3 --n_parallel_seeds 2 $SHARED_ARGS $DQN_ARGS
# launch_job/atari/${PLATFORM}_dqn.sh --first_seed 0 --last_seed 0 --n_parallel_seeds 1 $SHARED_ARGS $DQN_ARGS

# DQNRC_ARGS="--experiment_name L2_WD${WEIGHT_DECAY}_${SHARED_NAME}_${GAME} --weight_decay $WEIGHT_DECAY"
# launch_job/atari/${PLATFORM}_dqnrc.sh --first_seed 1 --last_seed 1 --n_parallel_seeds 3 $SHARED_ARGS $DQNRC_ARGS
# launch_job/atari/${PLATFORM}_dqnrc.sh --first_seed 0 --last_seed 0 --n_parallel_seeds 2 $SHARED_ARGS $DQNRC_ARGS

# IDQN_ARGS="--experiment_name L2_K${N_BELLMAN_ITERATIONS}_${SHARED_NAME}_${GAME} --n_bellman_iterations $N_BELLMAN_ITERATIONS"
# launch_job/atari/${PLATFORM}_idqn.sh --first_seed 1 --last_seed 1 --n_parallel_seeds 3 $SHARED_ARGS $IDQN_ARGS
# launch_job/atari/${PLATFORM}_idqn.sh --first_seed 0 --last_seed 0 --n_parallel_seeds 2 $SHARED_ARGS $IDQN_ARGS

# GIDQN_ARGS="--n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY"
# if [ $FREEZE_FIRST_HEAD == 1 ]
# then
#     launch_job/atari/${PLATFORM}_gidqn.sh --first_seed 1 --last_seed 1 --n_parallel_seeds 3 $SHARED_ARGS $GIDQN_ARGS --freeze_first_head \
#     --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_${GAME}
#     launch_job/atari/${PLATFORM}_gidqn.sh --first_seed 0 --last_seed 0 --n_parallel_seeds 2 $SHARED_ARGS $GIDQN_ARGS --freeze_first_head \
#     --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_${GAME}
# else
#     launch_job/atari/${PLATFORM}_gidqn.sh --first_seed 1 --last_seed 1 --n_parallel_seeds 3 $SHARED_ARGS $GIDQN_ARGS \
#     --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_UNFROZEN_${GAME}
#     launch_job/atari/${PLATFORM}_gidqn.sh --first_seed 0 --last_seed 0 --n_parallel_seeds 2 $SHARED_ARGS $GIDQN_ARGS \
#     --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_UNFROZEN_${GAME}
# fi


# ------- Shared Architectures ----------
if [ $LINEAR_HEADS == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --linear_heads"
    SHARED_NAME="${SHARED_NAME}_LINEAR"
fi
DQNRCSHARED_ARGS="--experiment_name L2_WD${WEIGHT_DECAY}_${SHARED_NAME}_${GAME} --weight_decay $WEIGHT_DECAY"
# launch_job/atari/${PLATFORM}_dqnrcshared.sh --first_seed 3 --last_seed 3 --n_parallel_seeds 2 $SHARED_ARGS $DQNRCSHARED_ARGS
# launch_job/atari/${PLATFORM}_dqnrcshared.sh --first_seed 0 --last_seed 0 --n_parallel_seeds 2 $SHARED_ARGS $DQNRCSHARED_ARGS

# IDQNSHARED_ARGS="--experiment_name L2_K${N_BELLMAN_ITERATIONS}_${SHARED_NAME}_${GAME} --n_bellman_iterations $N_BELLMAN_ITERATIONS"
# launch_job/atari/${PLATFORM}_idqnshared.sh --first_seed 3 --last_seed 3 --n_parallel_seeds 2 $SHARED_ARGS $IDQNSHARED_ARGS
# launch_job/atari/${PLATFORM}_idqnshared.sh --first_seed 0 --last_seed 0 --n_parallel_seeds 1 $SHARED_ARGS $IDQNSHARED_ARGS

# GIDQNSHARED_ARGS="--experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_${GAME} --n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY"
# if [ $FREEZE_FIRST_HEAD == 1 ]
# then
#     launch_job/atari/${PLATFORM}_gidqnshared.sh --first_seed 3 --last_seed 3 --n_parallel_seeds 2 $SHARED_ARGS $GIDQNSHARED_ARGS --freeze_first_head \
#     --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_${GAME}
#     # launch_job/atari/${PLATFORM}_gidqnshared.sh --first_seed 3 --last_seed 3 --n_parallel_seeds 1 $SHARED_ARGS $GIDQNSHARED_ARGS --freeze_first_head \
#     # --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_${GAME}
# # else
#     # launch_job/atari/${PLATFORM}_gidqnshared.sh --first_seed 1 --last_seed 1 --n_parallel_seeds 3 $SHARED_ARGS $GIDQNSHARED_ARGS \
#     # --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_UNFROZEN_${GAME}
#     # launch_job/atari/${PLATFORM}_gidqnshared.sh --first_seed 0 --last_seed 0 --n_parallel_seeds 2 $SHARED_ARGS $GIDQNSHARED_ARGS \
#     # --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_UNFROZEN_${GAME}
# fi
