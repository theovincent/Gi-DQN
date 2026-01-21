SHARED_ARGS="--replay_buffer_capacity 1_000_000 --batch_size 32 --gamma 0.99 --horizon 27_000 \
    --n_initial_samples 20_000 --epsilon_end 0.01 --epsilon_duration 250_000 --learning_rate 6.25e-5"

GAME="Alien"
UPDATE_TO_DATA=0.03125  # 0.03125 0.25 4
ARCHITECTURE_TYPE="impala"  # cnn impala
GAP=1 # 0 1
UPDATE_HORIZON=3  # 1 3
PER=1  # 0 1
LAYER_NORM=0  # 0 1
TARGET_UPDATE_PERIOD=8000
LINEAR_HEADS=1 # 0 1
WEIGHT_DECAY=1
N_BELLMAN_ITERATIONS=5
FREEZE_FIRST_HEAD=1 # 0 1
DISABLE_WANDB=0 # 0 1

PLATFORM="cluster/cluster"  # cluster/cluster local/local pegasus/cluster

if [ $ARCHITECTURE_TYPE == "cnn" ]
then
    SHARED_ARGS="$SHARED_ARGS --features 32 64 64 512"
else
    SHARED_ARGS="$SHARED_ARGS --features 16 32 32 512"
fi
if [ $UPDATE_TO_DATA == 4 ]
then
    SHARED_ARGS="$SHARED_ARGS --n_epochs 20 --n_training_steps_per_epoch 80_000"
else
    SHARED_ARGS="$SHARED_ARGS --n_epochs 100 --n_training_steps_per_epoch 250_000"
fi
if [ $GAP == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --gap"
    ADDGAP="+GAP"
fi
if [ $PER == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --per"
    ADDPER="+PER"
fi
if [ $LAYER_NORM == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --layer_norm"
fi
if [ $DISABLE_WANDB == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --disable_wandb"
fi

SHARED_ARGS="$SHARED_ARGS --target_update_period $TARGET_UPDATE_PERIOD --architecture_type $ARCHITECTURE_TYPE \
    --update_to_data $UPDATE_TO_DATA --update_horizon $UPDATE_HORIZON"
SHARED_NAME="LN${LAYER_NORM}_${ARCHITECTURE_TYPE}${ADDGAP}${ADDPER}_UTD${UPDATE_TO_DATA}_NSTEP${UPDATE_HORIZON}"

# DQN_ARGS="--experiment_name L2_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_${GAME}"
# launch_job/atari/${PLATFORM}_dqn.sh --first_seed 1 --last_seed 3 --n_parallel_seeds 3 $SHARED_ARGS $DQN_ARGS
# launch_job/atari/${PLATFORM}_dqn.sh --first_seed 4 --last_seed 5 --n_parallel_seeds 2 $SHARED_ARGS $DQN_ARGS

# DQNRC_ARGS="--experiment_name L2_WD${WEIGHT_DECAY}_${SHARED_NAME}_${GAME} --weight_decay $WEIGHT_DECAY"
# launch_job/atari/${PLATFORM}_dqnrc.sh --first_seed 1 --last_seed 3 --n_parallel_seeds 3 $SHARED_ARGS $DQNRC_ARGS
# launch_job/atari/${PLATFORM}_dqnrc.sh --first_seed 4 --last_seed 5 --n_parallel_seeds 2 $SHARED_ARGS $DQNRC_ARGS

# IDQN_ARGS="--experiment_name L2_K${N_BELLMAN_ITERATIONS}_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_${GAME} --n_bellman_iterations $N_BELLMAN_ITERATIONS"
# launch_job/atari/${PLATFORM}_idqn.sh --first_seed 1 --last_seed 3 --n_parallel_seeds 3 $SHARED_ARGS $IDQN_ARGS
# launch_job/atari/${PLATFORM}_idqn.sh --first_seed 4 --last_seed 5 --n_parallel_seeds 2 $SHARED_ARGS $IDQN_ARGS

# GIDQN_ARGS="--n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY"
# if [ $FREEZE_FIRST_HEAD == 1 ]
# then
#     GIDQN_ARGS="$GIDQN_ARGS --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_${GAME} --freeze_first_head"
# else
#     GIDQN_ARGS="$GIDQN_ARGS --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_UNFROZEN_${GAME}"
# fi
# launch_job/atari/${PLATFORM}_gidqn.sh --first_seed 1 --last_seed 3 --n_parallel_seeds 3 $SHARED_ARGS $GIDQN_ARGS
# launch_job/atari/${PLATFORM}_gidqn.sh --first_seed 4 --last_seed 5 --n_parallel_seeds 2 $SHARED_ARGS $GIDQN_ARGS


# ------- Shared Architectures ----------
if [ $LINEAR_HEADS == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --linear_heads"
    SHARED_NAME="${SHARED_NAME}_LINEAR"
fi
# DQNRCSHARED_ARGS="--experiment_name L2_WD${WEIGHT_DECAY}_${SHARED_NAME}_${GAME} --weight_decay $WEIGHT_DECAY"
# launch_job/atari/${PLATFORM}_dqnrcshared.sh --first_seed 1 --last_seed 3 --n_parallel_seeds 3 $SHARED_ARGS $DQNRCSHARED_ARGS
# launch_job/atari/${PLATFORM}_dqnrcshared.sh --first_seed 4 --last_seed 5 --n_parallel_seeds 2 $SHARED_ARGS $DQNRCSHARED_ARGS

# IDQNSHARED_ARGS="--experiment_name L2_K${N_BELLMAN_ITERATIONS}_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_${GAME} --n_bellman_iterations $N_BELLMAN_ITERATIONS"
# launch_job/atari/${PLATFORM}_idqnshared.sh --first_seed 1 --last_seed 3 --n_parallel_seeds 3 $SHARED_ARGS $IDQNSHARED_ARGS
# launch_job/atari/${PLATFORM}_idqnshared.sh --first_seed 4 --last_seed 5 --n_parallel_seeds 2 $SHARED_ARGS $IDQNSHARED_ARGS

# GIDQNSHARED_ARGS="--n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY"
# if [ $FREEZE_FIRST_HEAD == 1 ]
# then
#     GIDQNSHARED_ARGS="$GIDQNSHARED_ARGS --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_${GAME} --freeze_first_head"
# else
#     GIDQNSHARED_ARGS="$GIDQNSHARED_ARGS --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_UNFROZEN_${GAME}"
# fi
# launch_job/atari/${PLATFORM}_gidqnshared.sh --first_seed 1 --last_seed 3 --n_parallel_seeds 3 $SHARED_ARGS $GIDQNSHARED_ARGS
# launch_job/atari/${PLATFORM}_gidqnshared.sh --first_seed 4 --last_seed 5 --n_parallel_seeds 2 $SHARED_ARGS $GIDQNSHARED_ARGS