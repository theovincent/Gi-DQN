NE=25
RB=62_000
N_INIT_SMPL=4_000
NTSPE=50_000
LR=1e-3

SHARED_ARGS="--replay_buffer_capacity ${RB} --batch_size 24 --gamma 0.99 --horizon 10_000 \
    --n_initial_samples ${N_INIT_SMPL} --epsilon_end 0.01 --epsilon_duration 50_000 --learning_rate ${LR}"


GAME="Breakout" # SpaceInvaders, Breakout, ...
UPDATE_TO_DATA=1  # 0.03125 0.25 4
ARCHITECTURE_TYPE="cnn"  # cnn impala
GAP=1 # 0 1
UPDATE_HORIZON=1  # 1 3
PER=0  # 0 1
LAYER_NORM=1  # 0 1
TARGET_UPDATE_PERIOD=1500
LINEAR_HEADS=1 # 0 1
WEIGHT_DECAY=1
N_BELLMAN_ITERATIONS=5
DISABLE_WANDB=0 # 0 1

PLATFORM="cluster/cluster"  # cluster/cluster local/local

CUSTOM_TAG="stack2_skip4_pxl42_smaller_arch"
CUSTOM_TAG="${CUSTOM_TAG}_RB_${RB}_N_INIT_SMPL_${N_INIT_SMPL}_NE_${NE}_NTSPE_${NTSPE}_LR_${LR}"

if [ $ARCHITECTURE_TYPE == "cnn" ]
then
    SHARED_ARGS="$SHARED_ARGS --features 16 16 16 32"
else
    SHARED_ARGS="$SHARED_ARGS --features 32 64 64 512"
fi
if [ $UPDATE_TO_DATA == 4 ]
then
    SHARED_ARGS="$SHARED_ARGS --n_epochs ${NE} --n_training_steps_per_epoch ${NTSPE}"
else
    SHARED_ARGS="$SHARED_ARGS --n_epochs ${NE} --n_training_steps_per_epoch ${NTSPE}"
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
SHARED_NAME="${CUSTOM_TAG}_LN${LAYER_NORM}_${ARCHITECTURE_TYPE}${ADDGAP}${ADDPER}_UTD${UPDATE_TO_DATA}_NSTEP${UPDATE_HORIZON}"

DQN_ARGS="--experiment_name L2_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_${GAME}"
launch_job/atari/${PLATFORM}_dqn.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $DQN_ARGS

if [ $LINEAR_HEADS == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --linear_heads"
    SHARED_NAME="${SHARED_NAME}_LINEAR"
fi


GIDQNSHARED_ARGS="--n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY"
GIDQNSHARED_ARGS="$GIDQNSHARED_ARGS --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_${GAME}"
launch_job/atari/${PLATFORM}_gidqnshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $GIDQNSHARED_ARGS
