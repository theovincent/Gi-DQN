#!/bin/bash


NE=30
RB=50_000
N_INIT_SMPL=10_000
NTSPE=50_000

LR=10e-5
GAME="DoubleDunk"
TARGET_UPDATE_PERIOD=100
LINEAR_HEADS=1


WEIGHT_DECAY=1
N_BELLMAN_ITERATIONS=5
DISABLE_WANDB=0
PLATFORM="cluster/cluster"
UPDATE_TO_DATA=1
ARCHITECTURE_TYPE="cnn"

GAP=0
UPDATE_HORIZON=1
PER=0
LAYER_NORM_CONV=1
LAYER_NORM_FC=1
N_CONV=1
N_FC=2
LOW_SCALE=1
FRAME_STACK=2
FRAME_SKIP=4

# first/last seed defaults (can be overridden)
FIRST_SEED=1
LAST_SEED=5

# which algo to launch: all | dqn | gidqnshared | idqnshared | dqnrcshared
LAUNCH_ALGO="all"

# CLI argument overrides 
while [[ $# -gt 0 ]]; do
    case "$1" in
        --algo)           LAUNCH_ALGO="$2";           shift 2 ;;
        --lr)             LR="$2";                    shift 2 ;;
        --tup)            TARGET_UPDATE_PERIOD="$2";  shift 2 ;;
        --game)           GAME="$2";                  shift 2 ;;
        --first_seed)     FIRST_SEED="$2";            shift 2 ;;
        --last_seed)      LAST_SEED="$2";             shift 2 ;;
        --ne)             NE="$2";                    shift 2 ;;
        --platform)       PLATFORM="$2";              shift 2 ;;
        --disable_wandb)  DISABLE_WANDB=1;            shift 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

##########
CUSTOM_TAG="LR${LR}_NE${NE}"
ADDGAP=""
ADDPER=""

SHARED_ARGS="--replay_buffer_capacity ${RB} --batch_size 16 --gamma 0.99 --horizon 10_000 \
    --n_initial_samples ${N_INIT_SMPL} --epsilon_end 0.01 --epsilon_duration 100_000 --learning_rate ${LR}"

SHARED_ARGS="$SHARED_ARGS --n_epochs ${NE} --n_training_steps_per_epoch ${NTSPE} --features 16 128"

if [ $LOW_SCALE == 1 ]; then
    SHARED_ARGS="$SHARED_ARGS --low_scale"
fi
if [ $GAP == 1 ]; then
    SHARED_ARGS="$SHARED_ARGS --gap"
    ADDGAP="_GAP"
fi
if [ $PER == 1 ]; then
    SHARED_ARGS="$SHARED_ARGS --per"
    ADDPER="_PER"
fi
if [ $DISABLE_WANDB == 1 ]; then
    SHARED_ARGS="$SHARED_ARGS --disable_wandb"
fi

SHARED_ARGS="$SHARED_ARGS --architecture_type $ARCHITECTURE_TYPE \
    --update_to_data $UPDATE_TO_DATA --update_horizon $UPDATE_HORIZON \
    --n_frame_stack $FRAME_STACK --n_frame_skip $FRAME_SKIP \
    --layer_norm $LAYER_NORM_CONV $LAYER_NORM_FC --n_conv $N_CONV --n_fc $N_FC"

ARCHTAG="F_c16_f128"
SHARED_NAME="LS${LOW_SCALE}_ST${FRAME_STACK}_SK${FRAME_SKIP}_ncnv${N_CONV}_nfc${N_FC}_${ARCHTAG}_LN${LAYER_NORM_CONV}${LAYER_NORM_FC}${ARCHITECTURE_TYPE}${ADDGAP}${ADDPER}_${CUSTOM_TAG}"

# DQN
if [[ "$LAUNCH_ALGO" == "all" || "$LAUNCH_ALGO" == "dqn" ]]; then
    DQN_ARGS="--experiment_name L2_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_${GAME} \
              --target_update_period $TARGET_UPDATE_PERIOD"
    launch_job/atari/${PLATFORM}_dqn.sh \
        --first_seed $FIRST_SEED --last_seed $LAST_SEED --n_parallel_seeds 1 \
        $SHARED_ARGS $DQN_ARGS
fi

#
if [ $LINEAR_HEADS == 1 ]; then
    SHARED_ARGS="$SHARED_ARGS --linear_heads"
    SHARED_NAME="${SHARED_NAME}_LINEAR"
fi

# GiDQNShared 
if [[ "$LAUNCH_ALGO" == "all" || "$LAUNCH_ALGO" == "gidqnshared" ]]; then
    GIDQNSHARED_ARGS="--n_bellman_iterations $N_BELLMAN_ITERATIONS \
                      --weight_decay $WEIGHT_DECAY \
                      --target_update_period $TARGET_UPDATE_PERIOD \
                      --experiment_name L2_K${N_BELLMAN_ITERATIONS}_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_${GAME}"
    launch_job/atari/${PLATFORM}_gidqnshared.sh \
        --first_seed $FIRST_SEED --last_seed $LAST_SEED --n_parallel_seeds 1 \
        $SHARED_ARGS $GIDQNSHARED_ARGS
fi

# iDQNShared 
if [[ "$LAUNCH_ALGO" == "all" || "$LAUNCH_ALGO" == "idqnshared" ]]; then
    IDQNSHARED_ARGS="--n_bellman_iterations $N_BELLMAN_ITERATIONS \
                     --target_update_period $TARGET_UPDATE_PERIOD \
                     --experiment_name L2_K${N_BELLMAN_ITERATIONS}_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_${GAME}"
    launch_job/atari/${PLATFORM}_idqnshared.sh \
        --first_seed $FIRST_SEED --last_seed $LAST_SEED --n_parallel_seeds 1 \
        $SHARED_ARGS $IDQNSHARED_ARGS
fi

#  DQNRCShared
if [[ "$LAUNCH_ALGO" == "all" || "$LAUNCH_ALGO" == "dqnrcshared" ]]; then
    DQNRCSHARED_ARGS="--weight_decay $WEIGHT_DECAY \
                      --experiment_name L2_${SHARED_NAME}_${GAME}"
    launch_job/atari/${PLATFORM}_dqnrcshared.sh \
        --first_seed $FIRST_SEED --last_seed $LAST_SEED --n_parallel_seeds 1 \
        $SHARED_ARGS $DQNRCSHARED_ARGS
fi
