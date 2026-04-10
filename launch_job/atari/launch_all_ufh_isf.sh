TARGET_UPDATE_PERIOD=600

NE=30
RB=50_000
N_INIT_SMPL=10_000
NTSPE=50_000

UPDATE_TO_DATA=1
ARCHITECTURE_TYPE="cnn"
GAP=0
UPDATE_HORIZON=1
PER=0
########################
LAYER_NORM_CONV=1
LAYER_NORM_FC=1
N_CONV=1
N_FC=2
FEATURES="16 128"
ARCHTAG="F_c16_f128"
LOW_SCALE=1
FRAME_STACK=2
FRAME_SKIP=4
#########################
LINEAR_HEADS=1
WEIGHT_DECAY=1
N_BELLMAN_ITERATIONS=5
DISABLE_WANDB=0
PLATFORM="cluster/cluster"

for GAME in BattleZone DoubleDunk NameThisGame Phoenix Qbert; do
for LR in 1e-5 25e-5 100e-5; do
for ISF in 0 1; do
for UFH in 0 1; do

    CUSTOM_TAG="LR${LR}_NE${NE}"
    ADDGAP=""
    ADDPER=""
    ISF_TAG=""
    UFH_TAG=""

    SHARED_ARGS="--replay_buffer_capacity ${RB} --batch_size 16 --gamma 0.99 --horizon 10_000 \
        --n_initial_samples ${N_INIT_SMPL} --epsilon_end 0.01 --epsilon_duration 100_000 --learning_rate ${LR} \
        --n_epochs ${NE} --n_training_steps_per_epoch ${NTSPE} --features $FEATURES"

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
        --update_to_data $UPDATE_TO_DATA --update_horizon $UPDATE_HORIZON --n_frame_stack $FRAME_STACK \
        --n_frame_skip $FRAME_SKIP --layer_norm $LAYER_NORM_CONV $LAYER_NORM_FC --n_conv $N_CONV --n_fc $N_FC"

    SHARED_NAME="LS${LOW_SCALE}_ST${FRAME_STACK}_SK${FRAME_SKIP}_ncnv${N_CONV}_nfc${N_FC}_${ARCHTAG}_LN${LAYER_NORM_CONV}${LAYER_NORM_FC}${ARCHITECTURE_TYPE}${ADDGAP}${ADDPER}_${CUSTOM_TAG}"

    if [ $LINEAR_HEADS == 1 ]; then
        SHARED_ARGS="$SHARED_ARGS --linear_heads"
        SHARED_NAME="${SHARED_NAME}_LINEAR"
    fi
    if [ $ISF == 1 ]; then
        SHARED_ARGS="$SHARED_ARGS --iterated_shared_features"
        ISF_TAG="_ISF"
    fi

    GIDQNSHARED_ARGS="--n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY --target_update_period $TARGET_UPDATE_PERIOD"
    if [ $UFH == 1 ]; then
        GIDQNSHARED_ARGS="$GIDQNSHARED_ARGS --unfreeze_first_head"
        UFH_TAG="_UFH"
    fi
    GIDQNSHARED_ARGS="$GIDQNSHARED_ARGS --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}${UFH_TAG}${ISF_TAG}_T${TARGET_UPDATE_PERIOD}_${GAME}"
    launch_job/atari/${PLATFORM}_ufh_isf_gidqnshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $GIDQNSHARED_ARGS

    if [ $UFH == 0 ]; then
        IDQNSHARED_ARGS="--n_bellman_iterations $N_BELLMAN_ITERATIONS --target_update_period $TARGET_UPDATE_PERIOD"
        IDQNSHARED_ARGS="$IDQNSHARED_ARGS --experiment_name L2_K${N_BELLMAN_ITERATIONS}_${SHARED_NAME}${ISF_TAG}_T${TARGET_UPDATE_PERIOD}_${GAME}"
        launch_job/atari/${PLATFORM}_isf_idqnshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $IDQNSHARED_ARGS
    fi

done
done
done
done