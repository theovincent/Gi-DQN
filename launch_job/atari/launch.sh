NE=25
RB=100_000
N_INIT_SMPL=10_000
NTSPE=50_000
LR=6.25e-4

SHARED_ARGS="--replay_buffer_capacity ${RB} --batch_size 24 --gamma 0.99 --horizon 10_000 \
    --n_initial_samples ${N_INIT_SMPL} --epsilon_end 0.01 --epsilon_duration 50_000 --learning_rate ${LR}"


GAME="Breakout" # SpaceInvaders, Breakout, ...
UPDATE_TO_DATA=1  # 0.03125 0.25 4
ARCHITECTURE_TYPE="cnn"  # cnn impala
GAP=1 # 0 1
UPDATE_HORIZON=1  # 1 3
PER=0  # 0 1

#########################
LAYER_NORM_CONV=1 # 0 1
LAYER_NORM_FC=1 # 0 1

#Architecture
CONV2=0 # 0 1, whether to use 3 convolutional layers, when true the first 3 entries of features are Conv Layers
FC1=0 # 0 1, whether to use only one FC layer to map directly to actions

# Atari
LOW_SCALE=0 # 0 1  whether to use 42 x 42 pixels instead of 84 x 84
FRAME_STACK=2
FRAME_SKIP=4
#########################

TARGET_UPDATE_PERIOD=1500
LINEAR_HEADS=1 # 0 1
WEIGHT_DECAY=1
N_BELLMAN_ITERATIONS=5
DISABLE_WANDB=0 # 0 1



PLATFORM="cluster/cluster"  # cluster/cluster local/local


if [ $LOW_SCALE == 1 ]
then
  SHARED_ARGS="$SHARED_ARGS --low_scale"
fi
if [ $CONV2 == 1 ]
then
  SHARED_ARGS="$SHARED_ARGS --conv2"
  CONVLAYERS="cnv2"
else
  CONVLAYERS="cnv3"
fi
if [ $FC1 == 1 ]
then
  SHARED_ARGS="$SHARED_ARGS --fc1"
  FCLAYERS="fc1"
else
  FCLAYERS="fc2"
fi


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
    ADDGAP="_GAP"
fi
if [ $PER == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --per"
    ADDPER="_PER"
fi
if [ $DISABLE_WANDB == 1 ]
then
    SHARED_ARGS="$SHARED_ARGS --disable_wandb"
fi
CUSTOM_TAG="RB_${RB}_NE_${NE}_NTSPE_${NTSPE}_LR_${LR}"

SHARED_ARGS="$SHARED_ARGS --target_update_period $TARGET_UPDATE_PERIOD --architecture_type $ARCHITECTURE_TYPE \
    --update_to_data $UPDATE_TO_DATA --update_horizon $UPDATE_HORIZON --n_frame_stack $FRAME_STACK --n_frame_skip $FRAME_SKIP --layer_norm $LAYER_NORM_CONV $LAYER_NORM_FC"
SHARED_NAME="LS${LOW_SCALE}_ST${FRAME_STACK}_SK${FRAME_SKIP}_${CONVLAYERS}_${FCLAYERS}_LN${LAYER_NORM_CONV}${LAYER_NORM_FC}_${ARCHITECTURE_TYPE}${ADDGAP}${ADDPER}_UTD${UPDATE_TO_DATA}_NSTEP${UPDATE_HORIZON}_${CUSTOM_TAG}"

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
