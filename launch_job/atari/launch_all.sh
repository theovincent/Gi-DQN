
NE=30
RB=50_000
N_INIT_SMPL=10_000
NTSPE=50_000

UPDATE_TO_DATA=1  # 0.25 1 2 4 8
ARCHITECTURE_TYPE="cnn"  # cnn impala
GAP=0  # 0 
UPDATE_HORIZON=1  # 1
PER=0  # 0 
########################
LAYER_NORM_CONV=1 # 0 1
LAYER_NORM_FC=1 # 0 1
#Architecture
N_CONV=1 # 1; n conv layers min 1, max 3
N_FC=2 # 2;  n fc layers, min 1 
FEATURES="16 256"
ARCHTAG="c16_f256"
# Atari frames
FRAME_STACK=2
FRAME_SKIP=4
#########################UPDATE_TO_DATA=1  # 0.25 1 2 4 8

LINEAR_HEADS=1 # 1
WEIGHT_DECAY=1 # 0.01 1 10 100 
N_BELLMAN_ITERATIONS=5 # 5 10 20 50 100
DISABLE_WANDB=0 # 0 1
PLATFORM="cluster/cluster"  # cluster/cluster local/local


#TARGET_UPDATE_PERIOD=100 # 100 250 600 1500 4000
#LR=100e-5 # 1e-5 10e-5 25e-5 50e-5 100e-5
PIXELS=16 # 84, 42, 24, 10, 16 -> Number of pixels
KERNEL=2  #first K
STRIDE=2  #first S
CUSTOM_TAG=""
for TARGET_UPDATE_PERIOD in 600; do #100 600 4000; do 
for LR in 25e-5; do #1e-5 25e-5 100e-5; do 
for GAME in Atlantis Qbert Phoenix Enduro Boxing; do #Breakout Pong Freeway SpaceInvaders Assault MsPacman Kaboom Skiing BeamRider Enduro KungFuMaster Atlantis DemonAttack Galaxian Asterix Boxing BattleZone DoubleDunk NameThisGame Phoenix Qbert; do #BattleZone DoubleDunk NameThisGame Phoenix Qbert; do

    ADDGAP=""
    ADDPER=""

    SHARED_ARGS="--replay_buffer_capacity ${RB} --batch_size 16 --gamma 0.99 --horizon 10_000 \
        --n_initial_samples ${N_INIT_SMPL} --epsilon_end 0.01 --epsilon_duration 100_000 --learning_rate ${LR} \
        --n_epochs ${NE} --n_training_steps_per_epoch ${NTSPE} --features $FEATURES --kernel $KERNEL --stride $STRIDE"

    #RANDOMPOLICY_ARGS="--experiment_name RANDOMPOLICY_${GAME}"
    #launch_job/atari/${PLATFORM}_randompolicy.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $RANDOMPOLICY_ARGS

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
        --n_frame_skip $FRAME_SKIP --layer_norm $LAYER_NORM_CONV $LAYER_NORM_FC --n_conv $N_CONV --n_fc $N_FC --pixels $PIXELS"

    SHARED_NAME="PX${PIXELS}_K${KERNEL}_S${STRIDE}_ST${FRAME_STACK}_SK${FRAME_SKIP}_ncnv${N_CONV}_nfc${N_FC}_${ARCHITECTURE_TYPE}_${ARCHTAG}_LN${LAYER_NORM_CONV}${LAYER_NORM_FC}$_${ADDGAP}${ADDPER}_LR${LR}_UTD${UPDATE_TO_DATA}"

    DQN_ARGS="--experiment_name L2_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_${GAME} --target_update_period $TARGET_UPDATE_PERIOD"
    launch_job/atari/${PLATFORM}_dqn.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $DQN_ARGS

    if [ $LINEAR_HEADS == 1 ]; then
        SHARED_ARGS="$SHARED_ARGS --linear_heads"
        SHARED_NAME="${SHARED_NAME}_LINEAR"
    fi

    GIDQNSHARED_ARGS="--n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY --target_update_period $TARGET_UPDATE_PERIOD"
    GIDQNSHARED_ARGS="$GIDQNSHARED_ARGS --experiment_name L2_K${N_BELLMAN_ITERATIONS}_WD${WEIGHT_DECAY}_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_${GAME}"
    launch_job/atari/${PLATFORM}_gidqnshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $GIDQNSHARED_ARGS

    IDQNSHARED_ARGS="--n_bellman_iterations $N_BELLMAN_ITERATIONS --target_update_period $TARGET_UPDATE_PERIOD"
    IDQNSHARED_ARGS="$IDQNSHARED_ARGS --experiment_name L2_K${N_BELLMAN_ITERATIONS}_${SHARED_NAME}_T${TARGET_UPDATE_PERIOD}_${GAME}"
    #launch_job/atari/${PLATFORM}_idqnshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $IDQNSHARED_ARGS
    
    if [ $TARGET_UPDATE_PERIOD == 100 ]; then
    	DQNRCSHARED_ARGS="--weight_decay $WEIGHT_DECAY"
   	DQNRCSHARED_ARGS="$DQNRCSHARED_ARGS --experiment_name L2_WD${WEIGHT_DECAY}_${SHARED_NAME}_${GAME}"
    	#launch_job/atari/${PLATFORM}_dqnrcshared.sh --first_seed 1 --last_seed 5 --n_parallel_seeds 1 $SHARED_ARGS $DQNRCSHARED_ARGS
    fi 
done
done
done
