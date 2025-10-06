SHARED_ARGS="--features 100 100 --replay_buffer_capacity 10000 --batch_size 32 --update_horizon 1 --gamma 0.99 \
    --horizon 1_000 --n_epochs 25 --n_training_steps_per_epoch 10_000 --data_to_update 1 --n_initial_samples 1_000 \
    --epsilon_end 0.01 --epsilon_duration 1_000 -at fc"

GAME="LunarLander"
N_BELLMAN_ITERATIONS=5  # 1 3 5 10
TARGET_UPDATE_FREQ=(75, 2_500)
TARGET_SYNC_FREQ=1
DISABLE_WIND=0 # 0 1
WEIGHT_DECAY=0.0001
LEARNING_RATE=(1e-2, 1e-7)

if [[ $DISABLE_WIND == 1 ]]; then
    SHARED_ARGS="$SHARED_ARGS --deterministic"
fi


# ----- ------------------heatmap experiments -----
DQN_ARGS=""
IDQN_ARGS="--n_bellman_iterations $N_BELLMAN_ITERATIONS --target_sync_frequency $TARGET_SYNC_FREQ"
FIDQN_ARGS="--n_bellman_iterations $N_BELLMAN_ITERATIONS"
GIDQN_ARGS="--n_bellman_iterations $N_BELLMAN_ITERATIONS --weight_decay $WEIGHT_DECAY"
DQNRC_ARGS="--weight_decay $WEIGHT_DECAY"
FIRST_SEED=0
LAST_SEED=0
FLAG_GIDQN_ARGS="$GIDQN_ARGS --unfreeze_first_head"

for lr in "${LEARNING_RATE[@]}"; do
  for tuf in "${TARGET_UPDATE_FREQ[@]}"; do
    echo "Running with learning_rate=$lr, tuf=$tuf"
    SHARED_NAME="TUF${tuf}_LR${lr}_heatmap"
    launch_job/atari/cluster_gidqn.sh --first_seed $FIRST_SEED --last_seed $LAST_SEED --n_parallel_seeds 1 --learning_rate $lr --target_update_frequency $tuf --experiment_name $SHARED_NAME $SHARED_ARGS $GIDQN_ARGS
    launch_job/atari/cluster_dqn.sh --first_seed $FIRST_SEED --last_seed $LAST_SEED  --n_parallel_seeds 1 --learning_rate $lr --target_update_frequency $tuf --experiment_name $SHARED_NAME $SHARED_ARGS $DQN_ARGS
    launch_job/atari/cluster_idqn.sh --first_seed $FIRST_SEED --last_seed $LAST_SEED  --n_parallel_seeds 1 --learning_rate $lr --target_update_frequency $tuf --experiment_name $SHARED_NAME $SHARED_ARGS $IDQN_ARGS
    launch_job/atari/cluster_fidqn.sh --first_seed $FIRST_SEED --last_seed $LAST_SEED  --n_parallel_seeds 1 --learning_rate $lr --target_update_frequency $tuf --experiment_name $SHARED_NAME $SHARED_ARGS $FIDQN_ARGS
    launch_job/atari/cluster_dqnrc.sh --first_seed $FIRST_SEED --last_seed $LAST_SEED  --n_parallel_seeds 1 --learning_rate $lr --target_update_frequency $tuf --experiment_name $SHARED_NAME $SHARED_ARGS $DQNRC_ARGS
    launch_job/atari/cluster_gidqn.sh --first_seed $FIRST_SEED --last_seed $LAST_SEED --n_parallel_seeds 1 --learning_rate $lr --target_update_frequency $tuf --experiment_name $SHARED_NAME $SHARED_ARGS $FLAG_GIDQN_ARGS

  done
done



