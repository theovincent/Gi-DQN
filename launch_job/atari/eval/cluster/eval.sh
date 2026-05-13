#!/bin/bash



for GAME in Breakout; do

EXPERIMENT_NAME="L2_K5_WD1_PX16_K2_S2__ST2_SK4_ncnv1_nfc2_F_c16_f256_LN11cnn_LR25e-5_NE30_LINEAR_T600_${GAME}"
ALGO_NAME="gidqnshared"

FIRST_SEED=1
LAST_SEED=2

N_EVAL_EPISODES=3
FPS=15          # 15 = real-time (60 Hz ALE / 4 frame-skip)


EXPERIMENT_PATH="experiments/atari/exp_output/${EXPERIMENT_NAME}"
LOG_DIR="experiments/atari/logs/eval/${EXPERIMENT_NAME}/${ALGO_NAME}"
mkdir -p "$LOG_DIR"


echo "Submitting eval: ${ALGO_NAME} on ${EXPERIMENT_NAME} (seeds ${FIRST_SEED}–${LAST_SEED})"

sbatch \
    --job-name eval-${ALGO_NAME}_${GAME} \
    --array=${FIRST_SEED}-${LAST_SEED} \
    --cpus-per-task=2 \
    --mem-per-cpu=1500M \
    --time=03:30:00 \
    --partition stud \
    --output=${LOG_DIR}/eval_%a.out \
    launch_job/atari/eval/eval_worker.sh \
        "${EXPERIMENT_PATH}" "${ALGO_NAME}" "${N_EVAL_EPISODES}" "${FPS}"

done
