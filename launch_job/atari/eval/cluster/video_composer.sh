#!/bin/bash


FIRST_SEED=1
LAST_SEED=3




for ALGO in dqn gidqnshared; do 
for GAME in Atlantis Qbert Phoenix Enduro Boxing; do

if [ $ALGO == dqn ]; then
EXPERIMENT_NAME_GAME="L2_PX16_K2_S2__ST2_SK4_ncnv1_nfc2_F_c16_f256_LN11cnn_LR25e-5_NE30_T600_$GAME"
elif [ $ALGO == gidqnshared ]; then
EXPERIMENT_NAME_GAME="L2_K5_WD1_PX16_K2_S2__ST2_SK4_ncnv1_nfc2_F_c16_f256_LN11cnn_LR25e-5_NE30_LINEAR_T600_${GAME}"
fi


VIDEO_DIR="../Gi-DQN/experiments/atari/exp_output/$EXPERIMENT_NAME_GAME/$ALGO/videos"
OUTPUT_DIR="threeway/$GAME/$ALGO"
LOG_DIR="experiments/atari/logs_composed_videos/$EXPERIMENT_NAME_GAME/$ALGO"
echo "Submitting video composer: $ALGO on $EXPERIMENT_NAME_GAME for $GAME ; Seeds $FIRST_SEED - $LAST_SEED"

sbatch --job-name video-composer-$ALGO-$GAME \
       --array=$FIRST_SEED-$LAST_SEED \
       --cpus-per-task=1 \
       --mem-per-cpu=2000M \
       --time=01:00:00 \
       --partition stud \
       --output=$LOG_DIR/composed_videos_%a.out \
       launch_job/atari/eval/composer_worker.sh \
       "$VIDEO_DIR" "$OUTPUT_DIR"
done 
done
