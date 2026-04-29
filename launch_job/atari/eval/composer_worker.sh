#!/bin/bash

VIDEO_DIR=$1
OUTPUT_DIR=$2

SEED=$SLURM_ARRAY_TASK_ID
VIDEO_DIR="$VIDEO_DIR/seed$SEED"
OUTPUT_DIR="$OUTPUT_DIR/seed$SEED"

source env/bin/activate
python3 compose_eval_videos.py --videos_dir $VIDEO_DIR --include native ale agent --output_dir $OUTPUT_DIR
