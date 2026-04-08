#!/bin/bash



seeds=(1)
game="BattleZone"
algo="idqnshared"   # dqn | gidqnshared | idqnshared | dqnrcshared
LR=10e-5
tup=100             

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SH="${SCRIPT_DIR}/launch_job/atari/launch_restart.sh"



if [ "$algo" != "dqn" ]; then
    LINEAR="_LINEAR"
    K="K5_"
else
    LINEAR=""
    K=""
fi

#
if [ "$algo" == "dqnrcshared" ]; then
    TUP_FRAGMENT=""          # no _T{tup} in path
else
    TUP_FRAGMENT="_T${tup}"
fi

NE=30
LR_TAG="LR${LR}_NE${NE}"
BASE="L2_${K}LS1_ST2_SK4_ncnv1_nfc2_F_c16_f128_LN11cnn_${LR_TAG}_NE${NE}${TUP_FRAGMENT}_${game}"


if [ "$algo" == "dqnrcshared" ]; then
    EXP_NAME="L2_LS1_ST2_SK4_ncnv1_nfc2_F_c16_f128_LN11cnn_${LR_TAG}${LINEAR}_${game}"
else
    EXP_NAME="L2_${K}LS1_ST2_SK4_ncnv1_nfc2_F_c16_f128_LN11cnn_${LR_TAG}${LINEAR}${TUP_FRAGMENT}_${game}"
fi

echo "================================================"
echo " Restarting crashed seeds: ${seeds[*]}"
echo " algo=${algo}  game=${game}  LR=${LR}  tup=${tup}"
echo " experiment: ${EXP_NAME}"
echo "================================================"


for seed in "${seeds[@]}"; do
    echo ""
    echo "--- Seed ${seed}: deleting artifacts ---"

    LOG="experiments/atari/logs/${EXP_NAME}/${algo}/train_${seed}.out"
    RETURNS="experiments/atari/exp_output/${EXP_NAME}/${algo}/episode_returns_and_lengths/${seed}.json"
    MODEL="experiments/atari/exp_output/${EXP_NAME}/${algo}/models/${seed}"

    rm -rf -v "$LOG"
    rm -rf -v "$RETURNS"
    rm -rf -v "$MODEL"

    echo "--- Seed ${seed}: relaunching ---"
    bash "$LAUNCH_SH" \
        --algo   "$algo" \
        --lr     "$LR" \
        --tup    "$tup" \
        --game   "$game" \
        --first_seed "$seed" \
        --last_seed  "$seed"
done

echo ""
echo "All seeds restarted."
