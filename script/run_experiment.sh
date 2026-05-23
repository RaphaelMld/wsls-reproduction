#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.." || exit

export TRANSFORMERS_OFFLINE=1
export JAVA_HOME="C:/Program Files/Eclipse Adoptium/jdk-11.0.31.11-hotspot"
export PATH="$JAVA_HOME/bin/server:$PATH"
export PYTHONPATH="$(pwd)"

DATASET="mantis"
METHOD="BM25"
MODE="twsls"
EPS=0.4
INSTANCES=100000
CSV_FILE="results_ablation_beta_mantis.csv"
SEEDS=(0 1 2 3 4)

echo "Dataset : $DATASET"
echo "Mode : $MODE | eps=$EPS | instances=$INSTANCES"
echo "Résultats : res/$CSV_FILE"
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "   SEED : $SEED"
    echo "=========================================="

    COMMON_ARGS="--dataset $DATASET --method $METHOD \
        --mode $MODE --eps $EPS \
        --instances $INSTANCES \
        --seed $SEED \
        --save_history \
        --results_file $CSV_FILE"

    # Baseline Step (T-WSLS - eps=0.4)
    echo ">> Profil : STEP (Baseline)"
    python main.py $COMMON_ARGS --decay step

    # Beta Linéaire (1,1)
    echo ">> BETA Linéaire (1,1)"
    python main.py $COMMON_ARGS --decay beta --alpha 1.0 --beta 1.0

    # Beta Late Drop (5,1)
    echo ">> BETA Late Drop (5,1)"
    python main.py $COMMON_ARGS --decay beta --alpha 5.0 --beta 1.0

    # Beta Early Drop (1,5)
    echo ">> BETA Early Drop (1,5)"
    python main.py $COMMON_ARGS --decay beta --alpha 1.0 --beta 5.0

    # Beta S-Curve (3,3)
    echo ">> BETA S-Curve (3,3)"
    python main.py $COMMON_ARGS --decay beta --alpha 3.0 --beta 3.0

done

echo "------------------------------------------------------------"
echo "Toutes les expériences sont terminées"
echo "Résultats sauvegardés dans : res/$CSV_FILE"