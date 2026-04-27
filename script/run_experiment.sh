#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.." || exit

export TRANSFORMERS_OFFLINE=1
DATASET="mantis"
MODE="twsls"
EPS=0.4
INSTANCES=100000
CSV_FILE="results_ablation_beta_mantis.csv"
SEEDS=(0 1 2 3 4)

echo "Les résultats seront sauvegardés dans : $CSV_FILE"

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "   SEED : $SEED"
    echo "=========================================="

    COMMON_ARGS="--dataset $DATASET --mode $MODE --eps $EPS --instances $INSTANCES --save_history --results_file $CSV_FILE --seed $SEED"

    # Baseline Step (T-WSLS - eps=0.4)
    echo ">> Profil : STEP (Baseline)"
    python main.py $COMMON_ARGS --decay step
    
    # Beta
    echo ">> BETA Linéaire (1,1)"
    python main.py $COMMON_ARGS --decay beta --alpha 1.0 --beta 1.0
    
    echo ">> BETA Late Drop (5,1)"
    python main.py $COMMON_ARGS --decay beta --alpha 5.0 --beta 1.0
    
    echo ">> BETA Early Drop (1,5)"
    python main.py $COMMON_ARGS --decay beta --alpha 1.0 --beta 5.0
    
    echo ">> BETA S-Curve (3,3)"
    python main.py $COMMON_ARGS --decay beta --alpha 3.0 --beta 3.0
done

echo "------------------------------------------------------------"
echo "Toutes les expériences sont terminées"