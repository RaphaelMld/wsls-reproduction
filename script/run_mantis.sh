#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.." || exit

export TRANSFORMERS_OFFLINE=1
DATASET="mantis"
INSTANCES=100000
SEEDS=(0 1 2 3 4)

METHODS=("random" "BM25") 
TEST_METHODS=("random" "BM25")

echo "=== DÉMARRAGE DES EXPÉRIMENTATIONS ==="

for METHOD in "${METHODS[@]}"; do
    for TEST_METHOD in "${TEST_METHODS[@]}"; do
        
        RESULT_FILE="results_${DATASET}_${METHOD}_test_${TEST_METHOD}.csv"
        
        echo "------------------------------------------------------------"
        echo " CONFIG : Train=$METHOD | Test=$TEST_METHOD | Output=$RESULT_FILE"
        echo "------------------------------------------------------------"

        for SEED in "${SEEDS[@]}"; do
            echo " >>> SEED: $SEED"

            # Baseline (One-Hot, eps=0.0)
            python main.py --dataset $DATASET --method $METHOD --test_method $TEST_METHOD --mode baseline --eps 0.0 --instances $INSTANCES --seed $SEED --results_file $RESULT_FILE

            # Label Smoothing
            python main.py --dataset $DATASET --method $METHOD --test_method $TEST_METHOD --mode ls --eps 0.2 --instances $INSTANCES --seed $SEED --results_file $RESULT_FILE
            python main.py --dataset $DATASET --method $METHOD --test_method $TEST_METHOD --mode ls --eps 0.4 --instances $INSTANCES --seed $SEED --results_file $RESULT_FILE

            # Curriculum Learning (T-WSLS)
            python main.py --dataset $DATASET --method $METHOD --test_method $TEST_METHOD --mode twsls --eps 0.2 --instances $INSTANCES --seed $SEED --results_file $RESULT_FILE
            python main.py --dataset $DATASET --method $METHOD --test_method $TEST_METHOD --mode twsls --eps 0.4 --instances $INSTANCES --seed $SEED --results_file $RESULT_FILE
        done
    done
done

echo "=== TOUT EST TERMINÉ ! ==="