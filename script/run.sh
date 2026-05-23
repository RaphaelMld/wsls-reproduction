#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.." || exit

SEEDS=(0 1 2 3 4)
EPSILONS=(0.1 0.2 0.3 0.4)
DATASETS=("mantis" "qqp" "trec")

# Instances par dataset (comme dans le papier)
declare -A INSTANCES
INSTANCES["mantis"]=100000
INSTANCES["qqp"]=100000
INSTANCES["trec"]=50000

echo "=== EXPÉRIENCES COMPLÈTES : mantis | qqp | trec ==="
echo "=== Seeds: ${SEEDS[*]} ==="

for DATASET in "${DATASETS[@]}"; do

    N_INSTANCES=${INSTANCES[$DATASET]}
    RESULT_BM25="results_${DATASET}_final_BM25.csv"
    RESULT_RANDOM="results_${DATASET}_final_random.csv"

    echo ""
    echo "║  DATASET : $DATASET (instances=$N_INSTANCES)"


    # NSBM25 
    echo ""
    echo "=== $DATASET | NSBM25 ==="

    for SEED in "${SEEDS[@]}"; do
        echo "--- BM25 | SEED=$SEED ---"

        python main.py --dataset $DATASET --method BM25 \
            --mode baseline --eps 0.0 \
            --instances $N_INSTANCES --seed $SEED \
            --results_file $RESULT_BM25

        for EPS in "${EPSILONS[@]}"; do

            python main.py --dataset $DATASET --method BM25 \
                --mode ls --eps $EPS \
                --instances $N_INSTANCES --seed $SEED \
                --results_file $RESULT_BM25

            python main.py --dataset $DATASET --method BM25 \
                --mode tls --eps $EPS --decay step \
                --instances $N_INSTANCES --seed $SEED \
                --results_file $RESULT_BM25

            python main.py --dataset $DATASET --method BM25 \
                --mode twsls --eps $EPS --decay step \
                --instances $N_INSTANCES --seed $SEED \
                --results_file $RESULT_BM25

        done
    done

    #  NSrandom 
    echo ""
    echo "=== $DATASET | NSrandom (train) + BM25 (test) ==="

    for SEED in "${SEEDS[@]}"; do
        echo "--- random | SEED=$SEED ---"

        python main.py --dataset $DATASET \
            --method random --test_method BM25 \
            --mode baseline --eps 0.0 \
            --instances $N_INSTANCES --seed $SEED \
            --results_file $RESULT_RANDOM

        for EPS in "${EPSILONS[@]}"; do

            python main.py --dataset $DATASET \
                --method random --test_method BM25 \
                --mode ls --eps $EPS \
                --instances $N_INSTANCES --seed $SEED \
                --results_file $RESULT_RANDOM

            python main.py --dataset $DATASET \
                --method random --test_method BM25 \
                --mode tls --eps $EPS --decay step \
                --instances $N_INSTANCES --seed $SEED \
                --results_file $RESULT_RANDOM

            python main.py --dataset $DATASET \
                --method random --test_method BM25 \
                --mode twsls --eps $EPS --decay step \
                --instances $N_INSTANCES --seed $SEED \
                --results_file $RESULT_RANDOM

        done
    done

    echo ""
    echo "-> $DATASET terminé : res/$RESULT_BM25 | res/$RESULT_RANDOM"

done

echo ""
echo "=== TOUT EST TERMINÉ ==="
echo ""
echo "Résultats générés :"
for DATASET in "${DATASETS[@]}"; do
    echo "  res/results_${DATASET}_final_BM25.csv"
    echo "  res/results_${DATASET}_final_random.csv"
done
echo ""
echo "Total runs : 3 datasets x 130 runs = 390 runs"