# WSLS-Reproduction

> Reproduction of the paper: "Weakly Supervised Label Smoothing" (ECIR 2021).

## About The Project

This repository contains a PyTorch implementation of the methods described in the paper [*Weakly Supervised Label Smoothing*](https://arxiv.org/abs/2012.08575) by Gustavo Penha and Claudia Hauff. This work was conducted as part of an academic project in Information Retrieval (IR) at Sorbonne University.

The paper proposes T-WSLS (Two-stage Weakly Supervised Label Smoothing), a Curriculum Learning approach for fine-tuning pointwise BERT rankers. Instead of relying solely on sparse human labels or a uniform smoothing distribution, T-WSLS uses BM25 retrieval scores as a weak supervision signal and progressively removes smoothing during training.

We evaluate this approach on three retrieval tasks: **TREC-DL 2020** (passage retrieval), **QQP** (similar question retrieval), and **MANtIS** (conversation response ranking).

---

## Repository Structure

```
├── main.py                  # Entry point: training and evaluation
├── download.py              # Dataset download script
├── src/
│   ├── data_prep.py         # Data preparation entry point (unified for all datasets)
│   ├── config.py            # Dataset configurations (columns, BM25 params, score normalization)
│   ├── dataset.py           # PyTorch Dataset class
│   ├── train.py             # Training loop with LS / T-LS / WSLS / T-WSLS
│   ├── evaluate.py          # R10@1 evaluation metric
│   ├── indexing.py          # PyTerrier index builders
│   ├── negative_sampling.py # BM25 and random negative sampling
│   ├── loaders/             # Dataset-specific loaders
│   │   ├── mantis.py
│   │   ├── qqp.py
│   │   └── trec.py
│   └── builders/            # Parquet file builders
│       ├── standard.py      # Mantis / QQP
│       └── trec.py          # TREC
├── scripts/
│   ├── run_experiments.sh   # Full experiment script (all datasets, seeds, modes)
│   ├── plot_results.py      # Generate tables and Figure 2
│   └── statistical_tests.py # Paired t-tests (Table 1 & 2)
└── res/                     # Results (CSV files and plots)
```

---

## Requirements

```bash
pip install torch transformers pandas numpy tqdm pyterrier scipy matplotlib
```

Java 11+ is required for PyTerrier. Set `JAVA_HOME` before running any data preparation script.

---

## Usage

### 1. Download the Data

```bash
python download.py --dataset mantis   # or qqp, trec
```

### 2. Data Preparation & Negative Sampling

Builds a PyTerrier index and generates `.parquet` files with hard negatives.

```bash
# Mantis or QQP
python src/data_prep.py --dataset mantis --method BM25
python src/data_prep.py --dataset qqp    --method BM25

# TREC (full collection, requires ~3GB disk for the index)
python src/data_prep.py --dataset trec --method BM25

# TREC subset for quick testing
python src/data_prep.py --dataset trec --method BM25 --subset 5000
```

*Arguments:*
- `--dataset`: `mantis`, `qqp`, or `trec`
- `--method`: `BM25` or `random`
- `--subset` *(TREC only)*: number of training queries to sample

### 3. Training & Evaluation

```bash
python main.py --dataset trec --method BM25 --mode twsls --eps 0.4 --instances 50000 --seed 0
```

*Key arguments:*

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset name (`mantis`, `qqp`, `trec`) | required |
| `--method` | Train negative sampler (`BM25`, `random`) | `BM25` |
| `--test_method` | Test negative sampler (defaults to `--method`) | — |
| `--mode` | `baseline`, `ls`, `tls`, `wsls`, `twsls` | required |
| `--eps` | Smoothing strength ε | `0.2` |
| `--decay` | Schedule for T-LS/T-WSLS (`step`, `linear`, `exp`, `cosine`, `beta`) | `step` |
| `--instances` | Total training instances | `100000` |
| `--eval_split` | Evaluation split (`valid` for Table 1, `test` for Table 2) | `test` |
| `--seed` | Random seed | `0` |
| `--results_file` | Output CSV filename (saved in `res/`) | — |

### 4. Run All Experiments

```bash
bash scripts/run_experiments.sh
```

Runs all 390 experiments (3 datasets × 5 seeds × all modes and ε values) and saves results in `res/`.

### 5. Generate Tables and Plots

```bash
# Tables 1 & 2 + Figure 2
python scripts/plot_results.py \
    --files res/results_trec_final_BM25.csv res/results_trec_final_random.csv \
    --train_methods BM25 random \
    --eps_display 0.2

# Statistical tests (paired t-test)
python scripts/statistical_tests.py \
    --files res/results_trec_final_BM25.csv res/results_trec_final_random.csv \
    --train_methods BM25 random
```

---

## Training Modes

| Mode | Description |
|---|---|
| `baseline` | Standard BERT, no smoothing |
| `ls` | Label Smoothing (uniform distribution, constant ε) |
| `tls` | Two-stage LS: uniform smoothing → hard labels at T/2 |
| `wsls` | Weakly Supervised LS: BM25 scores replace uniform distribution |
| `twsls` | Two-stage WSLS: BM25 smoothing → hard labels at T/2 *(paper's main contribution)* |
