
# WSLS-Reproduction

> Reproduction of the paper: "Weakly Supervised Label Smoothing" (ECIR 2021).

## About The Project

This repository contains a PyTorch implementation of the methods described in the paper [*Weakly Supervised Label Smoothing*](https://arxiv.org/abs/2012.08575) by Gustavo Penha and Claudia Hauff. This work was conducted as part of an academic project in Information Retrieval (IR) at Sorbonne University.

The paper proposes T-WSLS (Two-stage Weakly Supervised Label Smoothing), a Curriculum Learning approach for fine-tuning pointwise BERT rankers. Instead of relying solely on sparse human labels (one-hot encoding) or pure weak supervision, T-WSLS combines both dynamically during training.

We evaluate this approach on Conversational Response Ranking datasets (e.g., MANTIS, TREC 2020, QQP).

---

## Repository Structure

* `download.py`: Script to download and fetch the required datasets (creates `.tsv` files).
* `main.py`: Main entry point to run the training and evaluation pipeline.
* `src/`: Core source code modules:
  * `data_prep.py`: Processes datasets, builds PyTerrier indices, and generates negative samples (BM25 or random) saved as `.parquet` files.
  * `dataset.py`: PyTorch custom dataset classes and data loaders.
  * `train.py`: Training loops, loss computation, and T-WSLS curriculum learning implementation.
  * `evaluate.py`: Evaluation functions and metrics computation (e.g., R_10@1).
* `notebooks/`: Jupyter notebooks used for preliminary experiments, data exploration, and result visualization.

---

## Requirements

To run the code, the following dependencies are required. We recommend using a virtual environment.

```bash
pip install torch transformers pandas numpy tqdm pyterrier gdown
```

---

## Usage

Here is the step-by-step workflow to reproduce our experiments:

### 1. Download the Data
First, download the required datasets. This step will fetch the raw data and place the corresponding `train.tsv`, `valid.tsv`, and `test.tsv` files in the `data/<dataset_name>/` directory.

```bash
python download.py --dataset mantis
```
*Arguments:*
* `--dataset`: Name of the dataset folder (e.g., `mantis`, `qqp`, `trec`).

### 2. Data Preparation & Negative Sampling
Once downloaded, you must prepare the data for training. The `data_prep.py` script builds a PyTerrier index and generates hard negatives (using either `BM25` or `random` sampling). It outputs `.parquet` files ready for the PyTorch DataLoader.

```bash
# Example: Prepare MANTIS dataset using BM25 negative sampling
python src/data_prep.py --dataset mantis --method BM25
```
*Arguments:*
* `--dataset`: Name of the dataset folder (e.g., `mantis`, `qqp`, `trec`).
* `--method`: Negative sampling strategy (`BM25` or `random`). Defaults to `BM25`.


### 3. Run the Training Pipeline
Finally, use the main script to start the training and evaluation loops. The script will automatically load the `.parquet` files generated in the previous step and save the trained model in the `./models/` directory.

```bash
# Example command to run the T-WSLS method on MANTIS
python main.py --dataset mantis --method BM25 --mode twsls --eps 0.2 --instances 100000
```

*Arguments for `main.py`:*
* `--dataset` (Required): Name of the dataset (e.g., `mantis`, `qqp`, `trec`).
* `--method` (Optional): Negative sampling method used during data prep (`BM25` or `random`). Defaults to `BM25`.
* `--mode` (Required): Training mode to use (`baseline`, `ls` for standard label smoothing, or `twsls` for Two-stage WSLS).
* `--eps` (Optional): Epsilon value for label smoothing. Defaults to `0.2`.
* `--instances` (Optional): Total number of instances to process. Defaults to `100000`.
  
