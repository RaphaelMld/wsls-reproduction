# WSLS-Reproduction

> Reproduction of the paper: "Weakly Supervised Label Smoothing" (ECIR 2021).

## About The Project

This repository contains a PyTorch implementation of the methods described in the paper [*Weakly Supervised Label Smoothing*](https://arxiv.org/abs/2012.08575) by Gustavo Penha and Claudia Hauff. This work was conducted as part of an academic project in Information Retrieval (IR).

The paper proposes T-WSLS (Two-stage Weakly Supervised Label Smoothing), a Curriculum Learning approach for fine-tuning pointwise BERT rankers. Instead of relying solely on sparse human labels (one-hot encoding) or pure weak supervision, T-WSLS combines both dynamically during training.

We evaluate this approach on several Conversational Response Ranking datasets (including MANTIS, TREC 2020 or QQP).

## Repository Structure


* `data_preparation/`: Scripts to generate training data using BM25 negative sampling.
* `models/`: PyTorch training loops, custom dataset classes, and evaluation scripts (e.g., R_10@1 metric).
* `notebooks/`: Jupyter notebooks used for experiments and result visualization.

## Requirements

To run the code, the following dependencies are required:

```bash
pip install torch transformers pandas numpy tqdm pyterrier
