import pandas as pd
import os
from src.config import DATASET_CONFIG
from src.indexing import build_index_from_corpus
from src.builders.standard import get_or_create_parquet

DATA_DIR = "./data"

def process_df(df, cfg):
    df = df.copy()
    df["label"] = 1
    df["qid"] = "q" + (df.index + 1).astype(str)
    df["docno"] = "d" + (df.index + 1).astype(str)
    df = df.rename(columns={cfg["query_col"]: "query", cfg["text_col"]: "text"})
    return df[["qid","query"]], df[["docno","text"]], df[["qid","docno","label"]]


def load_qqp(method_name):
    cfg = DATASET_CONFIG["qqp"]
    source_file = cfg["source_file"]

    if not os.path.exists(source_file):
        print(f"Erreur : {source_file} introuvable")
        exit(1)

    print(f"Chargement {source_file} …")
    df_raw = pd.read_csv(source_file).dropna(subset=["question1","question2"])
    df_raw = df_raw[df_raw["is_duplicate"] == 1].copy()
    total_len = len(df_raw)

    splits = {
        "train": df_raw.iloc[:int(total_len * 0.8)].copy(),
        "valid": df_raw.iloc[int(total_len * 0.8):int(total_len * 0.9)].copy(),
        "test": df_raw.iloc[int(total_len * 0.9):].copy(),
    }

    for split_name, df_split in splits.items():
        print(f"\n>>> Split : {split_name} ({len(df_split)} requêtes)")
        queries, corpus, labels = process_df(df_split, cfg)
        index_path = f"{DATA_DIR}/index_qqp_{method_name}_{split_name}"
        parquet_path = f"{DATA_DIR}/qqp_{method_name}_{split_name}.parquet"
        index = build_index_from_corpus(corpus, index_path)
        get_or_create_parquet(queries, corpus, labels, index,method=method_name, cfg=cfg,file_path=parquet_path)