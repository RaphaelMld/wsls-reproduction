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


def load_mantis(method_name):
    cfg = DATASET_CONFIG["mantis"]
    for split in ["train", "valid", "test"]:
        source_file = f"{DATA_DIR}/mantis/{split}.tsv"
        if not os.path.exists(source_file):
            print(f"Erreur : {source_file} introuvable")
            exit(1)

        print(f"Chargement {source_file} …")
        df_raw = pd.read_csv(source_file, sep="\t")
        queries, corpus, labels = process_df(df_raw, cfg)
        index_path = f"{DATA_DIR}/index_mantis_{split}"
        parquet_path = f"{DATA_DIR}/mantis_{method_name}_{split}.parquet"
        index = build_index_from_corpus(corpus, index_path)
        get_or_create_parquet(queries, corpus, labels, index, method=method_name, cfg=cfg,file_path=parquet_path)