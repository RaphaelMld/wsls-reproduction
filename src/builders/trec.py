import pandas as pd
import random
import os
from tqdm.auto import tqdm
from src.negative_sampling import get_negatives_trec

DATA_DIR = "./data"

def build_split_parquet_trec(split_name, split_df, index,method, collection_str):
    out_path = f"{DATA_DIR}/trec_{method}_{split_name}.parquet"
    if os.path.exists(out_path):
        print(f"  {out_path} déjà existant, skip.")
        return

    rows, skipped = [], 0
    for _, row in tqdm(split_df.iterrows(), total=len(split_df),desc=f"Build {split_name}"):
        pos_docno = f"d{row['docid']}"
        negs = get_negatives_trec(method, row["query"],pos_docno, index, collection_str)
        if negs is None or len(negs) < 9:
            skipped += 1
            continue

        group = [{
            "query": row["query"], "text": row["passage"],
            "label": 1.0, "score": -1.0, "docno": pos_docno,
        }] + [{
            "query": row["query"], "text": neg["text"],
            "label": 0.0, "score": float(neg["score"]),
            "docno": neg["docno"],
        } for _, neg in negs.iterrows()]

        random.shuffle(group)
        rows.extend(group)

    df_out = pd.DataFrame(rows)[["query","text","label","score","docno"]]
    df_out.to_parquet(out_path, index=False)
    print(f"  -> {out_path} | {len(df_out):,} lignes | "
          f"{len(df_out)//10:,} groupes | {skipped} skipped")