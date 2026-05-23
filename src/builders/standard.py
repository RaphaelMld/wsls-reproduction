import pandas as pd
import os
from tqdm.auto import tqdm
from src.negative_sampling import get_negatives_standard

def get_or_create_parquet(queries, corpus, labels, index,method, cfg, k=9, file_path="data.parquet"):
    if os.path.exists(file_path):
        print(f"  Chargement depuis {file_path}")
        return pd.read_parquet(file_path)

    all_groups = []
    for i in tqdm(range(len(queries)), desc="Génération"):
        request = queries.iloc[i]
        qid, query = request["qid"], request["query"]
        ground_truth = labels[(labels["qid"]==qid) & (labels["label"]==1)]["docno"].item()

        negs = get_negatives_standard(method, query, ground_truth, corpus, index, cfg, k)
        negs.drop(columns=["rank","docid"], errors="ignore", inplace=True)
        if "score" not in negs.columns:
            negs["score"] = 0.0

        pos_row = pd.DataFrame([{
            "docno": ground_truth, "score": -1.0,
            "text":  corpus[corpus["docno"]==ground_truth]["text"].item()
        }])

        group = pd.concat([negs, pos_row], ignore_index=True)
        group["query"] = query
        group["label"] = (group["docno"] == ground_truth).astype(float)
        group = group.sample(frac=1, random_state=i).reset_index(drop=True)
        all_groups.append(group)

    df_out = pd.concat(all_groups, ignore_index=True)[
        ["query","text","label","score","docno"]
    ]
    df_out.to_parquet(file_path, index=False)
    print(f"  -> {file_path} ({len(df_out):,} lignes | {len(df_out)//10:,} groupes)")
    return df_out