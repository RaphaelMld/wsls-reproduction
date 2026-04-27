import pandas as pd
import pyterrier as pt
import os
import re
import numpy as np
import argparse
from tqdm.auto import tqdm
import random

if not pt.started():
    pt.init()

def process_dataset(df):
    """
    processing du dataset
    Args:
        df (pd.DataFrame) : dataframe contenant deux colonnes, context et response
    Returns:
        queries, corpus, label
    """
    df["label"] = 1
    df['qid'] = "q" + (df.index + 1).astype(str)
    df['docno'] = "d" + (df.index + 1).astype(str)

    # Renommage générique
    df = df.rename(columns={
        "context" : "query",
        "response" : "text"
    })

    queries = df[["qid", "query"]]
    corpus = df[["docno", "text"]]
    label = df[["qid", "docno", "label"]]

    return queries, corpus, label


def build_index(corpus, index_path):
    """
    Création de l'index dynamique en fonction du chemin passé en paramètre
    """
    if os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "data.properties")):
        print(f"Chargement de l'index depuis '{index_path}'")
        index = pt.IndexFactory.of(os.path.abspath(index_path))
    else:
        print("Création de l'index PyTerrier en cours...")
        indexer = pt.IterDictIndexer(index_path, fields=True)
        index_ref = indexer.index(corpus.to_dict(orient="records"))
        index = pt.IndexFactory.of(index_ref)
    return index


def negative_sampling_dataset(method, qid, query, ground_truth, corpus, index, k=10):
    """
    Negative sampling 
    """
    if method =="random":
        filtered_corpus = corpus[corpus["docno"] != ground_truth]
        random_samples = filtered_corpus.sample(n=k).copy()
        random_samples["qid"] = qid
        random_samples["query"] = query
        return random_samples
        
    else:
        retriever = pt.terrier.Retriever(index, wmodel=method, num_results=k)
        safe_query = re.sub(r'[^\w\s]', ' ', query)
        
        results_df = retriever.search(safe_query)
        results_df["query"] = query
        results_df["qid"] = qid
        
        filtered_df = results_df[results_df["docno"] != ground_truth]
        top_k_df = filtered_df.head(k).copy()

        score_min = top_k_df["score"].min()
        score_max = top_k_df["score"].max()
        
        if score_max > score_min:
            top_k_df["score"] = (top_k_df["score"] - score_min) / (score_max - score_min)
        else:
            top_k_df["score"] = 1.0
        
        top_k_with_content = top_k_df.merge(corpus, on="docno", how="left")
        return top_k_with_content


def build_query_context(method, request, corpus, label, index, k=9):
    """
    Prépare les données pour une requête
    """
    qid, query = request
    ground_truth = label[(label["qid"] == qid) & (label["label"] == 1)]["docno"].item()
    
    negative_sampling = negative_sampling_dataset(method, qid, query, ground_truth, corpus, index, k)
    negative_sampling.drop(columns=["rank", "docid"], errors="ignore", inplace=True)
    
    if "score" not in negative_sampling.columns:
        negative_sampling["score"] = 0

    row_ground_truth = pd.DataFrame({
        "qid": [qid],
        "query": [query],
        "docno": [ground_truth], 
        "score": [0.5], 
        "text": [corpus[corpus["docno"]==ground_truth]["text"].item()]
    })
    
    df = pd.concat([negative_sampling, row_ground_truth], ignore_index=True)
    df["label"] = (df["docno"] == ground_truth).astype(int)
    
    return df


def get_or_create_training_data(queries, corpus, labels, index, method="BM25", k=9, file_path="training_data.parquet"):
    if os.path.exists(file_path):
        df_dataset = pd.read_parquet(file_path)
        print(f"Chargement terminé depuis {file_path}")
        return df_dataset
        
    print(f"Le fichier n'existe pas, génération")
    all_contexts = []
    
    for i in tqdm(range(len(queries)), desc="Génération des contextes"):
        request = queries.iloc[i]
        df_context = build_query_context(method, request, corpus, labels, index, k)
        all_contexts.append(df_context)
        
    df_dataset = pd.concat(all_contexts, ignore_index=True)
    
    print(f"Sauvegarde du dataset : {file_path}")
    df_dataset.to_parquet(file_path, index=False)
    
    return df_dataset



# python src/data_prep.py --dataset mantis --method BM25
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Nom du dataset (ex: mantis, qqp, trec)")
    parser.add_argument("--method", type=str, default="BM25", choices=["BM25", "random"], help="Méthode de negative sampling (BM25 ou random)")
    args = parser.parse_args()

    dataset_name = args.dataset
    method_name = args.method
    path_data = f"data/{dataset_name}"
    os.makedirs(path_data, exist_ok=True)

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)

    for split in ["train", "valid", "test"]:
        source_file = os.path.join(path_data, f"{split}.tsv")
        
        if not os.path.exists(source_file):
            print(f"erreur, le fichier est introuvable : {source_file}")
            exit(1)
                
        print(f"Fichier {source_file} trouvé")
        df_train_raw = pd.read_csv(source_file, sep="\t")
        
        queries_data, corpus_data, label_data = process_dataset(df_train_raw)
        
        index_path = f"./data/index_{dataset_name}_{split}"
        parquet_path = f"data/{dataset_name}_{method_name}_{split}.parquet"
       
        
        index_data = build_index(corpus_data, index_path=index_path)
        
        training_data_df = get_or_create_training_data(
            queries=queries_data, 
            corpus=corpus_data, 
            labels=label_data, 
            index=index_data, 
            method=method_name, 
            k=9,
            file_path=parquet_path
        )
        
    
        
