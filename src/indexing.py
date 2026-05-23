import os
from tqdm.auto import tqdm
import pyterrier as pt

TREC_PATH = "./data/trec"

def build_index_from_corpus(corpus_df, index_path):
    index_path = os.path.abspath(index_path)
    os.makedirs(index_path, exist_ok=True)
    if os.path.exists(os.path.join(index_path, "data.properties")):
        print(f"  Index existant : {index_path}")
        return pt.IndexFactory.of(index_path)
    print(f"  Construction index : {index_path} …")
    indexer = pt.IterDictIndexer(index_path, overwrite=True)
    index_ref = indexer.index(corpus_df[["docno","text"]].to_dict(orient="records"))
    return pt.IndexFactory.of(index_ref)


def build_full_index_trec(index_path):
    index_path = os.path.abspath(index_path)
    os.makedirs(index_path, exist_ok=True)
    if os.path.exists(os.path.join(index_path, "data.properties")):
        print(f"  Index TREC existant : {index_path}")
        return pt.IndexFactory.of(index_path)
    print("  Indexation complète TREC …")
    def collection_iter():
        with open(f"{TREC_PATH}/collection.tsv", "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Indexation", total=8_841_823):
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    yield {"docno": f"d{parts[0]}", "text": parts[1]}
    indexer = pt.IterDictIndexer(index_path, overwrite=True)
    index_ref = indexer.index(collection_iter())
    return pt.IndexFactory.of(index_ref)