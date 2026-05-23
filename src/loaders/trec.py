import pandas as pd
from tqdm.auto import tqdm

TREC_PATH = "./data/trec"

def load_trec(subset=None, seed=0):
    print("Chargement des fichiers TREC …")

    queries_str_train = pd.read_csv(f"{TREC_PATH}/queries.train.tsv", names=["qid","query_string"], sep="\t")\
        .assign(qid=lambda x: x["qid"].astype(int))\
        .set_index("qid")["query_string"].to_dict()

    queries_str_dev = pd.read_csv(f"{TREC_PATH}/queries.dev.tsv", names=["qid","query_string"], sep="\t")\
        .assign(qid=lambda x: x["qid"].astype(int))\
        .set_index("qid")["query_string"].to_dict()

    print("  Chargement collection …")
    collection_str = pd.read_csv(f"{TREC_PATH}/collection.tsv", sep="\t", names=["docid","document_string"])\
        .set_index("docid")["document_string"].to_dict()

    qrels_train = pd.read_csv(f"{TREC_PATH}/qrels.train.tsv", sep="\t", names=["topicid","_","docid","rel"])
    qrels_dev = pd.read_csv(f"{TREC_PATH}/qrels.dev.tsv", sep="\t", names=["topicid","_","docid","rel"])

    print("  Construction train …")
    train_df = pd.DataFrame([
        {"query": queries_str_train[r.topicid],
         "passage": collection_str[r.docid],
         "qid": r.topicid, "docid": r.docid}
        for r in tqdm(qrels_train.sort_values("topicid").itertuples(),total=len(qrels_train))
        if r.topicid in queries_str_train and r.docid in collection_str
    ])

    print("  Construction dev/test …")
    all_dev = pd.DataFrame([
        {"query": queries_str_dev[r.topicid],
         "passage": collection_str[r.docid],
         "qid": r.topicid, "docid": r.docid}
        for r in tqdm(qrels_dev.sort_values("topicid").itertuples(),total=len(qrels_dev))
        if r.topicid in queries_str_dev and r.docid in collection_str
    ])
    n = len(all_dev)
    val_df = all_dev.iloc[:n//2].reset_index(drop=True)
    test_df = all_dev.iloc[n//2:].reset_index(drop=True)

    print(f"  -> train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}")

    if subset:
        train_df = train_df.sample(n=subset, random_state=seed).reset_index(drop=True)
        val_df = val_df.sample(n=min(subset//8, len(val_df)),random_state=seed).reset_index(drop=True)
        test_df = test_df.sample(n=min(subset//8, len(test_df)), random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df, collection_str