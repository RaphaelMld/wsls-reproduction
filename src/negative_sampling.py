import re
import random
import pandas as pd
import pyterrier as pt


def normalize_scores(scores, mode):
    s_min, s_max = scores.min(), scores.max()
    if mode == "minmax":
        if s_max > s_min:
            return (scores - s_min) / (s_max - s_min)
        return pd.Series([1.0] * len(scores), index=scores.index)
    elif mode == "minmax_clipped":
        if s_max > s_min:
            return 0.01 + 0.98 * ((scores - s_min) / (s_max - s_min))
        return pd.Series([0.99] * len(scores), index=scores.index)
    return scores


def get_negatives_standard(method, query, ground_truth,corpus, index, cfg, k=9):
    if method == "random":
        filtered = corpus[corpus["docno"] != ground_truth]
        sample = filtered.sample(n=min(k, len(filtered))).copy()
        sample["score"] = 0.0
        return sample[["docno","text","score"]]

    controls = cfg.get("bm25_controls", {})
    retriever = pt.terrier.Retriever(index, wmodel="BM25",
                                       controls=controls, num_results=k + 5)
    safe_query = re.sub(r'[^\w\s]', ' ', str(query))
    results = retriever.search(safe_query)
    results = results[results["docno"] != ground_truth].head(k).copy()
    results["score"] = normalize_scores(results["score"], cfg["score_norm"])
    results = results.merge(corpus[["docno","text"]], on="docno", how="left")
    results.drop(columns=["rank","docid","qid","query"], errors="ignore", inplace=True)
    return results[["docno","text","score"]]


def get_negatives_trec(method, query, pos_docno,index, collection_str, k=9):
    if method == "random":
        pos_docid = int(pos_docno[1:])
        all_docids = [d for d in collection_str.keys() if d != pos_docid]
        sampled_ids = random.sample(all_docids, min(k, len(all_docids)))
        return pd.DataFrame([{
            "docno": f"d{d}", "text": collection_str[d], "score": 0.0
        } for d in sampled_ids])

    retriever = pt.terrier.Retriever(index, wmodel="BM25", num_results=k + 10)
    safe_query = re.sub(r'[^\w\s]', ' ', str(query))
    try:
        results = retriever.search(safe_query)
    except Exception:
        return None

    results = results[results["docno"] != pos_docno].head(k).copy()
    if len(results) < k:
        return None
    results["score"] = normalize_scores(results["score"], "minmax")
    results["text"] = results["docno"].apply(lambda d: collection_str.get(int(d[1:]), ""))
    results.drop(columns=["rank","docid","qid","query"], errors="ignore", inplace=True)
    return results[["docno","text","score"]]