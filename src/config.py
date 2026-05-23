DATASET_CONFIG = {
    "mantis": {
        "format":        "tsv",
        "query_col":     "context",
        "text_col":      "response",
        "split":         "files",
        "bm25_controls": {},
        "score_norm":    "minmax",
    },
    "qqp": {
        "format":        "csv",
        "query_col":     "question1",
        "text_col":      "question2",
        "split":         "inline",
        "source_file":   "/users/Etu0/21500150/Documents/RI/datasets/qqp/train.csv",
        "bm25_controls": {"c": 0.9, "bm25.b": 0.4},
        "score_norm":    "minmax_clipped",
    },
    "trec": {
        "format":        "trec",
        "bm25_controls": {},
        "score_norm":    "minmax",
    },
}