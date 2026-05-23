import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jnius_config
jnius_config.add_options('-Xmx4g', '-Xms1g')

import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import argparse
import random
import numpy as np

from src.loaders.mantis import load_mantis
from src.loaders.qqp import load_qqp
from src.loaders.trec import load_trec
from src.indexing import build_full_index_trec
from src.builders.trec import build_split_parquet_trec

DATA_DIR = "./data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mantis","qqp","trec"])
    parser.add_argument("--method", default="BM25", choices=["BM25","random"])
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "trec":
        train_df, val_df, test_df, collection_str = load_trec(subset=args.subset, seed=args.seed)
        index_path = f"{DATA_DIR}/index_trec_full"
        index = build_full_index_trec(index_path)
        for name, df in [("train",train_df),("val",val_df),("test",test_df)]:
            build_split_parquet_trec(name, df, index,args.method, collection_str)

    elif args.dataset == "qqp":
        load_qqp(args.method)

    else:
        load_mantis(args.method)

# python src/data_prep.py --dataset mantis --method BM25
# python src/data_prep.py --dataset mantis --method random
# python src/data_prep.py --dataset qqp --method BM25
# python src/data_prep.py --dataset trec --method BM25
# python src/data_prep.py --dataset trec --method BM25 --subset 5000
# python src/data_prep.py --dataset trec --method random --subset 5000