import pandas as pd
import numpy as np
from scipy import stats
import argparse
import os

COLS = ["dataset","train_method","test_method","eval_split","mode","decay","alpha","beta","eps","seed","r10_at_1"]

def read_file(path):
    if not os.path.exists(path):
        print(f"  Manquant : {path}")
        return pd.DataFrame(columns=COLS)
    first = open(path).readline().strip()
    if first.startswith("dataset"):
        return pd.read_csv(path)
    return pd.read_csv(path, header=None, names=COLS)


def load_results(files):
    dfs = []
    for f in files:
        df_tmp = read_file(f["path"])
        if df_tmp.empty:
            continue
        for mode in f.get("exclude_modes", []):
            df_tmp = df_tmp[df_tmp["mode"] != mode]
        print(f"  {f['path']} : {len(df_tmp)} lignes")
        dfs.append(df_tmp)

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(
        subset=["train_method","test_method","eval_split","mode","eps","seed"],
        keep="last"
    ).reset_index(drop=True)

    print(f"\nTotal : {len(df)} runs")
    print(df.groupby(["train_method","mode"])["r10_at_1"].count().to_string())
    return df


def get_vals_and_seeds(df, train_method, mode, eps):
    sub = df[
        (df["train_method"]==train_method) &
        (df["mode"]==mode) &
        (df["eps"]==eps)
    ].sort_values("seed")
    return sub["r10_at_1"].values, sub["seed"].values


def ttest_report(name_a, vals_a, seeds_a, name_b, vals_b, seeds_b, alpha=0.05, bonferroni=False):
    common = sorted(set(seeds_a) & set(seeds_b))
    if len(common) < 2:
        print(f"  X {name_a} vs {name_b} : pas assez de seeds communes ({common})")
        return None
    if len(common) < 5:
        print(f"   Seeds communes seulement : {common}")

    a = np.array([vals_a[list(seeds_a).index(s)] for s in common])
    b = np.array([vals_b[list(seeds_b).index(s)] for s in common])

    threshold = alpha / 2 if bonferroni else alpha
    t, p = stats.ttest_rel(a, b)
    sig = p < threshold
    direction = "↑ gain" if a.mean() > b.mean() else "↓ perte"

    print(f"  {name_a:<25} vs {name_b:<25} : "
          f"t={t:+.3f}  p={p:.4f}  "
          f"{'V sig.' if sig else 'X n.s.':8} {direction if sig else ''}")
    return p


def best_eps_vals_seeds(df, train_method, mode, epsilons):
    best_mean, best_vals, best_seeds, best_e = -1, None, None, None
    for e in epsilons:
        v, s = get_vals_and_seeds(df, train_method, mode, e)
        if len(v) > 0 and v.mean() > best_mean:
            best_mean, best_vals, best_seeds, best_e = v.mean(), v, s, e
    return best_vals, best_seeds, best_e


def run_tests(df, train_methods, epsilons, eps_display=0.2):

    # Tableau 1
    print("\n" + "="*75)
    print(f"TABLEAU 1 — Tests pairés vs BERT baseline (ε={eps_display}, α=0.05)")
    print("="*75)

    for tm in train_methods:
        print(f"\nNS{tm} :")
        base_vals, base_seeds = get_vals_and_seeds(df, tm, "baseline", 0.0)
        for mode, label in [("ls","w. LS"), ("tls","w. T-LS")]:
            v, s = get_vals_and_seeds(df, tm, mode, eps_display)
            ttest_report(f"{label} ({tm})", v, s,
                         f"BERT ({tm})", base_vals, base_seeds)

    # Tableau 2 
    print("\n" + "="*75)
    print("TABLEAU 2 — Tests pairés avec correction Bonferroni (α=0.025)")
    print("="*75)

    base_vals, base_seeds = get_vals_and_seeds(df, "BM25", "baseline", 0.0)
    tls_vals, tls_seeds, tls_eps = best_eps_vals_seeds(df, "BM25", "tls",   epsilons)
    twsls_vals, twsls_seeds, twsls_eps = best_eps_vals_seeds(df, "BM25", "twsls", epsilons)

    print(f"\nMeilleur ε — T-LS : {tls_eps} | T-WSLS : {twsls_eps}\n")

    print("vs BERT baseline (Bonferroni α=0.025) :")
    ttest_report("w. T-LS",   tls_vals,   tls_seeds,"BERT", base_vals, base_seeds, bonferroni=True)
    ttest_report("w. T-WSLS", twsls_vals, twsls_seeds,"BERT", base_vals, base_seeds, bonferroni=True)

    print("\nvs T-LS (Bonferroni α=0.025) :")
    ttest_report("w. T-WSLS", twsls_vals, twsls_seeds, "w. T-LS",   tls_vals,   tls_seeds, bonferroni=True)


    print("\n" + "="*75)
    print("VUE COMPLÈTE — tous les ε, NSBM25")
    print("="*75)
    print(f"\n{'Mode':<10} {'eps':>5} {'mean':>8} {'std':>7} {'n_seeds':>8} {'p vs BERT':>12} {'sig':>6}")
    print("-"*60)

    for mode in ["ls", "tls", "twsls"]:
        for e in epsilons:
            v, s = get_vals_and_seeds(df, "BM25", mode, e)
            if len(v) == 0:
                continue
            common = sorted(set(s) & set(base_seeds))
            if len(common) < 2:
                continue
            a = np.array([v[list(s).index(c)] for c in common])
            b = np.array([base_vals[list(base_seeds).index(c)] for c in common])
            _, p = stats.ttest_rel(a, b)
            sig = "V↑" if (p < 0.05 and a.mean() > b.mean()) else \
                   "V↓" if (p < 0.05 and a.mean() < b.mean()) else "X"
            print(f"{mode:<10} {e:>5.1f} {v.mean():>8.4f} {v.std():>7.4f} "
                  f"{len(v):>8} {p:>12.4f} {sig:>6}")


# python scripts/statistical_tests.py \
#     --files res/results_trec_BM25.csv res/results_trec_random_train_BM25_test.csv res/results_trec_missing.csv \
#     --exclude_modes ls \
#     --train_methods BM25 random \
#     --eps_display 0.2
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True,help="Fichiers CSV à charger")
    parser.add_argument("--exclude_modes", nargs="*", default=[],help="Modes à exclure de TOUS les fichiers")
    parser.add_argument("--train_methods", nargs="+", default=["BM25"], help="Méthodes de train pour tableau 1")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--eps_display", type=float, default=0.2)
    args = parser.parse_args()

    files = [{"path": f, "exclude_modes": args.exclude_modes} for f in args.files]
    df = load_results(files)
    run_tests(df, args.train_methods, args.epsilons, args.eps_display)