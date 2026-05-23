import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import os

os.makedirs("../res/plots", exist_ok=True)

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
    """
    Charge et concatène une liste de fichiers CSV.
    files : liste de dict {"path": ..., "exclude_modes": [...]}
    """
    dfs = []
    for f in files:
        df_tmp = read_file(f["path"])
        if df_tmp.empty:
            continue
        # Exclure certains modes si demandé
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


def ttest_paired(vals_a, seeds_a, vals_b, seeds_b, alpha=0.05, bonferroni=False):
    common = sorted(set(seeds_a) & set(seeds_b))
    if len(common) < 2:
        return None, None, ""
    a = np.array([vals_a[list(seeds_a).index(s)] for s in common])
    b = np.array([vals_b[list(seeds_b).index(s)] for s in common])
    threshold = alpha / 2 if bonferroni else alpha
    t, p = stats.ttest_rel(a, b)
    sig = "↑" if (p < threshold and a.mean() > b.mean()) else \
          "↓" if (p < threshold and a.mean() < b.mean()) else ""
    return t, p, sig


def best_eps_vals_seeds(df, train_method, mode, epsilons):
    best_mean, best_vals, best_seeds, best_e = -1, None, None, None
    for e in epsilons:
        v, s = get_vals_and_seeds(df, train_method, mode, e)
        if len(v) > 0 and v.mean() > best_mean:
            best_mean, best_vals, best_seeds, best_e = v.mean(), v, s, e
    return best_vals, best_seeds, best_e


# Tableaux
def print_table1(df, train_methods, epsilons, eps_display=0.2):
    print("\n" + "="*75)
    print(f"TABLEAU 1 — R10@1 mean±std (ε={eps_display}, 5 seeds)")
    print("="*75)

    header = f"{'':20}"
    for tm in train_methods:
        header += f" {'NS'+tm:>20}"
    print(header)
    print("-"*75)

    base_vals = {}
    base_seeds = {}
    for tm in train_methods:
        v, s = get_vals_and_seeds(df, tm, "baseline", 0.0)
        base_vals[tm], base_seeds[tm] = v, s

    rows = [("BERT", "baseline", 0.0), ("w. LS", "ls", eps_display), ("w. T-LS", "tls", eps_display)]
    for label, mode, e in rows:
        line = f"{label:20}"
        for tm in train_methods:
            v, s = get_vals_and_seeds(df, tm, mode, e)
            if len(v) == 0:
                line += f" {'N/A':>20}"
                continue
            _, _, sig = ttest_paired(v, s, base_vals[tm], base_seeds[tm]) \
                        if mode != "baseline" else (None, None, "")
            line += f" {v.mean():.3f}±{v.std():.3f}{sig:>20}"
        print(line)


def print_table2(df, train_method, epsilons):
    print("\n" + "="*65)
    print("TABLEAU 2 — Résultats test set, meilleur ε")
    print("="*65)
    print(f"{'':20} {'TREC-DL':>20}")
    print("-"*42)

    base_vals, base_seeds = get_vals_and_seeds(df, train_method, "baseline", 0.0)
    tls_vals, tls_seeds, tls_eps = best_eps_vals_seeds(df, train_method, "tls", epsilons)
    twsls_vals, twsls_seeds, twsls_eps = best_eps_vals_seeds(df, train_method, "twsls", epsilons)

    m, s = base_vals.mean(), base_vals.std()
    print(f"{'BERT':20} {m:.3f}±{s:.3f}")

    _, _, sig = ttest_paired(tls_vals, tls_seeds, base_vals, base_seeds, bonferroni=True)
    m, s = tls_vals.mean(), tls_vals.std()
    print(f"{'w. T-LS':20} {m:.3f}±{s:.3f}{sig}  (ε={tls_eps})")

    _, _, sig_bert = ttest_paired(twsls_vals, twsls_seeds, base_vals, base_seeds, bonferroni=True)
    _, _, sig_tls = ttest_paired(twsls_vals, twsls_seeds, tls_vals,  tls_seeds, bonferroni=True)
    sig_tls = sig_tls.replace("↑","✦").replace("↓","✧")
    m, s = twsls_vals.mean(), twsls_vals.std()
    print(f"{'w. T-WSLS':20} {m:.3f}±{s:.3f}{sig_bert}{sig_tls}  (ε={twsls_eps})")


# Figure 2

def plot_figure2(df, train_method, epsilons, output_path):
    base_vals, _ = get_vals_and_seeds(df, train_method, "baseline", 0.0)
    baseline_mean = base_vals.mean()

    modes = {"tls": ("T-LS", "#4C72B0"), "twsls": ("T-WSLS", "#55A868")}
    x = np.arange(len(epsilons))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, (mode, (label, color)) in enumerate(modes.items()):
        means, cis = [], []
        for e in epsilons:
            v, _ = get_vals_and_seeds(df, train_method, mode, e)
            if len(v) > 0:
                means.append(v.mean())
                cis.append(1.96 * v.std() / np.sqrt(len(v)))
            else:
                means.append(np.nan)
                cis.append(np.nan)

        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, label=label, color=color,alpha=0.85, yerr=cis, capsize=4, error_kw={"linewidth": 1.5})

    ax.axhline(y=baseline_mean, color="black", linestyle="--",linewidth=1.5, label="BERT baseline (ε=0)")

    ax.set_xlabel("Smoothing strength (ε)", fontsize=12)
    ax.set_ylabel("R10@1", fontsize=12)
    ax.set_title(f"T-LS vs T-WSLS sensitivity to ε — NS{train_method}\n" "Error bars = 95% CI over 5 seeds", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in epsilons])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    all_means = [v.mean() for mode in ["tls","twsls"]
                 for e in epsilons
                 for v, _ in [get_vals_and_seeds(df, train_method, mode, e)]
                 if len(v) > 0]
    if all_means:
        ymin = min(all_means + [baseline_mean]) * 0.97
        ymax = max(all_means + [baseline_mean]) * 1.03
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n-> {output_path}")


# python scripts/plot_results.py \
#     --files res/results_trec_BM25.csv res/results_trec_random_train_BM25_test.csv res/results_trec_missing.csv \
#     --exclude_modes ls \
#     --train_methods BM25 random \
#     --eps_display 0.2 \
#     --plot_method BM25
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True, help="Fichiers CSV à charger (ex: res/results_trec_BM25.csv)")
    parser.add_argument("--exclude_modes", nargs="*", default=[],help="Modes à exclure de TOUS les fichiers (ex: ls)")
    parser.add_argument("--train_methods", nargs="+", default=["BM25"], help="Méthodes de train à afficher dans tableau 1")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--eps_display", type=float, default=0.2,help="Epsilon affiché dans tableau 1")
    parser.add_argument("--plot_method", type=str, default="BM25",help="train_method pour la figure 2")
    parser.add_argument("--plot_output", type=str,default="res/plots/figure2_epsilon_sensitivity.png")
    args = parser.parse_args()

    files = [{"path": f, "exclude_modes": args.exclude_modes} for f in args.files]
    df = load_results(files)

    print_table1(df, args.train_methods, args.epsilons, args.eps_display)
    print_table2(df, args.plot_method, args.epsilons)
    plot_figure2(df, args.plot_method, args.epsilons, args.plot_output)