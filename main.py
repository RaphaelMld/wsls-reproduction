import argparse
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
import random
import numpy as np
from src.dataset import MantisDynamicDataset
from src.train import train_model


# python main.py --dataset mantis --method BM25 --mode twsls --eps 0.2 --instances 100000
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, required=True, help="Nom du dataset (ex: mantis, qqp, trec)")
    parser.add_argument("--method", type=str, default="BM25", choices=["BM25", "random"], help="Méthode de negative sampling")
    parser.add_argument("--test_method", type=str, default=None, choices=["BM25", "random"], help="Méthode pour le test (par défaut = --method)")
    parser.add_argument("--seed", type=int, default=0, help="graine aléatoire")
    parser.add_argument("--results_file", type=str, default="results_mantis.csv", help="Fichier CSV pour stocker les résultats")
    parser.add_argument("--mode", type=str, choices=["baseline", "ls", "twsls"], required=True, help="Mode d'entraînement")
    parser.add_argument("--eps", type=float, default=0.2, help="Valeur de l'epsilon (par défaut 0.2)")
    parser.add_argument("--instances", type=int, default=100000, help="Nombre total d'instances")
    parser.add_argument("--save_models", action="store_true", help="sauvegarde les models")
    parser.add_argument("--save_history", action="store_true", help="sauvegarde les dataframe historique de la loss")
    parser.add_argument("--decay", type=str, default="step", choices=["step", "linear", "exp", "cosine", "beta"], help="Type de décroissance pour le T-WSLS")
    parser.add_argument("--alpha", type=float, default=1.0, help="Paramètre alpha pour la loi Beta (si decay=beta)")
    parser.add_argument("--beta", type=float, default=1.0, help="Paramètre beta pour la loi Beta (si decay=beta)")
    
    args = parser.parse_args()

    dataset_name = args.dataset
    test_method = args.test_method if args.test_method else args.method
    print(f"Dataset {dataset_name} - Méthode {args.method} - Test: {test_method} - Mode {args.mode} - Epsilon {args.eps} ")

    train_file = f"data/{dataset_name}_{args.method}_train.parquet"
    test_file = f"data/{dataset_name}_{test_method}_test.parquet"
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"erreur, fichiers introuvables : lancer python src/data_prep.py --dataset {dataset_name} --method {args.method} --test_method {test_method}")
        exit(1)

    # Fixation de la seed pour la reproductibilité
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
    model.to(device)

    # dataloader
    dataset_train = MantisDynamicDataset(train_df, tokenizer, max_len=256)
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=2)

    dataset_test = MantisDynamicDataset(test_df, tokenizer, max_len=256)
    dataloader_test = DataLoader(dataset_test, batch_size=10, shuffle=False, num_workers=2)

    # optim et loss
    optimizer = AdamW(model.parameters(), lr=5e-6, eps=1e-8)
    loss_fn = BCEWithLogitsLoss()

    final_r10, hist_instances, hist_loss, hist_eps = train_model(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        total_instances=args.instances,
        initial_eps=args.eps,
        mode=args.mode,
        decay_type=args.decay,
        alpha_param=args.alpha,
        beta_param=args.beta
    )

    os.makedirs("./res/loss", exist_ok=True)

    if args.save_history:
        df_history = pd.DataFrame({
            "instances_vues": hist_instances,
            "loss": hist_loss,
            "epsilon": hist_eps
        })
        history_file = f"res/loss/history_loss_{dataset_name}_{args.method}_{args.mode}_{args.decay}_a{args.alpha}_b{args.beta}_eps{args.eps}_seed{args.seed}.csv"
        df_history.to_csv(history_file, index=False)
        print(f"Historique de l'entraînement sauvegardé dans {history_file}")
    
    res_file = f"res/{args.results_file}"
    file_exists = os.path.isfile(res_file)
    with open(res_file, "a") as f:
        if not file_exists:
            f.write("dataset,train_method,test_method,mode,decay,alpha,beta,eps,seed,r10_at_1\n")
        f.write(f"{dataset_name},{args.method},{test_method},{args.mode},{args.decay},{args.alpha},{args.beta},{args.eps},{args.seed},{final_r10:.4f}\n")
    
    print(f"\nRésultat ajouté dans {res_file}")

    if args.save_models:
        os.makedirs("./models", exist_ok=True)
        save_path = f"./models/{dataset_name}_{args.method}_{args.mode}_{args.decay}_a{args.alpha}_b{args.beta}_eps{args.eps}_seed{args.seed}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)



if __name__ == "__main__":
    main()