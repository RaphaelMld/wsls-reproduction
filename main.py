import argparse
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

from src.dataset import MantisDynamicDataset
from src.train import train_model


# python main.py --dataset mantis --method BM25 --mode twsls --eps 0.2 --instances 100000
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, required=True, help="Nom du dataset (ex: mantis, qqp, trec)")
    parser.add_argument("--method", type=str, default="BM25", choices=["BM25", "random"], help="Méthode de negative sampling")
    
    parser.add_argument("--mode", type=str, choices=["baseline", "ls", "twsls"], required=True, help="Mode d'entraînement")
    parser.add_argument("--eps", type=float, default=0.2, help="Valeur de l'epsilon (par défaut 0.2)")
    parser.add_argument("--instances", type=int, default=100000, help="Nombre total d'instances")
    
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Dataset {dataset_name} - Méthode {args.method} - Mode {args.mode} - Epsilon {args.eps} ")

    train_file = f"data/{dataset_name}_{args.method}_train.parquet"
    test_file = f"data/{dataset_name}_{args.method}_test.parquet"
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"erreur, fichiers introuvables : lancer python src/data_prep.py --dataset {dataset_name} --method {args.method} ")
        exit(1)

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

    final_r10 = train_model(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        total_instances=args.instances,
        initial_eps=args.eps,
        mode=args.mode
    )

    # Sauvegarde du modèle
    os.makedirs("./models", exist_ok=True)
    save_path = f"./models/{dataset_name}_{args.method}_bert_{args.mode}_eps{args.eps}"
    
    print(f"Sauvegarde du modèle dans {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()