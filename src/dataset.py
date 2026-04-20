import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


class MantisDynamicDataset(Dataset):
    def __init__(self, training_data_df, tokenizer, max_len=128):
        self.data = training_data_df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        inputs = self.tokenizer(
            row["query"],
            row["text"],
            return_tensors="pt", 
            truncation=True,
            max_length=self.max_len, 
            padding='max_length',
            return_attention_mask=True
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # On renvoie les labels bruts les scores 
        #pour faire le smoothing dynamiquement lors de l'entrainement -> curriculum learning
        label = torch.tensor(row["label"], dtype=torch.float)
        score = torch.tensor(row["score"], dtype=torch.float)
        
        return input_ids, attention_mask, label, score


