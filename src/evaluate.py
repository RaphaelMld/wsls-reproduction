import torch
from tqdm.auto import tqdm

def evaluate_r10_at_1(model, dataloader_test, device):
    """
    Évalue le modèle en calculant le R_10@1
    """
    model.eval()
    correct_predictions = 0
    total_queries = 0
    
    with torch.no_grad(): 
        for batch in tqdm(dataloader_test, desc="Évaluation R10@1"):
            input_ids, attention_mask, labels, _ = [t.to(device) for t in batch]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1) 
            
            best_doc_index = torch.argmax(logits).item()
            
            if labels[best_doc_index].item() == 1.0:
                correct_predictions += 1
                
            total_queries += 1
            
    r10_at_1 = correct_predictions / total_queries
    return r10_at_1