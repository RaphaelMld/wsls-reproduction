from tqdm.auto import tqdm
import torch
from src.evaluate import evaluate_r10_at_1
import math
from scipy.stats import beta

def train_model(model, dataloader_train, dataloader_test, optimizer, loss_fn, device, 
                 total_instances=50000, initial_eps=0.2, mode="twsls", decay_type="step", alpha_param=1.0, beta_param=1.0):
    """
    Entraîne le modèle avec la méthode Two-stage Weakly Supervised Label Smoothing (T-WSLS).
    
    Args:
        model: Le modèle PyTorch.
        dataloader_train: Le DataLoader d'entraînement.
        dataloader_test: Le DataLoader de test pour l'évaluation.
        optimizer: L'optimiseur (ex: AdamW).
        loss_fn: La fonction de perte (ex: BCEWithLogitsLoss).
        device: 'cuda' ou 'cpu'.
        total_instances (int): Nombre total d'instances à voir.
        initial_eps (float): Valeur de l'epsilon pour la première moitié de l'entraînement
        mode: le mode de lissage (par défaut twsls) : parmi ["twsls", "ls", "baseline"]
        decay: Type de décroissance pour le T-WSLS (par défaut step) : parmi ["step", "linear", "exp", "cosine", "beta"]

    """
    seen_instances = 0
    
    history_loss = []
    history_eps = []
    history_instances = []
    
    model.train()
    dataloader_iterator = iter(dataloader_train)

    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
        
    progress_bar = tqdm(total=total_instances, desc=f"Entrainement {mode}")

    while seen_instances < total_instances:
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader_train)
            batch = next(dataloader_iterator)
            
        input_ids, attention_mask, labels, scores = [t.to(device) for t in batch]
        current_batch_size = input_ids.size(0)
        
        if mode == "twsls":
            progress = seen_instances / total_instances
            
            if decay_type == "step":
                # L'approche du papier (Escalier)
                current_eps = initial_eps if progress < 0.5 else 0.0
                
            elif decay_type == "linear":
                # Baisse progressivement de initial_eps jusqu'à 0
                current_eps = initial_eps * (1.0 - progress)
                
            elif decay_type == "exp":
                # Baisse de façon quasi-logarithmique (exponentielle négative)
                # Le multiplicateur -5.0 permet d'atteindre ~0 à la fin
                current_eps = initial_eps * math.exp(-5.0 * progress)
                
            elif decay_type == "cosine":
                # Baisse en cloche 
                current_eps = initial_eps * 0.5 * (1.0 + math.cos(math.pi * progress))

            elif decay_type == "beta": 
                current_eps = initial_eps * (1.0 - beta.cdf(progress, alpha_param, beta_param))
                
        elif mode == "ls":
            # Reste constant
            current_eps = initial_eps
        else: 
            # baseline, pas de lissage
            current_eps = 0.0
        
        smoothed_labels = torch.where(
            scores != -1,
            (1 - current_eps) * labels + current_eps * scores,
            labels
        )
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits.squeeze(-1), smoothed_labels) 

        history_loss.append(loss.item())
        history_eps.append(current_eps)
        history_instances.append(seen_instances)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        seen_instances += current_batch_size
        progress_bar.update(current_batch_size)
        progress_bar.set_postfix(loss=loss.item(), eps=f"{current_eps:.3f}")

    
    progress_bar.close()
    print("\nEntraînement terminé")
    
    # Évaluation finale 
    final_r10 = evaluate_r10_at_1(model, dataloader_test, device)
    print(f"R10@1 FINAL : {final_r10:.4f}")

    return final_r10, history_instances, history_loss, history_eps