from tqdm.auto import tqdm
import torch
from src.evaluate import evaluate_r10_at_1


def train_model(model, dataloader_train, dataloader_test, optimizer, loss_fn, device, 
                 total_instances=50000, initial_eps=0.2, mode="twsls"):
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
        initial_eps (float): Valeur de l'epsilon pour la première moitié de l'entraînement.
        eval_every (int): Fréquence d'évaluation (en nombre d'instances).
        
    Returns:
        tuple: (history_steps, history_r10) pour tracer les courbes.
    """
    seen_instances = 0
    
    history_steps = []
    history_r10 = []
    
    model.train()
    dataloader_iterator = iter(dataloader_train)

    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
        
    progress_bar = tqdm(total=total_instances, desc="Entrainement T-WSLS")

    while seen_instances < total_instances:
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader_train)
            batch = next(dataloader_iterator)
            
        input_ids, attention_mask, labels, scores = [t.to(device) for t in batch]
        current_batch_size = input_ids.size(0)
        
        if mode == "twsls":
            # Baisse à 0.0 à la moitié
            current_eps = initial_eps if seen_instances < (total_instances // 2) else 0.0
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

    return final_r10