from tqdm.auto import tqdm
import torch
from src.evaluate import evaluate_r10_at_1
import math
from scipy.stats import beta as beta_dist

def compute_smoothed_labels(labels, scores, current_eps, mode):
    """
    Calcule les labels lissés selon le mode.

    Formules :
    - baseline : label original (eps=0)
    - ls : LS classique uniforme -> pos: 1-eps/2, neg: eps/2
    - wsls : WSLS -> pos: 1-eps/2, neg: BM25_norm
    - tls : T-LS (step) -> même que ls mais eps décroît
    - twsls : T-WSLS (step) -> même que wsls mais eps décroît
    """
    if current_eps == 0.0 or mode == "baseline":
        return labels

    K = 2  # classification binaire

    if mode in ("ls", "tls"):
        pos_label = 1.0 - current_eps + current_eps / K 
        neg_label = current_eps / K 
        smoothed = torch.where(labels == 1.0,
                               torch.full_like(labels, pos_label),
                               torch.full_like(labels, neg_label))

    elif mode in ("wsls", "twsls"):
        pos_label = 1.0 - current_eps + current_eps / K
        smoothed = torch.where(
            labels == 1.0,
            torch.full_like(labels, pos_label),
            current_eps * scores 
        )
        smoothed = torch.where(scores == -1, labels, smoothed)

    else:
        smoothed = labels

    return smoothed


def get_epsilon(mode, decay_type, progress, initial_eps, alpha_param=1.0, beta_param=1.0):
    """Calcule epsilon en fonction du mode et du schedule."""

    if mode == "baseline":
        return 0.0

    if mode in ("ls", "wsls"):
        return initial_eps

    if mode in ("tls", "twsls"):
        # Curriculum learning
        if decay_type == "step":
            return initial_eps if progress < 0.5 else 0.0

        elif decay_type == "linear":
            return initial_eps * (1.0 - progress)

        elif decay_type == "exp":
            return initial_eps * math.exp(-5.0 * progress)

        elif decay_type == "cosine":
            return initial_eps * 0.5 * (1.0 + math.cos(math.pi * progress))

        elif decay_type == "beta":
            return initial_eps * (1.0 - beta_dist.cdf(
                progress, alpha_param, beta_param))

    return initial_eps


def train_model(model, dataloader_train, dataloader_test,optimizer, loss_fn, device,total_instances=50000,
                 initial_eps=0.2,mode="twsls", decay_type="step",alpha_param=1.0, beta_param=1.0):
    """
    Modes disponibles :
        baseline : labels originaux (eps=0)
        ls : Label Smoothing classique uniforme (eps constant)
        tls : Two-stage LS (eps décroît selon schedule)
        wsls : Weakly Supervised LS avec scores BM25 (eps constant)
        twsls : Two-stage WSLS (eps décroît selon schedule) <- papier
    """
    seen_instances = 0
    history_loss, history_eps, history_instances = [], [], []

    model.train()
    dataloader_iterator = iter(dataloader_train)

    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()

    progress_bar = tqdm(total=total_instances, desc=f"Entraînement [{mode}]")

    while seen_instances < total_instances:
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader_train)
            batch = next(dataloader_iterator)

        input_ids, attention_mask, labels, scores = [t.to(device) for t in batch]
        current_batch_size = input_ids.size(0)

        progress = seen_instances / total_instances
        current_eps = get_epsilon(mode, decay_type, progress,
                                   initial_eps, alpha_param, beta_param)
        smoothed = compute_smoothed_labels(labels, scores, current_eps, mode)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits.squeeze(-1), smoothed)

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
    final_r10 = evaluate_r10_at_1(model, dataloader_test, device)
    print(f"R10@1 FINAL : {final_r10:.4f}")

    return final_r10, history_instances, history_loss, history_eps