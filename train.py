import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import os

from model import GPTLanguageModel
from data import get_batch
from data import vocab_size

max_iters = 100000 
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dropout = 0.2
eval_iters = 250
eval_interval = 1000
grad_clip = 1.0
model_save_path = 'model.pth'
optimizer_save_path = 'optimizer.pth'
scheduler_save_path = 'scheduler.pth'

print(vocab_size)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with autocast():
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    print(f"train perplexity: {torch.exp(out['train']):.4f}, val perplexity: {torch.exp(out['val']):.4f}")
    return out

model = GPTLanguageModel().to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

# load checkpoints
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    print("Model loaded from checkpoint.")
if os.path.exists(optimizer_save_path):
    optimizer.load_state_dict(torch.load(optimizer_save_path))
    print("Optimizer loaded from checkpoint.")
if os.path.exists(scheduler_save_path):
    scheduler.load_state_dict(torch.load(scheduler_save_path))
    print("Scheduler loaded from checkpoint.")

# training loop
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        torch.save(model.state_dict(), model_save_path)
        torch.save(optimizer.state_dict(), optimizer_save_path)
        torch.save(scheduler.state_dict(), scheduler_save_path)
        print(f"Checkpoint saved at step {iter}.")

    xb, yb = get_batch('train')
    with autocast():
        logits, loss = model(xb, yb)
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

