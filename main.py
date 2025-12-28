# %%
import math
from dataclasses import dataclass

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import wandb


# %%
# Configuration - matches Neel Nanda's "Progress measures for grokking" paper
@dataclass
class Config:
    p: int = 113  # prime modulus
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 1  # single layer transformer
    d_mlp: int = 512
    dropout: float = 0.0
    max_seq_len: int = 3  # a, b, = (we predict c after =)
    vocab_size: int = 114  # 0-112 for numbers, 113 for '='
    train_frac: float = 0.3  # 30% train, 70% test
    lr: float = 1e-3
    weight_decay: float = 1.0
    grad_clip: float = 1.0
    epochs: int = 50000
    log_every: int = 100
    seed: int = 42
    dtype: str = "float32"
    device: str = (
        "mps"
        if t.backends.mps.is_available()
        else "cuda"
        if t.cuda.is_available()
        else "cpu"
    )


def set_seed(seed: int):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


config = Config()
EQUALS_TOKEN = config.p  # 113


# %%
# Dataset: (a + b) mod p
class ModAdditionDataset(Dataset):
    def __init__(self, p: int, indices: list[tuple[int, int]]):
        self.p = p
        self.data = indices  # list of (a, b) tuples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a, b = self.data[idx]
        c = (a + b) % self.p
        # Input: [a, b, =], Target: c
        x = t.tensor([a, b, EQUALS_TOKEN], dtype=t.long)
        y = t.tensor(c, dtype=t.long)
        return x, y


def create_datasets(p: int, train_frac: float):
    """Create train/test datasets with train_frac of all pairs for training."""
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    n_train = int(len(all_pairs) * train_frac)

    # Shuffle deterministically
    generator = t.Generator().manual_seed(42)
    perm = t.randperm(len(all_pairs), generator=generator).tolist()

    train_pairs = [all_pairs[i] for i in perm[:n_train]]
    test_pairs = [all_pairs[i] for i in perm[n_train:]]

    return ModAdditionDataset(p, train_pairs), ModAdditionDataset(p, test_pairs)


# %%
# Transformer model - No LayerNorm, no biases, ReLU activation (matching Nanda's setup)
class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        self.W_Q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_K = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_V = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_O = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        q = self.W_Q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Causal attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = t.triu(t.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(out)


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_mlp, bias=True)
        self.fc2 = nn.Linear(config.d_mlp, config.d_model, bias=True)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.unembed = nn.Linear(config.d_model, config.p, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Nanda uses 1/sqrt(d_model) ≈ 0.088 for d_model=128
        std = 1.0 / math.sqrt(self.config.d_model)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(t.arange(T, device=x.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        logits = self.unembed(x[:, -1])  # only predict from last position
        return logits


# %%
# Training - full batch, constant LR (matching Nanda's setup)
def train():
    set_seed(config.seed)

    wandb.init(
        project="ben-interp",
        name=f"mod{config.p}-add-L{config.n_layers}-wd{config.weight_decay}-gc{config.grad_clip}",
        config=config.__dict__,
    )

    train_dataset, test_dataset = create_datasets(config.p, config.train_frac)

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    print(f"Device: {config.device}")

    # Build full batch tensors directly (no DataLoader needed for full batch)
    train_x = t.stack([train_dataset[i][0] for i in range(len(train_dataset))]).to(
        config.device
    )
    train_y = t.stack([train_dataset[i][1] for i in range(len(train_dataset))]).to(
        config.device
    )
    test_x = t.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(
        config.device
    )
    test_y = t.stack([test_dataset[i][1] for i in range(len(test_dataset))]).to(
        config.device
    )

    model = Transformer(config).to(dtype=config.dtype, device=config.device)
    optimizer = t.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
    )

    converged_at = None
    for epoch in range(config.epochs):
        model.train()

        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)

        optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm and clip
        grad_norm = t.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.grad_clip
        )

        optimizer.step()

        train_acc = (logits.argmax(-1) == train_y).float().mean().item()

        # Evaluate on test set
        model.eval()
        with t.no_grad():
            test_logits = model(test_x)
            test_loss = F.cross_entropy(test_logits, test_y).item()
            test_acc = (test_logits.argmax(-1) == test_y).float().mean().item()

        if epoch % config.log_every == 0 or (test_acc == 1.0 and converged_at is None):
            print(
                f"Epoch {epoch}: train_loss={loss.item():.4f}, train_acc={train_acc:.4f}, "
                f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
            )

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": loss.item(),
                "train/accuracy": train_acc,
                "test/loss": test_loss,
                "test/accuracy": test_acc,
                "train/grad_norm": grad_norm,
            }
        )

        if test_acc == 1.0 and converged_at is None:
            print(
                f"Perfect test accuracy reached at epoch {epoch}! Running for 5k more steps..."
            )
            converged_at = epoch

        if converged_at is not None and epoch >= converged_at + 5000:
            print("Finished 5k steps after convergence.")
            break

    wandb.finish()
    return model


# %%
if __name__ == "__main__":
    model = train()
