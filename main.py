# %%
import math
import os
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
    max_seq_len: int = 4  # a, op, b, = (we predict c after =)
    op_pos: int = 1  # operator position: 0=prefix (op a b =), 1=infix (a op b =), 2=postfix (a b op =)
    operators: tuple[str, ...] = ("+", "*")  # list of operators to include

    @property
    def vocab_size(self) -> int:
        # p for numbers, 5 for '=', '+', '*', '-', '/'
        return self.p + 5

    @property
    def run_name(self) -> str:
        ops = ",".join(self.operators)
        pos_name = ["pre", "in", "post"][self.op_pos]
        return f"mod{self.p}-{ops}-{pos_name}-L{self.n_layers}-wd{self.weight_decay}-gc{self.grad_clip}"

    train_frac: float = 0.3  # 30% train, 70% test
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    grad_clip: float = 1.0
    epochs: int = 500_000
    log_every: int = 100
    save_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    dtype: t.dtype = t.float32
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
PLUS_TOKEN = config.p + 1  # 114
MULT_TOKEN = config.p + 2  # 115
MINUS_TOKEN = config.p + 3  # 116
DIV_TOKEN = config.p + 4  # 117

OP_TOKENS = {
    "+": PLUS_TOKEN,
    "*": MULT_TOKEN,
    "-": MINUS_TOKEN,
    "/": DIV_TOKEN,
}


# %%
# Dataset: (a + b) mod p and (a * b) mod p
class ModArithmeticDataset(Dataset):
    def __init__(self, p: int, data: list[tuple[int, int, str]]):
        self.p = p
        self.data = data  # list of (a, b, op) tuples where op is '+' or '*'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a, b, op = self.data[idx]

        if op == "+":
            c = (a + b) % self.p
        elif op == "*":
            c = (a * b) % self.p
        elif op == "-":
            c = (a - b) % self.p
        elif op == "/":
            # b != 0 is guaranteed by create_datasets
            # modular division: a / b = a * b^(-1) mod p
            assert b != 0, (
                "Division by zero, the dataset should not contain such examples"
            )
            c = (a * pow(b, self.p - 2, self.p)) % self.p
        else:
            raise ValueError(f"Unknown operator: {op}")

        op_token = OP_TOKENS[op]

        # Build sequence based on operator position
        if config.op_pos == 0:  # prefix: op a b =
            seq = [op_token, a, b, EQUALS_TOKEN]
        elif config.op_pos == 1:  # infix: a op b =
            seq = [a, op_token, b, EQUALS_TOKEN]
        else:  # postfix: a b op =
            seq = [a, b, op_token, EQUALS_TOKEN]

        seq_tensor = t.tensor(seq, dtype=t.long)
        out_tensor = t.tensor(c, dtype=t.long)
        return seq_tensor, out_tensor


def create_datasets(p: int, train_frac: float):
    """Create train/test datasets with train_frac of all pairs for configured operations."""
    assert len(config.operators) > 0, "Must specify at least one operator"

    # Create all pairs for selected operations
    all_examples = []
    for a in range(p):
        for b in range(p):
            for op in config.operators:
                # Skip b=0 for division (undefined)
                if op == "/" and b == 0:
                    continue
                all_examples.append((a, b, op))

    n_train = int(len(all_examples) * train_frac)

    # Shuffle deterministically
    generator = t.Generator().manual_seed(42)
    perm = t.randperm(len(all_examples), generator=generator).tolist()

    train_data = [all_examples[i] for i in perm[:n_train]]
    test_data = [all_examples[i] for i in perm[n_train:]]

    return ModArithmeticDataset(p, train_data), ModArithmeticDataset(p, test_data)


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

    @classmethod
    def load(cls, checkpoint_path: str, device: str | None = None):
        """Load a model from a checkpoint file. Returns (model, config)."""
        checkpoint = t.load(checkpoint_path, map_location=device or config.device)

        # Print checkpoint info
        print(f"Loading checkpoint: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(
            f"  Train loss: {checkpoint['train_loss']:.4f}, Train acc: {checkpoint['train_acc']:.4f}"
        )
        print(
            f"  Test loss: {checkpoint['test_loss']:.4f}, Test acc: {checkpoint['test_acc']:.4f}"
        )

        # Reconstruct config from checkpoint
        loaded_config = Config(**checkpoint["config"])
        print(f"  Run name: {loaded_config.run_name}")

        # Create model and load weights
        model = cls(loaded_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device or loaded_config.device)

        return model, loaded_config


# %%
# Training - full batch, constant LR (matching Nanda's setup)
def train():
    set_seed(config.seed)

    wandb.init(
        project="ben-interp",
        name=config.run_name,
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
        betas=config.betas,
    )

    # Create checkpoint directory
    os.makedirs(os.path.join(config.checkpoint_dir, config.run_name), exist_ok=True)

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

        # Save checkpoint
        if epoch % config.save_every == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": loss.item(),
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "config": config.__dict__,
            }
            checkpoint_path = os.path.join(
                config.checkpoint_dir, config.run_name, f"epoch_{epoch:06d}.pt"
            )
            t.save(checkpoint, checkpoint_path)

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
