import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)

vocab_size = 3193
d_model = 768

torch.manual_seed(42)
embedding = nn.Embedding(vocab_size, d_model)
lm_head = nn.Linear(d_model, vocab_size, bias=False)
lm_head.weight = embedding.weight

norm = RMSNorm(d_model)

input_ids = torch.randint(0, vocab_size, (4, 511))
x = embedding(input_ids)

# Let's say h goes through some arbitrary layers, getting unnormalized
h = x * 10 

h = norm(h)
logits = lm_head(h)

labels = torch.randint(0, vocab_size, (4, 511))
loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), labels.view(-1))
print(f"Loss unscaled: {loss.item()}")

# Now if we scale the embedding weight?
embedding.weight.data.normal_(mean=0.0, std=0.02)
h = norm(embedding(input_ids))
logits = lm_head(h)
loss2 = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), labels.view(-1))
print(f"Loss scaled 0.02: {loss2.item()}")

