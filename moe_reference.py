"""Single dropless top-K MoE layer — plain PyTorch reference.

Matches MoonEP notation: S tokens, K top-k, E experts, H hidden, Hp = H' FFN
intermediate. Single-GPU reference (no expert parallelism / dispatch-combine
comms) — the per-expert gather/scatter below is the local analogue of what
dispatch + combine do across ranks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Gated FFN (SwiGLU): H -> Hp -> H."""

    def __init__(self, H: int, Hp: int):
        super().__init__()
        self.gate = nn.Linear(H, Hp, bias=False)
        self.up = nn.Linear(H, Hp, bias=False)
        self.down = nn.Linear(Hp, H, bias=False)

    def forward(self, x):  # [n, H] -> [n, H]
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoELayer(nn.Module):
    def __init__(self, H: int, Hp: int, E: int, K: int):
        super().__init__()
        self.E, self.K = E, K
        self.router = nn.Linear(H, E, bias=False)
        self.experts = nn.ModuleList([Expert(H, Hp) for _ in range(E)])

    def forward(self, x):  # x: [S, H] -> [S, H]
        S = x.shape[0]

        # --- router: pick top-K experts per token, softmax over the K chosen
        logits = self.router(x)                              # [S, E]
        weights, idx = torch.topk(logits, self.K, dim=-1)    # [S, K]
        weights = F.softmax(weights, dim=-1)                 # [S, K]

        # flatten the S*K (token, expert) routing pairs
        flat_expert = idx.reshape(-1)                        # [S*K]
        flat_token = torch.arange(S, device=x.device).repeat_interleave(self.K)
        flat_weight = weights.reshape(-1)                    # [S*K]

        out = torch.zeros_like(x)
        for e in range(self.E):
            sel = flat_expert == e                           # pairs routed to expert e
            if not sel.any():
                continue
            toks = flat_token[sel]                           # gather  == dispatch
            y = self.experts[e](x[toks])                     # expert FFN
            out.index_add_(0, toks, y * flat_weight[sel, None])  # weighted scatter == combine
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    S, H, Hp, E, K = 8, 16, 32, 4, 2
    layer = MoELayer(H, Hp, E, K)
    x = torch.randn(S, H)
    y = layer(x)
    print("in:", x.shape, "out:", y.shape)   # -> [8, 16] both
