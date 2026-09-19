import math

import torch
from torch import nn
import torch.nn.functional as F


class MOE(nn.Module):
    """Top-K MoE with all expert weights stacked into single tensors and the
    per-expert loop replaced by sort-by-expert + grouped GEMM.

    w_gate / w_up are [E, H, Hi]; w_down is [E, Hi, H].
    """

    def __init__(self, E, K, H, Hi):
        super().__init__()

        self.E = E
        self.K = K
        self.Hi = Hi

        self.router = nn.Linear(H, E, bias=False)
        self.w_gate = nn.Parameter(torch.empty(E, H, Hi))
        self.w_up = nn.Parameter(torch.empty(E, H, Hi))
        self.w_down = nn.Parameter(torch.empty(E, Hi, H))
        self.reset_parameters()

    def reset_parameters(self):
        # per-expert slices follow nn.Linear's default init (weights are
        # stored [in, out], i.e. transposed relative to nn.Linear)
        for e in range(self.E):
            for w in (self.w_gate[e], self.w_up[e], self.w_down[e]):
                nn.init.kaiming_uniform_(w.mT, a=math.sqrt(5))

    def forward(self, x):
        logits = self.router(x)
        weights, idx = torch.topk(logits, k=self.K, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # sort the S*K (token, expert) pairs by expert so each expert's rows
        # are contiguous — the local analogue of dispatch
        flat_experts = idx.flatten()
        order = torch.argsort(flat_experts, stable=True)
        toks = order // self.K              # flat pair i belongs to token i // K
        w_sorted = weights.flatten()[order]

        counts = torch.bincount(flat_experts, minlength=self.E)
        offs = counts.cumsum(0).to(torch.int32)  # group end offsets, [E]

        xg = x[toks]                                            # [S*K, H]
        gate = torch._grouped_mm(xg, self.w_gate, offs=offs)    # [S*K, Hi]
        up = torch._grouped_mm(xg, self.w_up, offs=offs)        # [S*K, Hi]
        a = F.silu(gate) * up
        y = torch._grouped_mm(a, self.w_down, offs=offs)        # [S*K, H]

        # weighted scatter back to token order — the local analogue of combine
        out = torch.zeros_like(x)
        out.index_add_(0, toks, y * w_sorted[:, None])
        return out

    def forward_loop(self, x):
        """Naive per-expert loop over the same stacked weights (for parity checks)."""
        logits = self.router(x)
        weights, idx = torch.topk(logits, k=self.K, dim=-1)
        weights = F.softmax(weights, dim=-1)
        flat_weight = weights.flatten()
        flat_experts = idx.flatten()
        flat_tokens = torch.arange(x.size(0), device=x.device).repeat_interleave(self.K)

        out = torch.zeros_like(x)
        for e in range(self.E):
            sel = flat_experts == e
            if not sel.any():
                continue
            toks = flat_tokens[sel]
            gate = x[toks] @ self.w_gate[e]
            up = x[toks] @ self.w_up[e]
            y = (F.silu(gate) * up) @ self.w_down[e]
            out.index_add_(0, toks, y * flat_weight[sel, None])
        return out


if __name__ == '__main__':
    torch.manual_seed(0)
    moe = MOE(16, 4, 128, 256).to('cuda:0').to(torch.bfloat16)
    x = torch.randn(1024, 128, device='cuda:0', dtype=torch.bfloat16)
    y = moe(x)
    y_ref = moe.forward_loop(x)
    print(y.shape)
    print('max abs diff vs loop:', (y - y_ref).abs().max().item())
