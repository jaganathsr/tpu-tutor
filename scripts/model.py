"""Transformer parameter, FLOP and KV-cache accounting (Scaling Book ch. 4 conventions).

Shapes: L layers, D d_model, F d_ff (per expert for MoE), N query heads, K KV heads,
H head dim, V vocab, E experts, k active experts. Gated MLP (3 matrices).
Mirrored in src/lib/model.ts for the in-browser calculators; keep them in sync.
"""
from dataclasses import dataclass


@dataclass
class Model:
    name: str
    L: int
    D: int
    F: int
    N: int
    K: int
    H: int
    V: int
    E: int = 1
    k: int = 1
    tied: bool = False
    src: str = ""

    @property
    def attn_params(self):          # per layer: W_Q, W_O (N heads) and W_K, W_V (K heads)
        return 2 * self.D * self.H * (self.N + self.K)

    @property
    def mlp_params(self):           # per layer, all experts
        return 3 * self.D * self.F * self.E

    @property
    def mlp_active(self):
        return 3 * self.D * self.F * self.k

    @property
    def vocab_params(self):
        return self.D * self.V * (1 if self.tied else 2)

    @property
    def params(self):
        return self.L * (self.attn_params + self.mlp_params + 2 * self.D) + self.vocab_params

    @property
    def active_params(self):
        return self.L * (self.attn_params + self.mlp_active + 2 * self.D) + self.vocab_params

    def kv_bytes_per_token(self, bytes_per=2):
        return 2 * self.L * self.K * self.H * bytes_per

    def attn_flops_per_token(self, S):
        """Forward QK^T and AV for one query token against S keys (no causal saving)."""
        return 4 * S * self.N * self.H * self.L

    def fwd_flops_per_token(self, S=0):
        return 2 * self.active_params + self.attn_flops_per_token(S)


LLAMA3_70B = Model("LLaMA-3-70B", L=80, D=8192, F=28672, N=64, K=8, H=128, V=128256, src="SB6")
LLAMA2_13B = Model("LLaMA-2-13B", L=40, D=5120, F=13824, N=40, K=40, H=128, V=32000, src="SB5")
SB7_DENSE = Model("Scaling Book ch.7 model", L=64, D=4096, F=16384, N=32, K=8, H=256, V=32128, tied=True, src="SB7")
SB7_MOE = Model("ch.7 model as MoE (E=16, k=2)", L=64, D=4096, F=16384, N=32, K=8, H=256, V=32128, E=16, k=2, tied=True, src="SB7")
LLAMA3_8B = Model("LLaMA-3-8B", L=32, D=4096, F=14336, N=32, K=8, H=128, V=128256, src="P-Llama3 Table 3")
LLAMA3_405B = Model("LLaMA-3-405B", L=126, D=16384, F=53248, N=128, K=8, H=128, V=128256, src="P-Llama3 Table 3")
MODELS = [LLAMA3_8B, LLAMA3_70B, LLAMA3_405B, LLAMA2_13B, SB7_DENSE, SB7_MOE]
