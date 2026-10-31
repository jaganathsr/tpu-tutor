"""Numbers used in Chapter 6 (choosing a mesh for training).

Scaling Book ch. 5 model: each layer is In[B,D] W_in[D,F] W_out[F,D] (two matrices).
alpha = C / W_ici per axis. B is the global token batch, N chips.
  DP / FSDP compute-bound:      B/N > alpha / M_X
  TP compute-bound:             Y < M_Y * F / alpha
  FSDP + TP compute-bound:      B/N > alpha^2 / (M_X * M_Y * F)
  optimal FSDP degree:          X_opt = sqrt((B/F) * (M_X/M_Y) * N)
  cross-slice DP over DCN:      per-slice batch > C / W_dcn

    python3 scripts/ch06_training.py
"""
from math import sqrt
from specs import CHIPS
from model import LLAMA3_70B, LLAMA2_13B, LLAMA3_405B

ALPHA = {
    "v5p (book)": (4.59e14, 1.8e11, 6.25e9),
    "v5p (docs)": (4.59e14, 2.0e11, 6.25e9),
    "Ironwood": (CHIPS["7x"].flops_bf16, CHIPS["7x"].W_axis, 12.5e9),
    "8t (provisional bf16)": (CHIPS["8t"].flops_bf16, CHIPS["8t"].W_axis, 50e9),
}


def thresholds(C, W, Wdcn, F):
    a = C / W
    return dict(alpha=a, dp1=a, dp3=a / 3, tp1=F / a, fsdp_tp=a * a / (2 * F), dcn=C / Wdcn)


if __name__ == "__main__":
    F = LLAMA3_70B.F
    print(f"Thresholds for LLaMA-3-70B (F = {F}):")
    for name, (C, W, Wd) in ALPHA.items():
        t = thresholds(C, W, Wd, F)
        print(f"  {name:22} alpha {t['alpha']:7,.0f} | DP/FSDP B/N > {t['dp1']:6,.0f} (1 axis), {t['dp3']:5,.0f} (3 axes) | "
              f"TP Y < {t['tp1']:.1f} per axis | FSDP+TP B/N > {t['fsdp_tp']:5,.0f} | per-slice DCN batch > {t['dcn']:,.0f}")

    # SB6: LLaMA-3-70B, 4M-token batch (1024 x 4096), v5p pod 8960 chips.
    B = 1024 * 4096
    N = 8960
    print(f"\nSB6 on v5p pod: tokens per chip {B/N:.0f}; FSDP+TP threshold (book) {2550**2/(2*F):.0f}; "
          f"X_opt = {sqrt(2*B*N/F):.0f}")
    days = 6 * LLAMA3_70B.params * 15e12 / (N * 4.59e14 * 0.4) / 86400
    print(f"  time at 40% MFU: {days:.0f} days")

    # Same job on an Ironwood pod.
    C, W, Wd = ALPHA["Ironwood"]
    a = C / W
    N7 = 9216
    print(f"\nIronwood pod: tokens per chip {B/N7:.0f} against FSDP+TP threshold {a*a/(2*F):,.0f} -> communication-bound")
    nmax = B * 2 * F / a**2
    print(f"  largest compute-bound slice at 4M tokens: {nmax:,.0f} chips; days at 40% MFU there: "
          f"{6*LLAMA3_70B.params*15e12/(nmax*C*0.4)/86400:.0f}")
    for Bm in (8, 16, 32):
        Bt = Bm * 2**20
        print(f"  batch {Bm}M tokens: compute-bound up to {Bt*2*F/a**2:,.0f} chips")
    print(f"  ratio T_math/T_comms at X_opt on full pod: {sqrt(B*F)*sqrt(2)/(a*sqrt(N7)):.2f}")

    # SB5 Q3: LLaMA-2-13B, 3M tokens, v5p 16x16x16 (4096 chips).
    m = LLAMA2_13B
    B3, N3 = 3e6, 4096
    print(f"\nSB5 Q3: FSDP alone needs B > {850*N3/1e6:.2f}M (have 3M) -> comms-bound; FSDP+TP threshold "
          f"{2550**2/(2*m.F):.0f}/chip; X_opt {sqrt(B3*2*N3/m.F):.0f}; step at 40% MFU "
          f"{6*B3*m.params/(N3*4.59e14*0.4)*1e3:.0f} ms")

    # SB6 Q1 (no book answer): 70B on 4 v5p pods, same 4M batch.
    N4 = 4 * 8960
    print(f"\nSB6 Q1 (derived): 4 pods, per-pod batch {B/4:,.0f} vs DCN {4.59e14/6.25e9:,.0f} -> DCN DP fine; "
          f"per chip {B/N4:.0f} vs FSDP+TP {2550**2/(2*F):.0f} -> just compute-bound; days {days/4:.0f}")

    # SB6 Q2 (no book answer): 405B on 8 v5p pods.
    m = LLAMA3_405B
    N8 = 8 * 8960
    B405 = 16 * 2**20
    flops = 6 * m.params * 15.6e12   # P-Llama3: 15.6T tokens, reports 3.8e25
    print(f"SB6 Q2 (derived): 405B params {m.params/1e9:.1f}B; FLOPs/token {6*m.params:.3g}; total {flops:.3g}")
    print(f"  8 pods at 40% MFU: {flops/(N8*4.59e14*0.4)/86400:.0f} days; batch 16M: per pod {B405/8:,.0f}, per chip "
          f"{B405/N8:.0f} vs FSDP+TP {2550**2/(2*m.F):.0f}; TP limit Y < {m.F/2550:.1f} per axis; X_opt per pod "
          f"{sqrt(2*(B405/8)*8960/m.F):.0f}")

    # GPU (SB12): DP threshold and TP limit.
    for name, Cg, Wg in (("H100 in node", 990e12, 450e9), ("H100 across nodes", 990e12, 400e9), ("B200 across nodes", 2250e12, 400e9)):
        print(f"{name}: DP B/GPU > {Cg/Wg:,.0f}; TP Y < F/{Cg/Wg:,.0f} = {F/(Cg/Wg):.1f} for F = {F}")
