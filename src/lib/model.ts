// Transformer accounting, mirrored from scripts/model.py. Keep the two in sync.

export interface ModelCfg {
  name: string;
  L: number; D: number; F: number; N: number; K: number; H: number; V: number;
  E: number; k: number; tied: boolean; src: string;
}

export const PRESETS: ModelCfg[] = [
  { name: 'LLaMA-3-8B', L: 32, D: 4096, F: 14336, N: 32, K: 8, H: 128, V: 128256, E: 1, k: 1, tied: false, src: 'P-Llama3' },
  { name: 'LLaMA-3-70B', L: 80, D: 8192, F: 28672, N: 64, K: 8, H: 128, V: 128256, E: 1, k: 1, tied: false, src: 'SB6' },
  { name: 'LLaMA-3-405B', L: 126, D: 16384, F: 53248, N: 128, K: 8, H: 128, V: 128256, E: 1, k: 1, tied: false, src: 'P-Llama3' },
  { name: 'LLaMA-2-13B', L: 40, D: 5120, F: 13824, N: 40, K: 40, H: 128, V: 32000, E: 1, k: 1, tied: false, src: 'SB5' },
  { name: 'Scaling Book ch. 7 model (18B)', L: 64, D: 4096, F: 16384, N: 32, K: 8, H: 256, V: 32128, E: 1, k: 1, tied: true, src: 'SB7' },
  { name: 'Ch. 7 model as MoE (E=16, k=2)', L: 64, D: 4096, F: 16384, N: 32, K: 8, H: 256, V: 32128, E: 16, k: 2, tied: true, src: 'SB7' },
];

export const attnParams = (m: ModelCfg) => 2 * m.D * m.H * (m.N + m.K);
export const mlpParams = (m: ModelCfg) => 3 * m.D * m.F * m.E;
export const mlpActive = (m: ModelCfg) => 3 * m.D * m.F * m.k;
export const vocabParams = (m: ModelCfg) => m.D * m.V * (m.tied ? 1 : 2);
export const params = (m: ModelCfg) => m.L * (attnParams(m) + mlpParams(m) + 2 * m.D) + vocabParams(m);
export const activeParams = (m: ModelCfg) => m.L * (attnParams(m) + mlpActive(m) + 2 * m.D) + vocabParams(m);
export const kvBytesPerToken = (m: ModelCfg, bytesPer: number) => 2 * m.L * m.K * m.H * bytesPer;
/** Forward attention FLOPs (QK and AV) for one query token against S keys. */
export const attnFlopsPerToken = (m: ModelCfg, S: number) => 4 * S * m.N * m.H * m.L;
