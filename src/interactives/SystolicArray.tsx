// Weight-stationary systolic array: W[K, N] is preloaded, one value per cell;
// activation rows X[m, :] enter from the left, skewed one cycle per row of the
// array; partial sums flow down and leave at the bottom.
// Activation x[m, k] reaches cell (k, n) at cycle m + k + n, so cell (k, n) is
// busy at cycle t iff 0 <= t - k - n < M.
import { useEffect, useRef, useState } from 'preact/hooks';

const reduced = () => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;

export function SystolicArray() {
  const [S, setS] = useState(8);        // array size drawn (stands in for 128 or 256)
  const [K, setK] = useState(8);        // contraction depth used
  const [N, setN] = useState(8);        // output columns used
  const [M, setM] = useState(4);        // activation rows streamed per weight tile
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<number>();

  const k = Math.min(K, S), n = Math.min(N, S);
  const total = M + k + n - 2;          // cycles until the last output leaves
  const done = t >= total;

  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(() => setT((x) => (x >= total ? x : x + 1)), reduced() ? 900 : 380);
    return () => clearInterval(timer.current);
  }, [playing, total]);
  useEffect(() => { if (done) setPlaying(false); }, [done]);
  useEffect(() => { setT(0); setPlaying(false); }, [S, K, N, M]);

  const busy = (r: number, c: number, at: number) => r < k && c < n && at - r - c >= 0 && at - r - c < M;
  let busyNow = 0;
  for (let r = 0; r < S; r++) for (let c = 0; c < S; c++) if (busy(r, c, t)) busyNow++;
  let macs = 0;
  for (let x = 0; x <= Math.min(t, total); x++) for (let r = 0; r < k; r++) for (let c = 0; c < n; c++) if (busy(r, c, x)) macs++;
  const elapsed = Math.min(t, total) + 1;
  const util = macs / (S * S * elapsed);
  const outputsDone = (() => {
    let o = 0;
    for (let m = 0; m < M; m++) for (let c = 0; c < n; c++) if (t >= m + (k - 1) + c) o++;
    return o;
  })();
  const finalUtil = (M * k * n) / (S * S * total);

  const cell = 30, gap = 3, pad = 104;
  const W = pad + S * (cell + gap) + 150, H = 40 + S * (cell + gap) + 60;

  return (
    <div class="ix">
      <div class="ix-controls">
        <label class="field"><span class="lbl">Array size <output>{S}×{S}</output></span>
          <input type="range" min={4} max={10} value={S} onInput={(e) => setS(+(e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">K, contraction depth used <output>{k}</output></span>
          <input type="range" min={1} max={S} value={k} onInput={(e) => setK(+(e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">N, output columns used <output>{n}</output></span>
          <input type="range" min={1} max={S} value={n} onInput={(e) => setN(+(e.target as HTMLInputElement).value)} /></label>
        <label class="field"><span class="lbl">M, rows streamed per weight tile <output>{M}</output></span>
          <input type="range" min={1} max={32} value={M} onInput={(e) => setM(+(e.target as HTMLInputElement).value)} /></label>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Systolic array at cycle ${t}: ${busyNow} of ${S * S} cells busy.`}>
        <text x={pad} y={18}>W[K, N] stays in the cells; activation rows enter from the left.</text>
        {Array.from({ length: S }, (_, r) => (
          <g>
            <text x={pad - 10} y={40 + r * (cell + gap) + cell / 2 + 4} text-anchor="end">
              {r < k ? `k=${r}` : 'idle'}
            </text>
            {/* activation entering this row: x[m, r] with m = t - r - 0 at column 0 */}
            {r < k && t - r >= 0 && t - r < M ? (
              <text x={pad - 44} y={40 + r * (cell + gap) + cell / 2 + 4} text-anchor="end" style="fill: var(--s2)">
                x[{t - r},{r}]
              </text>
            ) : null}
            {Array.from({ length: S }, (_, c) => {
              const used = r < k && c < n;
              const b = busy(r, c, t);
              return (
                <rect
                  x={pad + c * (cell + gap)}
                  y={40 + r * (cell + gap)}
                  width={cell}
                  height={cell}
                  rx="2"
                  fill={b ? 'var(--s1)' : used ? 'var(--accent-tint)' : 'var(--paper)'}
                  stroke={used ? 'var(--rule-strong)' : 'var(--rule)'}
                  stroke-dasharray={used ? undefined : '3 3'}
                />
              );
            })}
          </g>
        ))}
        {Array.from({ length: S }, (_, c) => (
          <text x={pad + c * (cell + gap) + cell / 2} y={40 + S * (cell + gap) + 16} text-anchor="middle">
            {c < n ? `n=${c}` : ''}
          </text>
        ))}
        <text x={pad} y={H - 12}>Partial sums flow down; output y[m, n] leaves column n at cycle m + (K - 1) + n.</text>
        <g transform={`translate(${pad + S * (cell + gap) + 18}, 48)`}>
          <text y={0} style="fill: var(--ink)">Cycle {t} of {total}</text>
          <text y={22}>Busy cells now: {busyNow} / {S * S}</text>
          <text y={44}>MACs so far: {macs}</text>
          <text y={66}>Outputs done: {outputsDone} / {M * n}</text>
          <text y={88}>Utilization so far: {(util * 100).toFixed(0)}%</text>
          <text y={110} style="fill: var(--ink)">For the whole tile: {(finalUtil * 100).toFixed(0)}%</text>
        </g>
      </svg>

      <div class="btn-row">
        <button class="btn primary" onClick={() => (done ? (setT(0), setPlaying(true)) : setPlaying(!playing))}>
          {playing ? 'Pause' : done ? 'Replay' : 'Play'}
        </button>
        <button class="btn" onClick={() => setT((x) => Math.min(total, x + 1))} disabled={done}>Step one cycle</button>
        <button class="btn" onClick={() => { setPlaying(false); setT(0); }}>Reset</button>
      </div>
      <p class="ix-verdict">
        <b>Utilization for this tile: {(finalUtil * 100).toFixed(0)}%</b> ={' '}
        <span class="num">M·K·N / (S² · (M + K + N − 2))</span>. Raise M to amortize the fill and drain; shrink K or N below the array
        size and those rows or columns sit idle for the whole tile.
      </p>
    </div>
  );
}
