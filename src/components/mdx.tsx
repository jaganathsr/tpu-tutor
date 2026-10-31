// Components available inside chapter MDX without imports.
import type { ComponentChildren } from 'preact';
import sources from '../generated/sources.json';
import chips from '../generated/chips.json';
import { si } from '../lib/format';
import { RooflinePlotter } from '../interactives/RooflinePlotter';
import { GenerationChart } from '../interactives/GenerationChart';
import { SystolicArray } from '../interactives/SystolicArray';
import { TensorCoreDiagram } from '../interactives/TensorCoreDiagram';
import { TorusExplorer } from '../interactives/TorusExplorer';

type Src = { id: string; title: string; url: string; section: string };
const SRC = sources as Record<string, Src>;

export function sourceUrl(id: string, at?: string) {
  const s = SRC[id];
  if (!s) return undefined;
  return at && at.startsWith('#') ? s.url.replace(/#.*$/, '') + at : s.url;
}

/** Inline citation: <Cite id="SB1" at="#matrix-multiplication" s="§matmul" /> */
export function Cite({ id, at, s }: { id: string; at?: string; s?: string }) {
  const src = SRC[id];
  if (!src) return <span class="vtag">unknown source {id}</span>;
  const label = s ?? (at && !at.startsWith('#') ? at : undefined);
  return (
    <a class="cite" href={sourceUrl(id, at)} title={src.title} target="_blank" rel="noreferrer">
      [{id}
      {label ? `, ${label}` : ''}]
    </a>
  );
}

/** The plain-English line that follows every formula. */
export function Gloss({ children }: { children: ComponentChildren }) {
  return <p class="gloss">{children}</p>;
}

export function GpuAnalogy({ children, title }: { children: ComponentChildren; title?: string }) {
  return (
    <aside class="callout" aria-label="GPU analogy">
      <div class="callout-label">{title ?? 'GPU analogy'}</div>
      {children}
    </aside>
  );
}

export function AnalogyBreaks({ children }: { children: ComponentChildren }) {
  return (
    <aside class="callout breaks" aria-label="Where the analogy breaks">
      <div class="callout-label">Where the analogy breaks</div>
      {children}
    </aside>
  );
}

/** Lets a table or figure extend into the margin column on wide screens. */
export function Wide({ children }: { children: ComponentChildren }) {
  return <div class="wide">{children}</div>;
}

export function Note({ children }: { children: ComponentChildren }) {
  return <aside class="margin-note">{children}</aside>;
}

export function Unverified({ children }: { children?: ComponentChildren }) {
  return (
    <span class="vtag" title="No primary source confirms this. See research/sources.md.">
      unverified{children ? `: ` : ''}
      {children}
    </span>
  );
}

export function Provisional({ children }: { children?: ComponentChildren }) {
  return (
    <span class="vtag" title="From the JAX Pallas hardware table, not in Google's published spec.">
      provisional{children ? `: ` : ''}
      {children}
    </span>
  );
}

export function Derived() {
  return (
    <span class="vtag" title="Computed by scripts/ from cited inputs; no single source states it.">
      derived
    </span>
  );
}

export function Worked({ title, children }: { title: string; children: ComponentChildren }) {
  return (
    <section class="worked">
      <div class="worked-label">Worked example: {title}</div>
      {children}
    </section>
  );
}

export function Interactive({ caption, children }: { caption: ComponentChildren; children: ComponentChildren }) {
  return (
    <figure class="interactive">
      {children}
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

type ChipRow = (typeof chips)[keyof typeof chips];
const COLS: Record<string, { head: string; get: (c: ChipRow) => number | null; unit: string; prov?: string }> = {
  bf16: { head: 'BF16', get: (c) => c.flops.bf16, unit: 'FLOP/s', prov: 'flops_bf16' },
  fp8: { head: 'FP8', get: (c) => c.flops.fp8, unit: 'FLOP/s', prov: 'flops_fp8' },
  int8: { head: 'INT8', get: (c) => c.flops.int8, unit: 'OP/s', prov: 'flops_int8' },
  int4: { head: 'INT4', get: (c) => c.flops.int4, unit: 'OP/s', prov: 'flops_int4' },
  fp4: { head: 'FP4', get: (c) => c.flops.fp4, unit: 'FLOP/s' },
  hbm: { head: 'HBM', get: (c) => c.hbm_bytes, unit: 'B' },
  hbm_bw: { head: 'HBM BW', get: (c) => c.hbm_bw, unit: 'B/s' },
  ai: { head: 'BF16 intensity', get: (c) => (c.flops.bf16 ? c.flops.bf16 / c.hbm_bw : null), unit: 'FLOP/B', prov: 'flops_bf16' },
};

/** Spec table generated from scripts/specs.py, so prose never retypes numbers. */
export function ChipTable({ chips: ids, cols }: { chips: string; cols: string }) {
  const list = ids.split(/\s+/).map((k) => [k, (chips as Record<string, ChipRow>)[k]] as const);
  const cs = cols.split(/\s+/);
  return (
    <div class="table-wrap">
    <table class="spec-table">
      <thead>
        <tr>
          <th>Chip</th>
          {cs.map((c) => (
            <th class="r">{COLS[c].head}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {list.map(([k, c]) => (
          <tr key={k}>
            <td>{c?.name ?? k}</td>
            {cs.map((col) => {
              const def = COLS[col];
              const v = c ? def.get(c) : null;
              const prov = def.prov && (c?.provisional as string[] | undefined)?.includes(def.prov);
              const text = v == null ? '—' : col === 'ai' ? Math.round(v).toString() : col === 'hbm' ? `${Math.round(v / 2 ** 30)} GiB` : si(v, def.unit);
              return (
                <td class="r">
                  {text}
                  {prov && v != null ? (
                    <span class="vtag" title="Provisional: from the JAX Pallas hardware table, not in Google's published spec.">
                      p
                    </span>
                  ) : null}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
    {list.some(([, c]) => c?.provisional.length) ? (
      <p class="table-note">
        <span class="vtag">p</span> provisional: from the JAX Pallas hardware table, not in Google's published spec.
      </p>
    ) : null}
    </div>
  );
}

export const mdxComponents = {
  Cite,
  Gloss,
  GpuAnalogy,
  AnalogyBreaks,
  Note,
  Wide,
  Unverified,
  Provisional,
  Derived,
  Worked,
  Interactive,
  ChipTable,
  RooflinePlotter,
  GenerationChart,
  SystolicArray,
  TensorCoreDiagram,
  TorusExplorer,
};
