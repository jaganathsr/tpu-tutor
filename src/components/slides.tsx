// Components for content/chapters/<id>/slides.mdx. Every slide is a <Slide>;
// the deck page shows one at a time and prints one per page.
import type { ComponentChildren } from 'preact';
import { mdxComponents } from './mdx';

export function Slide({ title, children, kind }: { title?: string; children?: ComponentChildren; kind?: 'statement' }) {
  return (
    <section class={`slide ${kind ?? ''}`}>
      <div class="slide-inner">
        {title ? <h2 class="slide-title">{title}</h2> : null}
        <div class="slide-body">{children}</div>
      </div>
    </section>
  );
}

/** Two columns, e.g. TPU on the left and GPU on the right. */
export function Cols({ children }: { children: ComponentChildren }) {
  return <div class="slide-cols">{children}</div>;
}

/** A headline number with its meaning underneath. */
export function Big({ n, children }: { n: string; children: ComponentChildren }) {
  return (
    <div class="slide-big">
      <div class="slide-big-n">{n}</div>
      <div class="slide-big-l">{children}</div>
    </div>
  );
}

export const slideComponents = { ...mdxComponents, Slide, Cols, Big };
