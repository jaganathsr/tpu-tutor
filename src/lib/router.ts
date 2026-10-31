// Path routing under Vite's base URL (e.g. /tpu-tutor/ch/ch07/). Every
// indexable route is prerendered to static HTML at build time
// (scripts/prerender.mjs); other routes fall back to 404.html, which boots the
// app and renders from location.pathname. Old #/ links are redirected.
import { useEffect, useState } from 'preact/hooks';

const BASE = import.meta.env.BASE_URL.replace(/\/$/, ''); // '' or '/tpu-tutor'
let ssrPath: string | null = null;

/** Used by the prerenderer: render as if the browser were at `path`. */
export function setSsrPath(path: string | null) {
  ssrPath = path;
}

function normalize(p: string): string {
  const s = p.split(/[?#]/)[0].replace(/\/+$/, '');
  return s.startsWith('/') ? s || '/' : '/' + s;
}

export function currentPath(): string {
  if (ssrPath != null) return normalize(ssrPath);
  if (typeof location === 'undefined') return '/';
  const p = location.pathname.startsWith(BASE) ? location.pathname.slice(BASE.length) : location.pathname;
  return normalize(p || '/');
}

/** Internal link target for an app path such as '/ch/ch07'. Directory-style, with a trailing slash. */
export function href(path: string): string {
  const p = normalize(path);
  return `${BASE}${p === '/' ? '/' : p + '/'}`;
}

export function navigate(path: string, replace = false) {
  const url = href(path) + location.search;
  if (replace) history.replaceState(null, '', url);
  else history.pushState(null, '', url);
  window.dispatchEvent(new Event('app:navigate'));
}

export function useRoute(): string {
  const [path, setPath] = useState(currentPath());
  useEffect(() => {
    // Redirect legacy hash URLs (#/ch/ch07) to real paths.
    if (location.hash.startsWith('#/')) navigate(location.hash.slice(1), true);

    const on = () => {
      const next = currentPath();
      setPath((prev) => {
        if (prev !== next) window.scrollTo(0, 0);
        return next;
      });
    };
    // Same-origin internal links navigate without a full page load.
    const click = (e: MouseEvent) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const a = (e.target as HTMLElement).closest?.('a');
      if (!a || a.target || a.hasAttribute('download')) return;
      const url = new URL(a.href, location.href);
      if (url.origin !== location.origin || !url.pathname.startsWith(BASE + '/') || url.hash) return;
      e.preventDefault();
      history.pushState(null, '', url.pathname + url.search);
      on();
    };
    window.addEventListener('popstate', on);
    window.addEventListener('app:navigate', on);
    document.addEventListener('click', click);
    on();
    return () => {
      window.removeEventListener('popstate', on);
      window.removeEventListener('app:navigate', on);
      document.removeEventListener('click', click);
    };
  }, []);
  return path;
}
