// Hash routing, so the static build works on GitHub Pages without a 404 fallback.
import { useEffect, useState } from 'preact/hooks';

export function currentPath(): string {
  const h = location.hash.replace(/^#/, '');
  return h.startsWith('/') ? h.split('?')[0] : '/';
}

export function useRoute(): string {
  const [path, setPath] = useState(currentPath());
  useEffect(() => {
    const on = () => {
      const next = currentPath();
      setPath((prev) => {
        if (prev !== next) window.scrollTo(0, 0);
        return next;
      });
    };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return path;
}

export const href = (path: string) => `#${path}`;

export function navigate(path: string) {
  location.hash = path;
}
