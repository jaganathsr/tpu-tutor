import { render } from 'preact';
import '@fontsource-variable/source-serif-4/opsz.css';
import '@fontsource-variable/source-serif-4/opsz-italic.css';
import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/500.css';
import '@fontsource/ibm-plex-sans/600.css';
import '@fontsource/ibm-plex-mono/400.css';
import 'katex/dist/katex.min.css';
import './styles/tokens.css';
import './styles/base.css';
import './styles/components.css';
import { App } from './app';
import { currentPath } from './lib/router';
import { preloadForPath } from './lib/content';

// The page arrives prerendered for crawlers and first paint; the client app
// then takes over, since learner progress lives in this browser's storage.
// Fetch the current page's chunk first so the swap does not flash.
const root = document.getElementById('app')!;
preloadForPath(currentPath()).finally(() => {
  root.textContent = '';
  render(<App />, root);
});
