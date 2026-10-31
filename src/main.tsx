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

render(<App />, document.getElementById('app')!);
