declare module '*.yaml' {
  const data: any;
  export default data;
}

declare module '*.mdx' {
  import type { ComponentType } from 'preact';
  export const frontmatter: Record<string, any>;
  const Content: ComponentType<{ components?: Record<string, unknown> }>;
  export default Content;
}
