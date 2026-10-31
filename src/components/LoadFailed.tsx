// Shown when a lazily loaded chapter or slide chunk cannot be fetched, usually
// because the site was redeployed while the page was open.
export function LoadFailed({ what }: { what: string }) {
  return (
    <p class="empty">
      The {what} could not be loaded. The site may have been updated since this page was opened.{' '}
      <button class="btn" onClick={() => location.reload()}>Reload the page</button>
    </p>
  );
}
