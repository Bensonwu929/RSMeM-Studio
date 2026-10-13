// Ticks the elapsed-time readout inside the run-progress panel. Runs once at
// mount; each tick re-queries the DOM because streaming updates replace the
// panel's markup without removing this host element, so a stale node
// reference would otherwise stop updating after the first re-render.
setInterval(() => {
  const node = document.getElementById('workflow-timeline')?.querySelector('.run-elapsed');
  if (!node) return;
  const started = Date.parse(node.dataset.started);
  if (Number.isNaN(started)) return;
  const seconds = Math.max(0, Math.floor((Date.now() - started) / 1000));
  const minutes = Math.floor(seconds / 60);
  node.textContent = `elapsed ${minutes}:${String(seconds % 60).padStart(2, '0')}`;
}, 1000);
