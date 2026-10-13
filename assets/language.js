// The catalog is injected by studio_ui.language_script(). Only marked interface
// copy is translated; component values and analysis content stay untouched.
const appRoot = element.closest('.gradio-container');
const buttons = [...element.querySelectorAll('[data-ui-language]')];
const originals = new WeakMap();
const attributeOriginals = new WeakMap();
const protectedContent = 'textarea, input, code, pre, .filename, .rsmem-no-translate, [data-no-translate]';
const observedAttributes = ['placeholder', 'aria-label', 'title', 'alt'];
let language = 'en';
let pending = false;
try {
  if (localStorage.getItem('rsmem_ui_language') === 'zh') language = 'zh';
} catch { /* Language selection still works when browser storage is unavailable. */ }

function lookup(original) {
  if (language === 'en') return original;
  const trimmed = original.trim();
  const translated = translations[trimmed];
  if (!translated) return original;
  return original.slice(0, original.indexOf(trimmed)) + translated + original.slice(original.indexOf(trimmed) + trimmed.length);
}

function translateText(node) {
  const current = node.nodeValue;
  let record = originals.get(node);
  // A Gradio render may reuse a node with new content; capture that new source.
  if (!record || current !== record.rendered) record = {source: current, rendered: current};
  const translated = lookup(record.source);
  record.rendered = translated;
  originals.set(node, record);
  if (current !== translated) node.nodeValue = translated;
}

function translateAttributes(target) {
  if (target.closest('.rsmem-no-translate, [data-no-translate], .filename, code, pre')) return;
  let records = attributeOriginals.get(target);
  if (!records) {
    records = {};
    attributeOriginals.set(target, records);
  }
  for (const name of observedAttributes) {
    const current = target.getAttribute(name);
    if (current === null) continue;
    let record = records[name];
    if (!record || current !== record.rendered) record = {source: current, rendered: current};
    const translated = lookup(record.source);
    record.rendered = translated;
    records[name] = record;
    if (current !== translated) target.setAttribute(name, translated);
  }
}

function applyLanguage() {
  pending = false;
  observer.disconnect();
  document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
  for (const scope of appRoot.querySelectorAll('.rsmem-localized, .rsmem-localized-label > button')) {
    const walker = document.createTreeWalker(scope, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        return node.parentElement?.closest(protectedContent) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
      }
    });
    for (let node = walker.nextNode(); node; node = walker.nextNode()) translateText(node);
    translateAttributes(scope);
    scope.querySelectorAll('[placeholder], [aria-label], [title], [alt]').forEach(translateAttributes);
  }
  // Tab captions are Gradio-owned static navigation; panel contents are not.
  appRoot.querySelectorAll('.tab-container button').forEach(button => {
    const walker = document.createTreeWalker(button, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) translateText(node);
  });
  // Keep an accessible name after removing the redundant visible field label.
  const questionInput = appRoot.querySelector('#query-input textarea');
  if (questionInput) questionInput.setAttribute('aria-label', language === 'zh' ? translations.Query : 'Query');
  buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.uiLanguage === language)));
  bindDropHighlight();
  observer.observe(appRoot, {subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: observedAttributes});
}

const observer = new MutationObserver(() => {
  if (!pending) {
    pending = true;
    requestAnimationFrame(applyLanguage);
  }
});
buttons.forEach(button => button.addEventListener('click', () => {
  language = button.dataset.uiLanguage === 'zh' ? 'zh' : 'en';
  try { localStorage.setItem('rsmem_ui_language', language); } catch { /* Optional preference only. */ }
  applyLanguage();
}));

function bindDropHighlight() {
  const col = appRoot.querySelector('#imagery-col');
  if (!col || col.dataset.dropBound) return;
  col.dataset.dropBound = '1';
  let depth = 0;
  function isFiles(event) {
    return Boolean(event.dataTransfer) && Array.from(event.dataTransfer.types || []).includes('Files');
  }
  function reset() {
    depth = 0;
    col.classList.remove('dropping');
  }
  document.addEventListener('dragenter', event => {
    if (!isFiles(event)) return;
    depth += 1;
    col.classList.add('dropping');
  });
  document.addEventListener('dragover', event => {
    if (isFiles(event)) event.preventDefault();
  });
  document.addEventListener('dragleave', () => {
    depth = Math.max(0, depth - 1);
    if (depth === 0) col.classList.remove('dropping');
  });
  document.addEventListener('drop', reset);
  document.addEventListener('dragend', reset);
}

applyLanguage();
