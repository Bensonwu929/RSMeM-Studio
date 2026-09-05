// Local walkthrough only: these controls never start inference or edit run state.
const tabs = [...element.querySelectorAll('[data-step]')];
const panels = [...element.querySelectorAll('[role="tabpanel"]')];
const previous = element.querySelector('[data-route-prev]');
const next = element.querySelector('[data-route-next]');
let active = 0;
function selectStep(index, focus = false) {
  active = Math.max(0, Math.min(tabs.length - 1, index));
  tabs.forEach((tab, i) => {
    tab.setAttribute('aria-selected', String(i === active));
    tab.tabIndex = i === active ? 0 : -1;
    panels[i].hidden = i !== active;
  });
  element.querySelector('.route-counter').textContent = `0${active + 1} / 05`;
  previous.disabled = active === 0;
  next.disabled = active === tabs.length - 1;
  if (focus) tabs[active].focus();
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectStep(index));
  tab.addEventListener('keydown', event => {
    const keys = {ArrowDown: (index + 1) % tabs.length, ArrowUp: (index + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1};
    if (Object.hasOwn(keys, event.key)) {
      event.preventDefault();
      selectStep(keys[event.key], true);
    }
  });
});
previous.addEventListener('click', () => selectStep(active - 1, true));
next.addEventListener('click', () => selectStep(active + 1, true));
