/**
 * Help modal: open/close and tab switching.
 *
 * Uses the native <dialog> element, so Escape, focus trapping and the backdrop
 * come from the browser. Tabs follow the WAI-ARIA tabs pattern (arrow keys,
 * Home/End, roving tabindex).
 */

const dialog = document.getElementById("help-dialog");
const openButton = document.getElementById("btn-help");
const closeButton = document.getElementById("btn-help-close");
const tabs = Array.from(dialog?.querySelectorAll('[role="tab"]') ?? []);

/**
 * Show one tab panel and mark its tab selected.
 *
 * @param {HTMLElement} tab
 * @param {{ focus?: boolean }} [options]
 */
function selectTab(tab, { focus = false } = {}) {
  for (const other of tabs) {
    const selected = other === tab;
    other.setAttribute("aria-selected", String(selected));
    other.tabIndex = selected ? 0 : -1;
    const panel = document.getElementById(other.getAttribute("aria-controls") ?? "");
    if (panel) panel.hidden = !selected;
  }
  if (focus) tab.focus();
}

if (dialog && openButton) {
  openButton.addEventListener("click", () => {
    if (!dialog.open) dialog.showModal();
  });

  closeButton?.addEventListener("click", () => dialog.close());

  // A click on the backdrop lands on the dialog element itself.
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });

  dialog.addEventListener("close", () => openButton.focus());

  for (const tab of tabs) {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (event) => {
      const index = tabs.indexOf(tab);
      let next = null;
      if (event.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
      else if (event.key === "ArrowLeft") next = tabs[(index - 1 + tabs.length) % tabs.length];
      else if (event.key === "Home") next = tabs[0];
      else if (event.key === "End") next = tabs[tabs.length - 1];
      if (next) {
        event.preventDefault();
        selectTab(next, { focus: true });
      }
    });
  }
}
