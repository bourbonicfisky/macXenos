/**
 * Whether a keypress is a plain shortcut for the dashboard: no Command,
 * Control or Option held (those belong to the browser and the OS), and not
 * typed into a text field. Checkboxes and radios are part of the dashboard's
 * own menus, so the d-pad keeps working on them.
 */
export function isDashboardKey(event: KeyboardEvent): boolean {
  if (event.metaKey || event.ctrlKey || event.altKey) return false;
  const target = event.target as HTMLElement | null;
  return !target?.closest(
    'input:not([type="checkbox"]):not([type="radio"]), textarea, select'
  );
}
