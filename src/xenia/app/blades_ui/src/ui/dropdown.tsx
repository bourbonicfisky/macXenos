import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { isDashboardKey } from "./keys";
import { playSound } from "./sound";

/* ==========================================================================
   A drop-down for settings with too many choices to lay out as sheets
   (language, country, time zone, resolution).

   At rest it is one sheet showing the current choice, so in a sub-menu it is
   a single group like any button. A or a click opens the list over the pane:
   ↑ ↓ (or the d-pad) move, A picks, B or Escape closes without changing
   anything. While open it takes every key first, so the blade underneath
   does not move.
   ========================================================================== */

export type DropdownOption<V extends string | number> = { value: V; label: string };

export function Dropdown<V extends string | number>({
  title,
  options,
  value,
  onChoose,
  disabled = false
}: {
  title: string;
  options: DropdownOption<V>[];
  value: V;
  onChoose: (option: DropdownOption<V>) => void;
  /** Shown but locked (console settings while a game runs). */
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [cursor, setCursorState] = useState(0);
  const cursorRef = useRef(0);
  const setCursor = (index: number) => {
    cursorRef.current = index;
    setCursorState(index);
  };
  const buttonRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const current = options.find((o) => o.value === value);

  const show = () => {
    if (disabled) {
      playSound("bump");
      return;
    }
    const index = Math.max(0, options.findIndex((o) => o.value === value));
    setCursor(index);
    setOpen(true);
    playSound("open");
  };

  const close = (choice?: DropdownOption<V>) => {
    setOpen(false);
    buttonRef.current?.focus();
    if (choice && choice.value !== value) {
      onChoose(choice);
    } else {
      playSound("back");
    }
  };

  // Keys go to the open list before anything else on the page.
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (!isDashboardKey(event) && event.key !== "Escape") return;
      const { key } = event;
      let handled = true;
      const c = cursorRef.current;
      const clamp = (n: number) => Math.max(0, Math.min(options.length - 1, n));
      if (key === "ArrowDown" || key === "ArrowUp") {
        const next = clamp(c + (key === "ArrowDown" ? 1 : -1));
        playSound(next === c ? "bump" : "focus");
        setCursor(next);
      } else if (key === "PageDown" || key === "PageUp") {
        setCursor(clamp(c + (key === "PageDown" ? 8 : -8)));
      } else if (key === "a" || key === "A" || key === "Enter") {
        close(options[c]);
      } else if (key === "b" || key === "B" || key === "Escape") {
        close();
      } else if (key === "ArrowLeft" || key === "ArrowRight" || key === "y" || key === "Y") {
        // Swallowed: the list stays put, the blade too.
      } else {
        handled = false;
      }
      if (handled) {
        event.preventDefault();
        event.stopImmediatePropagation();
      }
    };
    // Clicking outside the list closes it.
    const onPointer = (event: PointerEvent) => {
      if (!listRef.current?.contains(event.target as Node) && event.target !== buttonRef.current) {
        close();
      }
    };
    window.addEventListener("keydown", onKey, { capture: true });
    window.addEventListener("pointerdown", onPointer, { capture: true });
    return () => {
      window.removeEventListener("keydown", onKey, { capture: true });
      window.removeEventListener("pointerdown", onPointer, { capture: true });
    };
  });

  // The cursor stays in view in a long list.
  useLayoutEffect(() => {
    if (!open) return;
    listRef.current
      ?.querySelector<HTMLElement>(`[data-index="${cursor}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [open, cursor]);

  return (
    <div className="relative mt-[0.3em]">
      <button
        ref={buttonRef}
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={`${title}: ${current?.label ?? ""}`}
        aria-disabled={disabled || undefined}
        className="pref-option pref-dropdown rounded-row flex min-h-[2.2em] w-full cursor-pointer items-center gap-[0.6em] px-[0.85em] text-left"
        onClick={() => (open ? close() : show())}
      >
        <span className="min-w-0 flex-1 truncate">{current?.label ?? "—"}</span>
        <span className="menu-chevron shrink-0" aria-hidden="true" />
      </button>
      {open ? (
        <ul
          ref={listRef}
          role="listbox"
          aria-label={title}
          className="pref-dropdown-list rounded-row absolute inset-x-0 top-[calc(100%+0.3em)] z-30 m-0 grid max-h-[44dvh] list-none overflow-y-auto p-[0.25em]"
        >
          {options.map((option, index) => (
            <li
              key={String(option.value)}
              role="option"
              data-index={index}
              aria-selected={option.value === value}
              data-focused={index === cursor ? "true" : undefined}
              className="pref-dropdown-item rounded-row flex cursor-pointer items-center gap-[0.5em] px-[0.7em] py-[0.35em]"
              onPointerEnter={() => setCursor(index)}
              onClick={() => close(option)}
            >
              <span className="pref-dropdown-tick shrink-0" aria-hidden="true">
                {option.value === value ? "✓" : ""}
              </span>
              <span className="min-w-0 flex-1">{option.label}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
