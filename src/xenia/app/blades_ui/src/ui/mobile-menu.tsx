import { useEffect, useRef, useState } from "react";
import type { BladeDef } from "./blade";
import { playSound } from "./sound";

/** "under the hood" → "Under the hood". Tab labels are lowercase, as on the
    console; a menu list reads better in sentence case. */
const sentenceCase = (label: string) => label.charAt(0).toUpperCase() + label.slice(1);

/**
 * Below the rails' breakpoint (860px) the blades are reached from a top bar:
 * the app's name on the left, a hamburger on the right, and a dropdown that
 * stacks every blade as a small sheet in its own colour. Picking one, tapping
 * outside, or Escape closes it.
 */
export function MobileMenu({
  blades,
  active,
  onOpen
}: {
  blades: BladeDef[];
  active: number;
  onOpen: (index: number) => void;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const navRef = useRef<HTMLElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const close = (returnFocus: boolean) => {
    setMenuOpen(false);
    if (returnFocus) buttonRef.current?.focus();
  };

  // Opening moves focus to the current blade, so the keyboard lands in the
  // list rather than behind it.
  useEffect(() => {
    if (!menuOpen) return;
    listRef.current?.querySelector<HTMLElement>('[aria-current="true"]')?.focus();

    // While open, the menu owns Escape and ↑ ↓; the blade underneath
    // doesn't see them.
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopImmediatePropagation();
        playSound("back");
        close(true);
      } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        event.stopImmediatePropagation();
        const items = Array.from(
          listRef.current?.querySelectorAll<HTMLElement>(".deck-menu-item") ?? []
        );
        const at = items.indexOf(document.activeElement as HTMLElement);
        const next = items[at + (event.key === "ArrowDown" ? 1 : -1)];
        if (next) {
          next.focus();
          playSound("focus");
        } else playSound("bump");
      }
    };
    const onPointer = (event: PointerEvent) => {
      if (!navRef.current?.contains(event.target as Node)) close(false);
    };
    // Widening past the breakpoint hides the bar; don't leave it open.
    const wide = window.matchMedia("(min-width: 861px)");
    const onWide = () => wide.matches && close(false);

    window.addEventListener("keydown", onKey, { capture: true });
    window.addEventListener("pointerdown", onPointer);
    wide.addEventListener("change", onWide);
    return () => {
      window.removeEventListener("keydown", onKey, { capture: true });
      window.removeEventListener("pointerdown", onPointer);
      wide.removeEventListener("change", onWide);
    };
  }, [menuOpen]);

  return (
    <nav
      ref={navRef}
      className="deck-strip sticky top-0 z-30 hidden max-[860px]:block"
      aria-label="Blades"
    >
      <div className="flex items-center justify-between gap-3 px-4 py-2">
        <a href="#home" className="deck-brand inline-flex items-center gap-2.5" onClick={() => close(false)}>
          <img src="/macxenos-icon-dark.webp" alt="" width={30} height={30} className="deck-brand-icon" />
          <span className="text-[1.15rem] font-semibold">macXenos</span>
        </a>

        <button
          ref={buttonRef}
          type="button"
          className="deck-burger grid h-11 w-11 cursor-pointer place-items-center rounded-row"
          aria-expanded={menuOpen}
          aria-controls="deck-menu"
          onClick={() => {
            playSound(menuOpen ? "back" : "open");
            setMenuOpen(!menuOpen);
          }}
        >
          <span className="sr-only">{menuOpen ? "Close menu" : "Menu"}</span>
          <span className="deck-burger-bars" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
        </button>
      </div>

      <ul
        ref={listRef}
        id="deck-menu"
        className="deck-menu absolute inset-x-0 top-full m-0 grid list-none px-3 pt-2 pb-4"
        hidden={!menuOpen}
      >
        {blades.map((blade, index) => (
          <li key={blade.id} className="deck-menu-row">
            <button
              type="button"
              className="deck-menu-item relative flex w-full cursor-pointer items-center px-5 pt-5 pb-3.5 text-left text-[1.3rem]"
              data-tone={blade.tone}
              aria-current={index === active ? "true" : undefined}
              style={{ zIndex: blades.length - index }}
              onClick={() => {
                onOpen(index);
                close(true);
              }}
            >
              {sentenceCase(blade.label)}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
