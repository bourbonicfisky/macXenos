import { Fragment, ReactNode, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Glyph, GlyphName } from "./glyph";
import { isDashboardKey } from "./keys";
import { SHOWN_EVENT } from "../native";
import { playSound } from "./sound";

export type MenuItem = {
  id: string;
  label: string;
  glyph: GlyphName;
  /** Rows with an href navigate on select; rows without one only take focus. */
  href?: string;
  /** A current setting, shown at the right of the row (Preferences). */
  value?: ReactNode;
  /** What the right-hand pane shows while this row has focus. */
  heading: string;
  detail: ReactNode;
  /** Rows that do something when chosen (launch a game, hide the menu)
      instead of opening their pane's options. */
  onSelect?: () => void;
  /** A heading set above this row, starting a group ("Dashboard"). */
  section?: string;
  /** A picture in place of the glyph (a game's icon). */
  image?: string;
  /** Takes the feature spot above the pane while this row has focus (a
      game's art in place of the app icon). */
  feature?: ReactNode;
};

export type Tray = { label: string; href?: string; onSelect?: () => void };

/** Fired on a focused slider when ← or → steps it in a sub-menu; the slider
    (settings-controls.tsx) listens and moves by one step. */
export const ADJUST_EVENT = "blades-adjust";

type BladeMenuProps = {
  bladeId: string;
  items: MenuItem[];
  /** Sits above the detail pane, where the dashboard put the Xbox 360 logo. */
  feature?: ReactNode;
  /** Only the open blade's list answers the d-pad. */
  active: boolean;
  /** Told when A opens a pane's sub-menu and B closes it, for the legend. */
  onSubmenuChange?: (open: boolean) => void;
};

/** Below this width the list and pane can't sit side by side (the same
    breakpoint as the @min-[640px] column switch), so rows become accordions. */
const ACCORDION_BELOW = 640;

/**
 * The dashboard's one layout: a focus list on the left, and on the right a
 * pane describing whichever row the cursor is on. Every pane is rendered and
 * the unfocused ones are hidden, so all the copy is in the static HTML and
 * each row can point a screen reader at its own description.
 *
 * In a single column there is no "right", so each row opens its own pane
 * beneath it instead — an accordion, one row open at a time, the first to
 * start with. Rows that link somewhere still just go there.
 */
export function BladeMenu({
  bladeId,
  items,
  feature,
  active,
  onSubmenuChange
}: BladeMenuProps) {
  const [focus, setFocus] = useState(0);
  const [submenu, setSubmenu] = useState(false);
  const [accordion, setAccordion] = useState(false);
  const [expanded, setExpanded] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const asideRef = useRef<HTMLDivElement>(null);

  const accordionRef = useRef(false);
  accordionRef.current = accordion;

  // Decided before paint, so a phone never shows the two-column markup
  // rearranging itself.
  useLayoutEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const measure = () => setAccordion(root.clientWidth < ACCORDION_BELOW);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(root);
    return () => observer.disconnect();
  }, []);

  // Taps click through the page-wide select sound (blade-deck.tsx); the
  // keyboard path plays its own.
  const toggleRow = (index: number) => {
    setExpanded((open) => (open === index ? -1 : index));
  };

  const focusRef = useRef(0);
  focusRef.current = focus;

  // The cursor moving to another row ticks, as it did on the dashboard.
  const focusRow = (index: number) => {
    if (index !== focusRef.current) playSound("focus");
    setFocus(index);
  };

  const submenuRef = useRef(false);
  submenuRef.current = submenu;

  // The legend swaps Select (A) for Back (B), and the deck leaves ← → to the
  // sub-menu while it is open (data-submenu on the root).
  useEffect(() => {
    onSubmenuChange?.(submenu);
    if (active) document.documentElement.dataset.submenu = submenu ? "on" : "off";
  }, [submenu, active, onSubmenuChange]);

  // Leaving the blade closes its sub-menu.
  useEffect(() => {
    if (!active) setSubmenu(false);
  }, [active]);

  // Each time the app shows the dashboard: the cursor goes to the top (or to
  // a deep-linked row) and keyboard focus goes with it, so the two can't
  // drift apart when the rows changed while it was hidden.
  useEffect(() => {
    if (!active) return;
    const onShown = () => {
      const [blade, row] = window.location.hash.replace("#", "").split("/");
      const linked = blade === bladeId && row ? items.findIndex((item) => item.id === row) : -1;
      const index = Math.max(0, linked);
      setSubmenu(false);
      setFocus(index);
      setExpanded(index);
      window.requestAnimationFrame(() => {
        listRef.current?.querySelectorAll<HTMLElement>(".menu-row")[index]?.focus({ preventScroll: false });
      });
    };
    window.addEventListener(SHOWN_EVENT, onShown);
    return () => window.removeEventListener(SHOWN_EVENT, onShown);
  }, [active, bladeId, items]);

  // A deep link to a row (#controller/hotkeys) puts the cursor on it.
  useEffect(() => {
    if (!active) return;
    const [blade, row] = window.location.hash.replace("#", "").split("/");
    if (blade !== bladeId || !row) return;
    const index = items.findIndex((item) => item.id === row);
    if (index >= 0) {
      setFocus(index);
      setExpanded(index);
    }
  }, [active, bladeId, items]);

  /*
   * The d-pad, wherever keyboard focus happens to be:
   *
   *   list      ↑ ↓ walk the rows. A or Return goes where a row links, or
   *             opens the sub-menu of a pane that has options, buttons or
   *             links. ← → stay with the deck, for changing blade.
   *   sub-menu  The pane's controls in groups: each set of options is one
   *             group, any other button, link or checkbox its own. ↑ ↓ move
   *             between groups, landing on a group's current choice. ← →
   *             choose within a group (a Preferences option applies as it
   *             is reached), or step to the next group when it has only
   *             one control. A or Return presses; B or Escape goes back.
   */
  useEffect(() => {
    if (!active) return;

    const rows = () =>
      Array.from(listRef.current?.querySelectorAll<HTMLElement>(".menu-row") ?? []);
    const groups = () => {
      const pane = asideRef.current?.querySelector(".menu-detail:not([hidden])");
      const found: HTMLElement[][] = [];
      const seen = new Set<Element>();
      for (const control of pane?.querySelectorAll<HTMLElement>(
        'button:not([data-nav="skip"]), a[href], input, [role="slider"]'
      ) ?? []) {
        const radios = control.closest('[role="radiogroup"]');
        if (!radios) found.push([control]);
        else if (!seen.has(radios)) {
          seen.add(radios);
          found.push(Array.from(radios.querySelectorAll<HTMLElement>('[role="radio"]')));
        }
      }
      return found;
    };
    // Where a group is entered: its current choice, or its first control.
    const entry = (group: HTMLElement[]) =>
      group.find((c) => c.getAttribute("aria-checked") === "true") ?? group[0];
    const locate = (found: HTMLElement[][]) => {
      const current = document.activeElement as HTMLElement | null;
      for (let g = 0; g < found.length; g += 1) {
        const c = current ? found[g].indexOf(current) : -1;
        if (c >= 0) return { g, c };
      }
      return { g: -1, c: -1 };
    };
    const toGroup = (found: HTMLElement[][], g: number) => {
      if (g < 0 || g >= found.length) {
        playSound("bump");
        return;
      }
      entry(found[g]).focus();
      playSound("focus");
    };

    const openSubmenu = () => {
      const found = groups();
      if (found.length === 0) return false;
      setSubmenu(true);
      entry(found[0]).focus();
      playSound("open");
      return true;
    };

    const closeSubmenu = () => {
      setSubmenu(false);
      rows()[focusRef.current]?.focus();
      playSound("back");
    };

    const onKey = (event: KeyboardEvent) => {
      if (!isDashboardKey(event)) return;
      const { key } = event;
      const isA = key === "a" || key === "A" || key === "Enter";
      const isB = key === "b" || key === "B" || key === "Escape";

      if (submenuRef.current) {
        if (isB) {
          event.preventDefault();
          closeSubmenu();
        } else if (key === "ArrowLeft" || key === "ArrowRight") {
          event.preventDefault();
          const found = groups();
          const { g, c } = locate(found);
          const step = key === "ArrowRight" ? 1 : -1;
          const current = document.activeElement as HTMLElement | null;
          if (current?.getAttribute("role") === "slider") {
            current.dispatchEvent(new CustomEvent(ADJUST_EVENT, { detail: step }));
            return;
          }
          if (g < 0) toGroup(found, 0);
          else if (found[g].length === 1) toGroup(found, g + step);
          else {
            const next = found[g][c + step];
            if (!next) {
              playSound("bump");
              return;
            }
            next.focus();
            if (next.getAttribute("role") === "radio") next.click();
            else playSound("focus");
          }
        } else if (key === "ArrowUp" || key === "ArrowDown") {
          event.preventDefault();
          const found = groups();
          toGroup(found, locate(found).g + (key === "ArrowDown" ? 1 : -1));
        } else if (isA) {
          event.preventDefault();
          const current = document.activeElement as HTMLElement | null;
          if (current && current.getAttribute("aria-checked") !== "true") current.click();
        }
        return;
      }

      if (key === "ArrowDown" || key === "ArrowUp") {
        const found = rows();
        if (found.length === 0) return;
        const step = key === "ArrowDown" ? 1 : -1;
        const next = Math.max(0, Math.min(found.length - 1, focusRef.current + step));
        event.preventDefault();
        if (next === focusRef.current) playSound("bump");
        found[next].focus();
        return;
      }

      if (isA) {
        // A focused tab, legend button or pane control keeps its own Return;
        // the A key presses it too. Only the list itself answers A here.
        const target = event.target as HTMLElement | null;
        const onList = !target || target === document.body || target.closest(".menu-row");
        if (!onList) {
          if (key !== "Enter" && target?.matches("button, a[href], input")) {
            event.preventDefault();
            target.click();
          }
          return;
        }
        event.preventDefault();
        const item = items[focusRef.current];
        if (item?.href) rows()[focusRef.current]?.click();
        else if (item?.onSelect) {
          playSound("select");
          item.onSelect();
        } else if (accordionRef.current) {
          toggleRow(focusRef.current);
          playSound("select");
        }
        else if (!openSubmenu()) playSound("bump");
      }
    };

    // Clicking away from an open sub-menu closes it quietly.
    const onPointer = (event: PointerEvent) => {
      if (!submenuRef.current) return;
      if (!asideRef.current?.contains(event.target as Node)) setSubmenu(false);
    };

    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onPointer);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onPointer);
    };
  }, [active, items]);

  const detailId = (item: MenuItem) => `${bladeId}-${item.id}-detail`;

  const detail = (item: MenuItem, visible: boolean) => (
    <article
      key={item.id}
      id={detailId(item)}
      className="menu-detail grid content-start gap-[0.35em]"
      hidden={!visible}
    >
      <h3 className="menu-detail-heading m-0">{item.heading}</h3>
      <div className="menu-detail-body grid gap-[0.7em]">{item.detail}</div>
    </article>
  );

  const shownFeature = items[focus]?.feature ?? feature;
  const featureBlock = shownFeature ? (
    <div className="menu-feature grid justify-items-center pb-[clamp(10px,2dvh,20px)]">
      {shownFeature}
    </div>
  ) : null;

  return (
    <div
      ref={rootRef}
      className="grid flex-1 grid-cols-1 content-start gap-x-[clamp(16px,2.6%,40px)] gap-y-[clamp(18px,3dvh,32px)] @min-[640px]:grid-cols-[minmax(0,1.04fr)_minmax(0,1fr)]"
      data-accordion={accordion ? "on" : undefined}
    >
      {accordion ? featureBlock : null}

      <ul
        ref={listRef}
        className="menu-list m-0 grid list-none content-start p-0"
      >
        {items.map((item, index) => {
          const opens = accordion && !item.href && !item.onSelect;
          const open = opens && expanded === index;
          const shared = {
            className:
              "menu-row relative flex w-full items-center gap-[0.7em] rounded-row px-[0.5em] py-[clamp(6px,1.25dvh,13px)] text-left",
            "data-focused": (accordion ? open : index === focus) ? "true" : undefined,
            "aria-describedby": accordion ? undefined : detailId(item),
            onFocus: () => focusRow(index),
            // In an accordion the pointer doesn't move anything; a tap does.
            onMouseEnter: accordion ? undefined : () => focusRow(index)
          };
          const inner = (
            <>
              {item.image ? (
                <img className="glyph glyph-image" src={item.image} alt="" aria-hidden="true" />
              ) : (
                <Glyph name={item.glyph} />
              )}
              <span className="min-w-0">{item.label}</span>
              {item.value ? (
                <span className="menu-value ml-auto shrink-0 pl-3 text-[0.8em]">
                  {item.value}
                </span>
              ) : null}
              {opens ? (
                <span
                  className={`menu-chevron shrink-0${item.value ? " ml-2" : " ml-auto"}`}
                  aria-hidden="true"
                />
              ) : null}
            </>
          );

          return (
            <Fragment key={item.id}>
            {item.section ? (
              <li className="menu-section" aria-hidden="true">
                {item.section}
              </li>
            ) : null}
            <li className="menu-item">
              {item.href ? (
                <a href={item.href} {...shared}>
                  {inner}
                </a>
              ) : (
                <button
                  type="button"
                  {...shared}
                  aria-expanded={opens ? open : undefined}
                  aria-controls={opens ? detailId(item) : undefined}
                  onClick={() =>
                    item.onSelect ? item.onSelect() : opens ? toggleRow(index) : focusRow(index)
                  }
                >
                  {inner}
                </button>
              )}
              {/* The pane stays rendered and folds to zero height when shut,
                  so opening and closing can animate (content.css); inert
                  keeps a shut pane out of the tab order and away from
                  screen readers. */}
              {opens ? (
                <div className="menu-collapse" data-open={open ? "true" : undefined} inert={!open}>
                  <div className="menu-collapse-inner">{detail(item, true)}</div>
                </div>
              ) : null}
            </li>
            </Fragment>
          );
        })}
      </ul>

      {accordion ? null : (
        <div
          ref={asideRef}
          className="menu-aside grid content-start gap-[clamp(10px,1.8dvh,18px)]"
          data-submenu={submenu ? "on" : undefined}
        >
          {featureBlock}
          {items.map((item, index) => detail(item, index === focus))}
        </div>
      )}
    </div>
  );
}

/**
 * The long pill above the legend — the dashboard's "Open Tray". It is fixed
 * there, outside the scrolling content, so a long pane scrolls under nothing
 * and the pill never rides down into the legend.
 */
export function TrayPill({ tray }: { tray: Tray }) {
  const inner = (
    <>
      <span className="tray-knob grid shrink-0 place-items-center rounded-full" aria-hidden="true">
        <span className="tray-chevron" />
      </span>
      {tray.label}
    </>
  );
  const className =
    "tray relative flex items-center gap-[0.8em] rounded-full py-[0.45em] pr-6 pl-[0.45em]";
  return tray.href ? (
    <a href={tray.href} className={className}>
      {inner}
    </a>
  ) : (
    <button
      type="button"
      className={`${className} w-full cursor-pointer border-0 text-left`}
      onClick={tray.onSelect}
    >
      {inner}
    </button>
  );
}

/** A numbered run inside a description pane, for things done in order. */
export function Steps({ items }: { items: ReactNode[] }) {
  return (
    <ol className="m-0 grid list-none gap-[0.3em] p-0">
      {items.map((item, index) => (
        <li key={index} className="flex gap-[0.55em]">
          <span className="menu-step tabular-nums" aria-hidden="true">
            {index + 1}
          </span>
          <span>{item}</span>
        </li>
      ))}
    </ol>
  );
}

/** The dashboard's bulleted run inside a description pane. */
export function Bullets({ items }: { items: ReactNode[] }) {
  return (
    <ul className="m-0 grid list-none gap-[0.15em] p-0">
      {items.map((item, index) => (
        <li key={index} className="flex gap-[0.5em]">
          <span aria-hidden="true">•</span>
          <span>{item}</span>
        </li>
      ))}
    </ul>
  );
}
