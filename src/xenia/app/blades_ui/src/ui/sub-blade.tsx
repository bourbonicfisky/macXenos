import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useSyncExternalStore,
  type ReactNode
} from "react";
import { SHOWN_EVENT, useNative } from "../native";
import type { BladeTone } from "./blade";
import { isDashboardKey } from "./keys";
import { LegendStack, type LegendEntry } from "./legend";
import { TabLabel } from "./tab-label";
import { playSound } from "./sound";

/**
 * A sub-blade: a list that slides out over the dashboard, as the 360's game
 * library and its system lists did. It comes from the left (All Games) or
 * from the right (Game Folders), dims the dashboard behind it, carries its
 * own tab with the signed-in player on it and a clock in its header, and
 * owns the keys while it is open: ↑ ↓ move, A chooses, B goes back, and Y
 * (day and night on the blades) does nothing here.
 */

// --- Which one is open ------------------------------------------------------------

export type SubBladeId = "all-games" | "game-folders";

let openId: SubBladeId | null = null;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((listener) => listener());

export function openSubBlade(id: SubBladeId) {
  if (openId === id) return;
  openId = id;
  emit();
}

export function closeSubBlade() {
  openId = null;
  emit();
}

export function useSubBladeOpen(id: SubBladeId) {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => openId === id,
    () => false
  );
}

// A deep link opens one too (#all-games, #game-folders).
if (typeof window !== "undefined") {
  const fromHash = () => {
    const id = window.location.hash.replace("#", "");
    if (id === "all-games" || id === "game-folders") openSubBlade(id);
  };
  window.addEventListener("hashchange", fromHash);
  window.setTimeout(fromHash, 0);
}

function useClock() {
  const format = () => new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const [now, setNow] = useState(format);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(format()), 15_000);
    return () => window.clearInterval(timer);
  }, []);
  return now;
}

// --- The frame ----------------------------------------------------------------------

/** The blades' slide (tokens.css --slide, longer for slow motion): leaving
    takes exactly as long as arriving, so going back is the same slide run in
    reverse rather than a cut. */
function slideMs(): number {
  const value = getComputedStyle(document.documentElement).getPropertyValue("--slide").trim();
  const ms = value.endsWith("ms") ? parseFloat(value) : parseFloat(value) * 1000;
  return Number.isFinite(ms) ? ms : 520;
}

export type RowProps = {
  ref: (element: HTMLButtonElement | null) => void;
  className: string;
  "data-focused": "true" | undefined;
  onMouseEnter: () => void;
  onFocus: () => void;
  onClick: () => void;
};

type SubBladeProps = {
  id: SubBladeId;
  side: "left" | "right";
  /** The blade's own colour, or "metal": the brushed silver of the
      console's system sub-blades (Profile, Settings). */
  tone: BladeTone | "metal";
  title: string;
  count: number;
  /** A, Return or a click on row `index`. */
  onActivate: (index: number) => void;
  /** X, when the list has a use for it. */
  onX?: () => void;
  /** Called whenever the highlighted row changes. */
  onFocusChange?: (index: number) => void;
  /** Starting over at the first row (a new sort order). */
  resetKey?: unknown;
  legend: (focus: number) => { left: LegendEntry[]; right: LegendEntry[] };
  empty?: ReactNode;
  children: (rowProps: (index: number) => RowProps, focus: number) => ReactNode;
};

export function SubBlade({
  id,
  side,
  tone,
  title,
  count,
  onActivate,
  onX,
  onFocusChange,
  resetKey,
  legend,
  empty,
  children
}: SubBladeProps) {
  const open = useSubBladeOpen(id);
  const { profiles } = useNative();
  const clock = useClock();
  const [focus, setFocusState] = useState(0);
  // Mounted while open and while sliding away.
  const [shown, setShown] = useState(false);
  const [entered, setEntered] = useState(false);
  const [moreBelow, setMoreBelow] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const rowRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const returnFocus = useRef<HTMLElement | null>(null);
  const player = profiles.find((profile) => profile.slot === 0);

  const clamped = Math.min(focus, Math.max(count - 1, 0));
  const setFocus = (index: number) => {
    setFocusState(index);
    onFocusChange?.(index);
  };

  // Slide in on open; slide out, then unmount, on close.
  useEffect(() => {
    if (open) {
      returnFocus.current = document.activeElement as HTMLElement | null;
      setFocus(0);
      setShown(true);
      const frame = window.requestAnimationFrame(() =>
        window.requestAnimationFrame(() => setEntered(true))
      );
      return () => window.cancelAnimationFrame(frame);
    }
    setEntered(false);
    // Unmounted once it is fully off screen. Reduced motion has no slide, so
    // this is a timer rather than a wait for transitionend.
    const timer = window.setTimeout(() => {
      setShown(false);
      returnFocus.current?.focus({ preventScroll: true });
    }, slideMs() + 40);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (open) setFocus(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resetKey]);

  // The highlighted row has focus and stays in view.
  useLayoutEffect(() => {
    if (!open || !shown) return;
    const row = rowRefs.current[clamped];
    row?.focus({ preventScroll: true });
    row?.scrollIntoView({ block: "nearest" });
  }, [open, shown, clamped, resetKey, count]);

  // Showing the dashboard again puts focus back on this list, not the blade.
  useEffect(() => {
    if (!open) return;
    const onShown = () =>
      window.setTimeout(() => rowRefs.current[clamped]?.focus({ preventScroll: true }), 0);
    window.addEventListener(SHOWN_EVENT, onShown);
    return () => window.removeEventListener(SHOWN_EVENT, onShown);
  }, [open, clamped]);

  const back = () => {
    playSound("back");
    closeSubBlade();
  };
  const move = (to: number) => {
    const next = Math.max(0, Math.min(to, count - 1));
    playSound(next === clamped ? "bump" : "focus");
    setFocus(next);
  };

  // While it is open the list has the keys (and the pad, which arrives as
  // keys): ← and → don't change blade underneath it.
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (!isDashboardKey(event)) return;
      const key = event.key;
      let handled = true;
      if (key === "ArrowUp") move(clamped - 1);
      else if (key === "ArrowDown") move(clamped + 1);
      else if (key === "PageUp") move(clamped - 6);
      else if (key === "PageDown") move(clamped + 6);
      else if (key === "Home") move(0);
      else if (key === "End") move(count - 1);
      else if (key === "ArrowLeft" || key === "ArrowRight" || key === "y" || key === "Y") {
        // Nothing beside this list; day and night belong to the blades.
      } else if ((key === "Enter" || key === "a" || key === "A" || key === " ") && !event.repeat) {
        if (count) {
          playSound("select");
          onActivate(clamped);
        } else playSound("bump");
      } else if ((key === "b" || key === "B" || key === "Escape") && !event.repeat) back();
      else if ((key === "x" || key === "X") && !event.repeat) onX?.();
      else handled = false; // The dashboard key passes on.
      if (handled) {
        event.preventDefault();
        event.stopImmediatePropagation();
      }
    };
    window.addEventListener("keydown", onKey, { capture: true });
    return () => window.removeEventListener("keydown", onKey, { capture: true });
  });

  const updateMore = () => {
    const element = scrollRef.current;
    setMoreBelow(!!element && element.scrollTop + element.clientHeight < element.scrollHeight - 4);
  };
  useEffect(updateMore, [shown, count, clamped, resetKey]);

  if (!shown) return null;

  const rowProps = (index: number): RowProps => ({
    ref: (element) => {
      rowRefs.current[index] = element;
    },
    // Metal rows are square, on grooves; the coloured ones are rounded like
    // the blades' rows.
    className: `menu-row subblade-row relative grid w-full items-center gap-x-[0.7em] ${
      tone === "metal" ? "" : "rounded-row "
    }px-[0.4em] py-[clamp(4px,0.8dvh,8px)] text-left`,
    "data-focused": index === clamped ? "true" : undefined,
    onMouseEnter: () => {
      if (index !== clamped) setFocus(index);
    },
    onFocus: () => {
      if (index !== clamped) setFocus(index);
    },
    onClick: () => onActivate(index)
  });
  const { left, right } = legend(clamped);
  // Content sits clear of the screen edge on the open side and of the tab
  // on the other.
  // The tab bows in over the sheet by --bow at mid-height, so that side
  // keeps clear of it too.
  const inset =
    side === "left"
      ? "pr-[calc(var(--pad)+var(--bow))] pl-[calc(var(--pad)+var(--field-inset))]"
      : "pl-[calc(var(--pad)+var(--bow))] pr-[calc(var(--pad)+var(--field-inset))]";

  return (
    <div className="subblade-layer fixed inset-0 z-40" data-entered={entered || undefined}>
      <div className="subblade-dim absolute inset-0" onClick={back} aria-hidden="true" />
      <section
        className="subblade absolute inset-y-0"
        data-side={side}
        data-tone={tone}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <span className="subblade-shadow pointer-events-none absolute inset-0" aria-hidden="true">
          <span className="blade-skin blade-skin--panel subblade-skin absolute inset-y-0" />
        </span>

        <div className="subblade-inner absolute inset-y-0 grid grid-rows-[auto_minmax(0,1fr)_auto]">
          <header
            className={`blade-head flex min-h-[clamp(64px,12.5dvh,128px)] items-end justify-between gap-4 pb-[clamp(8px,1.6dvh,16px)] ${inset}`}
          >
            <h2 className="m-0 text-[clamp(1.5rem,4.1dvh,2.6rem)] leading-none font-normal tracking-[0.01em]">
              {title}
            </h2>
            <span className="subblade-clock text-[clamp(1rem,2.8dvh,1.6rem)] leading-none">{clock}</span>
          </header>

          <div
            ref={scrollRef}
            onScroll={updateMore}
            className={`subblade-scroll relative overflow-y-auto pt-[clamp(8px,1.4dvh,14px)] pb-[clamp(8px,1.6dvh,16px)] text-[clamp(1.02rem,3dvh,1.7rem)] leading-[1.25] [overscroll-behavior:contain] ${inset}`}
          >
            {count === 0 && empty ? <p className="subblade-empty m-0 py-[1em]">{empty}</p> : null}
            {children(rowProps, clamped)}
          </div>

          <footer
            className={`blade-legend relative flex min-h-[clamp(64px,16.5dvh,170px)] items-center justify-between gap-4 pt-[clamp(8px,1.6dvh,16px)] pb-[clamp(12px,3.4dvh,32px)] text-[clamp(0.9rem,2.6dvh,1.4rem)] ${inset}`}
          >
            {/* More below: the dashboard's small arrow under the list. */}
            {moreBelow ? (
              <span
                className={`subblade-more-arrow absolute top-[0.45em] ${
                  side === "left" ? "right-[calc(var(--pad)+0.35em)]" : "right-[calc(var(--pad)+var(--field-inset)+0.35em)]"
                }`}
                aria-hidden="true"
              />
            ) : null}
            {/* Kept even when empty, so B and A stay at the right. */}
            <span>
              <LegendStack side="left" entries={left} />
            </span>
            <LegendStack
              side="right"
              entries={right.map((entry) => (entry.button === "b" && !entry.onClick ? { ...entry, onClick: back } : entry))}
            />
          </footer>
        </div>

        {/* The sub-blade's tab is a blade tab: the same ribbon, the same
            highlight (the folder-tab profile measured off the console) and
            the same curved label, built from the blades' own parts. It
            reads as the open blade's tab, mirrored when the sub-blade comes
            from the left (data-bend, as on the rails). It lies over the
            sheet's header and legend bars, which run in under its curve. */}
        <span
          className="subblade-tab absolute inset-y-0"
          data-bend={side === "right" ? "left" : "right"}
          aria-hidden="true"
        >
          <span className="blade-shadow blade-shadow--tab pointer-events-none absolute inset-0">
            <span className="blade-skin blade-skin--tab absolute inset-0" />
            <span className="blade-ribbon absolute inset-0">
              <span className="blade-skin blade-skin--gloss absolute inset-0" />
            </span>
          </span>
          <span className="blade-tab relative flex h-full items-start pt-[21.5dvh] pb-8 text-[clamp(1.05rem,3.2dvh,2.1rem)] font-normal tracking-[0.01em]">
            <TabLabel id={`sub-${id}`} label={player?.gamertag ?? ""} side={side === "right" ? "center" : "right"} />
          </span>
        </span>
      </section>
    </div>
  );
}
