import { useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore } from "react";
import { SHOWN_EVENT, send, useNative, type RecentTitle } from "../native";
import { whenPlayed } from "../content/home";
import type { BladeTone } from "./blade";
import { isDashboardKey } from "./keys";
import { LegendStack } from "./legend";
import { playSound } from "./sound";
import { toggleTheme, useTheme } from "./theme";
import { showToast } from "./toast";

/**
 * All Games: every game in the game folders and every game played on this
 * Mac, on a sub-blade that slides out
 * over the dashboard from the left, as the 360's game library did. Each row is
 * the game's icon, its name, and its achievements and gamerscore when a
 * signed-in profile has them. A plays, B goes back, X changes the order.
 */

// --- Open / closed, so Home's row can open it ---------------------------------

let isOpen = false;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((listener) => listener());

export function openAllGames() {
  if (isOpen) return;
  isOpen = true;
  emit();
}

function closeAllGames() {
  isOpen = false;
  emit();
}

// A deep link opens it too (#all-games), as #blade/row picks a blade.
if (typeof window !== "undefined") {
  const fromHash = () => {
    if (window.location.hash === "#all-games") openAllGames();
  };
  window.addEventListener("hashchange", fromHash);
  window.setTimeout(fromHash, 0);
}

function useAllGamesOpen() {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => isOpen,
    () => false
  );
}

// --- Order -----------------------------------------------------------------------

type Sort = "recent" | "alpha";
const SORT_KEY = "macxenos.allGames.sort";
const SORT_LABEL: Record<Sort, string> = { recent: "Most recently played", alpha: "Alphabetical" };

function savedSort(): Sort {
  try {
    return window.localStorage.getItem(SORT_KEY) === "alpha" ? "alpha" : "recent";
  } catch {
    return "recent";
  }
}

const titleOf = (game: RecentTitle) => game.name || game.path.split("/").pop() || game.path;

const byName = (a: RecentTitle, b: RecentTitle) =>
  titleOf(a).localeCompare(titleOf(b), undefined, { numeric: true, sensitivity: "base" });

/** Most recently played: played games newest first, then the rest A to Z. */
function sorted(games: RecentTitle[], sort: Sort): RecentTitle[] {
  if (sort === "alpha") return [...games].sort(byName);
  return [...games].sort((a, b) => b.lastPlayed - a.lastPlayed || byName(a, b));
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

// --- The sub-blade ----------------------------------------------------------------

const CLOSE_MS = 420;

export function AllGames({ tone }: { tone: BladeTone }) {
  const open = useAllGamesOpen();
  const { allGames, gameFolders, profiles } = useNative();
  const theme = useTheme();
  const clock = useClock();
  const [sort, setSort] = useState<Sort>(savedSort);
  const [focus, setFocus] = useState(0);
  // Mounted while open and while sliding away.
  const [shown, setShown] = useState(false);
  const [entered, setEntered] = useState(false);
  const [moreBelow, setMoreBelow] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const rowRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const returnFocus = useRef<HTMLElement | null>(null);

  const games = sorted(allGames, sort);
  const player = profiles.find((profile) => profile.slot === 0);

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
    const timer = window.setTimeout(() => {
      setShown(false);
      returnFocus.current?.focus({ preventScroll: true });
    }, CLOSE_MS);
    return () => window.clearTimeout(timer);
  }, [open]);

  const clamped = Math.min(focus, Math.max(games.length - 1, 0));

  // The highlighted row has focus and stays in view.
  useLayoutEffect(() => {
    if (!open || !shown) return;
    const row = rowRefs.current[clamped];
    row?.focus({ preventScroll: true });
    row?.scrollIntoView({ block: "nearest" });
  }, [open, shown, clamped, sort]);

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
    closeAllGames();
  };
  const play = (game: RecentTitle) => {
    closeAllGames();
    send({ op: "launch", path: game.path });
  };
  const changeSort = () => {
    const next: Sort = sort === "recent" ? "alpha" : "recent";
    setSort(next);
    setFocus(0);
    try {
      window.localStorage.setItem(SORT_KEY, next);
    } catch {
      // Remembering the order is a convenience.
    }
    playSound("toggle");
    showToast("All Games", SORT_LABEL[next], { sound: false });
  };
  const move = (to: number) => {
    const next = Math.max(0, Math.min(to, games.length - 1));
    playSound(next === clamped ? "bump" : "focus");
    setFocus(next);
  };

  // While it is open, the list has the keys (and the pad, which arrives as
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
      else if (key === "End") move(games.length - 1);
      else if (key === "ArrowLeft" || key === "ArrowRight") {
        // Nothing beside this list.
      } else if ((key === "Enter" || key === "a" || key === "A" || key === " ") && !event.repeat) {
        const game = games[clamped];
        if (game) {
          playSound("select");
          play(game);
        } else playSound("bump");
      } else if ((key === "b" || key === "B" || key === "Escape") && !event.repeat) back();
      else if ((key === "x" || key === "X") && !event.repeat) changeSort();
      else handled = false; // Y (day / night) and the dashboard key pass on.
      if (handled) {
        event.preventDefault();
        event.stopImmediatePropagation();
      }
    };
    window.addEventListener("keydown", onKey, { capture: true });
    return () => window.removeEventListener("keydown", onKey, { capture: true });
  });

  const updateMore = () => {
    const el = scrollRef.current;
    setMoreBelow(!!el && el.scrollTop + el.clientHeight < el.scrollHeight - 4);
  };
  useEffect(updateMore, [shown, games.length, sort, clamped]);

  if (!shown) return null;
  return (
    <div className="subblade-layer fixed inset-0 z-40" data-entered={entered || undefined}>
      <div className="subblade-dim absolute inset-0" onClick={back} aria-hidden="true" />
      <section
        className="subblade absolute inset-y-0 left-0"
        data-tone={tone}
        role="dialog"
        aria-modal="true"
        aria-label="All Games"
      >
        <span className="subblade-shadow pointer-events-none absolute inset-0" aria-hidden="true">
          <span className="blade-skin blade-skin--panel subblade-skin absolute inset-y-0 left-0" />
          <span className="subblade-tab-skin absolute inset-y-0 right-0" />
        </span>

        <div className="subblade-inner absolute inset-y-0 left-0 grid grid-rows-[auto_minmax(0,1fr)_auto]">
          <header className="blade-head flex min-h-[clamp(64px,12.5dvh,128px)] items-end justify-between gap-4 pr-[var(--pad)] pb-[clamp(8px,1.6dvh,16px)] pl-[calc(var(--pad)+var(--field-inset))]">
            <h2 className="m-0 text-[clamp(1.5rem,4.1dvh,2.6rem)] leading-none font-normal tracking-[0.01em]">
              All Games
            </h2>
            <span className="subblade-clock text-[clamp(1rem,2.8dvh,1.6rem)] leading-none">{clock}</span>
          </header>

          <div
            ref={scrollRef}
            onScroll={updateMore}
            className="subblade-scroll relative overflow-y-auto pr-[var(--pad)] pl-[calc(var(--pad)+var(--field-inset))] pt-[clamp(8px,1.4dvh,14px)] pb-[clamp(8px,1.6dvh,16px)] text-[clamp(1.02rem,3dvh,1.7rem)] leading-[1.25] [overscroll-behavior:contain]"
          >
            {games.length === 0 ? (
              <p className="subblade-empty m-0 py-[1em]">
                {gameFolders.scanning
                  ? "Looking for games in your game folders."
                  : gameFolders.folders.length
                    ? "No games were found in your game folders."
                    : "Add the folders that hold your games on Home, under Game folders."}
              </p>
            ) : (
              <ul className="game-list m-0 grid list-none p-0" aria-label={`All Games, ${SORT_LABEL[sort]}`}>
                {games.map((game, index) => (
                  <li key={game.path} className="game-item">
                    <button
                      ref={(el) => {
                        rowRefs.current[index] = el;
                      }}
                      type="button"
                      className="menu-row game-row grid w-full items-center gap-x-[0.7em] rounded-row px-[0.4em] py-[clamp(4px,0.8dvh,8px)] text-left"
                      data-focused={index === clamped ? "true" : undefined}
                      onMouseEnter={() => setFocus(index)}
                      onFocus={() => setFocus(index)}
                      onClick={() => play(game)}
                    >
                      <img
                        className="game-icon"
                        src={game.icon || "/macxenos-icon-dark.webp"}
                        alt=""
                        aria-hidden="true"
                      />
                      <span className="grid min-w-0">
                        <span className="game-name truncate">{titleOf(game)}</span>
                        <span className="game-sub truncate">
                          {game.achievements
                            ? `${game.achievements.unlocked} of ${game.achievements.total} Achievements`
                            : game.lastPlayed
                              ? whenPlayed(game.lastPlayed)
                              : "Not played yet"}
                        </span>
                      </span>
                      {game.achievements ? (
                        <span className="game-score inline-flex items-center gap-[0.25em]">
                          {game.achievements.score}
                          <span className="gamerscore-g" aria-label="gamerscore">
                            G
                          </span>
                        </span>
                      ) : (
                        <span />
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <footer className="blade-legend relative flex min-h-[clamp(64px,16.5dvh,170px)] items-center justify-between gap-4 pr-[var(--pad)] pt-[clamp(8px,1.6dvh,16px)] pb-[clamp(12px,3.4dvh,32px)] pl-[calc(var(--pad)+var(--field-inset))] text-[clamp(0.9rem,2.6dvh,1.4rem)]">
            {/* More below: the dashboard's small arrow under the list. */}
            {moreBelow ? <span className="subblade-more-arrow absolute top-[0.45em] right-[calc(var(--pad)+0.35em)]" aria-hidden="true" /> : null}
            <LegendStack
              side="left"
              entries={[
                { button: "y", label: theme === "dark" ? "Day" : "Night", onClick: toggleTheme },
                ...(games.length > 1
                  ? [{ button: "x" as const, label: `Sort: ${SORT_LABEL[sort]}`, onClick: changeSort }]
                  : [])
              ]}
            />
            <LegendStack
              side="right"
              entries={[
                { button: "b", label: "Back", onClick: back },
                ...(games.length ? [{ button: "a" as const, label: "Select" }] : [])
              ]}
            />
          </footer>
        </div>

        {/* The sub-blade's own tab, with the signed-in player on it. */}
        <span className="subblade-tab absolute inset-y-0 right-0 flex justify-center pt-[21.5dvh] text-[clamp(1.05rem,3.2dvh,2.1rem)]" aria-hidden="true">
          <span className="subblade-tab-label">{player?.gamertag ?? ""}</span>
        </span>
      </section>
    </div>
  );
}
