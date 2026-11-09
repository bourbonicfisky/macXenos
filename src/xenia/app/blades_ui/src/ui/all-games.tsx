import { useState } from "react";
import { send, useNative, type RecentTitle } from "../native";
import { whenPlayed } from "../content/home";
import type { BladeTone } from "./blade";
import { playSound } from "./sound";
import { closeSubBlade, openSubBlade, SubBlade } from "./sub-blade";
import { showToast } from "./toast";
import { t, type MessageKey } from "../i18n";

/**
 * All Games: every game in the game folders and every game played on this
 * Mac, on a sub-blade from the left, as the 360's game library was. Each row
 * is the game's icon, its name, and its achievements and gamerscore when a
 * signed-in profile has them. A plays, B goes back, X changes the order.
 */

export const openAllGames = () => openSubBlade("all-games");

type Sort = "recent" | "alpha";
const SORT_KEY = "macxenos.allGames.sort";
const SORT_LABEL: Record<Sort, MessageKey> = { recent: "allGames.sort.recent", alpha: "allGames.sort.alpha" };

function savedSort(): Sort {
  try {
    return window.localStorage.getItem(SORT_KEY) === "alpha" ? "alpha" : "recent";
  } catch {
    return "recent";
  }
}

const titleOf = (game: RecentTitle) => {
  const name = game.name || game.path.split("/").pop() || game.path;
  return game.disc ? t("allGames.disc", { name, disc: game.disc }) : name;
};

const byName = (a: RecentTitle, b: RecentTitle) =>
  titleOf(a).localeCompare(titleOf(b), undefined, { numeric: true, sensitivity: "base" });

/** Most recently played: played games newest first, then the rest A to Z. */
function sorted(games: RecentTitle[], sort: Sort): RecentTitle[] {
  if (sort === "alpha") return [...games].sort(byName);
  return [...games].sort((a, b) => b.lastPlayed - a.lastPlayed || byName(a, b));
}

export function AllGames({ tone }: { tone: BladeTone }) {
  const { allGames, gameFolders } = useNative();
  const [sort, setSort] = useState<Sort>(savedSort);
  const games = sorted(allGames, sort);

  const play = (index: number) => {
    const game = games[index];
    if (!game) return;
    closeSubBlade();
    send({ op: "launch", path: game.path });
  };
  const changeSort = () => {
    if (games.length < 2) return;
    const next: Sort = sort === "recent" ? "alpha" : "recent";
    setSort(next);
    try {
      window.localStorage.setItem(SORT_KEY, next);
    } catch {
      // Remembering the order is a convenience.
    }
    playSound("toggle");
    showToast(t("allGames.title"), t(SORT_LABEL[next]), { sound: false });
  };

  return (
    <SubBlade
      id="all-games"
      side="left"
      tone={tone}
      title={t("allGames.title")}
      count={games.length}
      onActivate={play}
      onX={changeSort}
      resetKey={sort}
      legend={() => ({
        left:
          games.length > 1
            ? [{ button: "x", label: t("allGames.sort", { order: t(SORT_LABEL[sort]) }), onClick: changeSort }]
            : [],
        right: [
          { button: "b", label: t("legend.back") },
          ...(games.length ? [{ button: "a" as const, label: t("legend.select") }] : [])
        ]
      })}
      empty={
        gameFolders.scanning
          ? t("allGames.looking")
          : gameFolders.folders.length
            ? t("allGames.noneFound")
            : t("allGames.noFolders")
      }
    >
      {(rowProps) =>
        games.length ? (
          <ul className="subblade-list m-0 grid list-none p-0" aria-label={`${t("allGames.title")}, ${t(SORT_LABEL[sort])}`}>
            {games.map((game, index) => (
              <li key={game.path} className="subblade-item">
                <button type="button" {...rowProps(index)}>
                  <img className="game-icon" src={game.icon || "/macxenos-icon-dark.webp"} alt="" aria-hidden="true" />
                  <span className="grid min-w-0">
                    <span className="subblade-name truncate">{titleOf(game)}</span>
                    <span className="subblade-sub truncate">
                      {game.achievements
                        ? t("allGames.achievements", {
                            unlocked: game.achievements.unlocked,
                            total: game.achievements.total
                          })
                        : game.lastPlayed
                          ? whenPlayed(game.lastPlayed)
                          : t("played.never")}
                    </span>
                  </span>
                  {game.achievements ? (
                    <span className="game-score inline-flex items-center gap-[0.25em]">
                      {game.achievements.score}
                      <span className="gamerscore-g" aria-label={t("allGames.gamerscore")}>
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
        ) : null
      }
    </SubBlade>
  );
}
