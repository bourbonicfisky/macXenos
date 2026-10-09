import type { BladeDef } from "../ui/blade";
import type { MenuItem } from "../ui/blade-menu";
import { AppIcon } from "../ui/app-icon";
import { openAllGames } from "../ui/all-games";
import { openFolderEditor } from "../ui/folder-editor";
import { send, useNative, type NativeState } from "../native";
import { Aside, PaneAction } from "./parts";
import { getLanguage, t } from "../i18n";

/* ==========================================================================
   Home: the dashboard you land on, as the console's own was. What is
   playing, the games you played last, and the way to start another. Rows
   here do something when chosen (A), rather than opening options.
   ========================================================================== */

export function whenPlayed(seconds: number): string {
  if (!seconds) return "";
  const then = new Date(seconds * 1000);
  const days = Math.floor((Date.now() - then.getTime()) / 86_400_000);
  if (days <= 0) return t("played.today");
  if (days === 1) return t("played.yesterday");
  if (days < 7) return t("played.daysAgo", { days });
  const language = getLanguage();
  const date = then.toLocaleDateString(language === "en" || language === "en-XA" ? undefined : language, {
    month: "long",
    day: "numeric",
    year: "numeric"
  });
  return t("played.on", { date });
}

/** A game's own icon, as a lit tile where the app icon otherwise stands. */
function GameArt({ icon, name }: { icon: string; name: string }) {
  return (
    <div className="game-art relative grid aspect-square w-[min(17dvh,170px)] place-items-center overflow-hidden rounded-[14%]">
      <img className="h-full w-full object-cover" src={icon} alt={t("home.gameIcon", { name })} />
      <span className="game-art-gloss pointer-events-none absolute inset-0" aria-hidden="true" />
    </div>
  );
}

/** "2 folders", or that there are none yet. */
function GameFoldersValue() {
  const { gameFolders } = useNative();
  const count = gameFolders.folders.length;
  if (gameFolders.scanning) return <>{t("home.gameFolders.scanning")}</>;
  return <>{count === 0 ? t("home.gameFolders.none") : t("home.gameFolders.count", { count })}</>;
}

/** The folders All Games looks in, and the two things to do with them:
    change the list (on its own sub-blade) or look through them again. */
function GameFoldersPane() {
  const { gameFolders, allGames } = useNative();
  const { folders, scanning } = gameFolders;
  return (
    <>
      <p className="m-0">{t("home.gameFolders.detail")}</p>
      {folders.length ? (
        <ul className="game-folder-list m-0 grid list-none gap-[0.2em] p-0">
          {folders.map((folder) => (
            <li key={folder} className="menu-spec text-[0.8em] [overflow-wrap:anywhere]">
              {folder}
            </li>
          ))}
        </ul>
      ) : (
        <Aside>{t("home.gameFolders.empty")}</Aside>
      )}
      {/* Side by side: → steps from one to the other, as ↓ does. */}
      <div className="flex flex-wrap gap-x-[0.6em]">
        <PaneAction onSelect={openFolderEditor}>{t("home.gameFolders.edit")}</PaneAction>
        <PaneAction disabled={!folders.length || scanning} onSelect={() => send({ op: "gameFoldersRescan" })}>
          {t("home.gameFolders.refresh")}
        </PaneAction>
      </div>
      {folders.length ? (
        <Aside>
          {scanning ? t("home.gameFolders.looking") : t("home.gameFolders.games", { count: allGames.length })}
        </Aside>
      ) : null}
    </>
  );
}

function fileName(path: string): string {
  return path.split("/").pop() ?? path;
}

/** "Guide on a controller, F6 or ⌘," — how to bring the dashboard back. */
export function dashboardShortcuts(state: NativeState): string {
  const ways: string[] = [];
  const button = String(state.cvars.preferences_button?.value ?? "guide");
  if (button === "guide") ways.push(t("shortcuts.guide"));
  else if (button === "back_start") ways.push(t("shortcuts.backStart"));
  const key = String(state.cvars.preferences_key?.value ?? "F6");
  if (key && key !== "none") ways.push(key);
  ways.push("⌘,");
  return ways.length > 1
    ? t("shortcuts.list", { first: ways.slice(0, -1).join(", "), last: ways[ways.length - 1] })
    : ways[0];
}

export function home(state: NativeState): BladeDef {
  const menu: MenuItem[] = [];

  if (state.titleOpen) {
    menu.push(
      {
        id: "resume",
        section: t("home.nowPlaying"),
        label: state.titleName ? t("home.resume", { title: state.titleName }) : t("home.resume.untitled"),
        glyph: "play",
        heading: state.titleName || t("home.nowPlaying"),
        detail: <p>{t("home.resume.detail")}</p>,
        onSelect: () => send({ op: "close" })
      },
      {
        id: "restart",
        label: t("home.restart"),
        glyph: "branch",
        heading: t("home.restart"),
        detail: (
          <>
            <p>{t("home.restart.detail")}</p>
            <Aside>{t("home.unsavedLost")}</Aside>
          </>
        ),
        onSelect: () => send({ op: "action", id: "relaunch" })
      },
      {
        id: "quit",
        label: t("home.quit"),
        glyph: "console",
        heading: t("home.quit"),
        detail: (
          <>
            <p>{t("home.quit.detail")}</p>
            <Aside>{t("home.unsavedLost")}</Aside>
          </>
        ),
        onSelect: () => send({ op: "stopGame" })
      }
    );
  }

  state.recentTitles.slice(0, 8).forEach((title, index) => {
    menu.push({
      id: `recent-${index}`,
      section: index === 0 ? t("home.recentGames") : undefined,
      label: title.name || fileName(title.path),
      glyph: "disc",
      image: title.icon || undefined,
      feature: title.icon ? <GameArt icon={title.icon} name={title.name} /> : undefined,
      heading: title.name || fileName(title.path),
      detail: (
        <>
          <p>{whenPlayed(title.lastPlayed)}</p>
          <p className="menu-spec m-0 text-[0.78em] [overflow-wrap:anywhere]">{title.path}</p>
          <Aside>
            {state.titleOpen ? t("home.recent.switch") : t("home.recent.play")}
          </Aside>
        </>
      ),
      onSelect: () => send({ op: "launch", path: title.path })
    });
  });

  menu.push(
    {
      id: "all-games",
      section: t("home.more"),
      label: t("home.allGames"),
      glyph: "blocks",
      value: state.allGames.length ? <>{state.allGames.length}</> : undefined,
      heading: t("home.allGames"),
      detail: (
        <>
          <p>{t("home.allGames.detail")}</p>
          {state.gameFolders.folders.length ? (
            <Aside>{t("home.allGames.sortHint")}</Aside>
          ) : (
            <Aside>{t("home.allGames.addHint")}</Aside>
          )}
        </>
      ),
      onSelect: openAllGames
    }
  );

  menu.push(
    {
      id: "open",
      label: t("home.open"),
      glyph: "folder",
      heading: t("home.open"),
      detail: (
        <>
          <p>{t("home.open.detail")}</p>
          <Aside>{t("home.open.hint")}</Aside>
        </>
      ),
      onSelect: () => send({ op: "openGame" })
    },
    {
      id: "game-folders",
      label: t("home.gameFolders"),
      glyph: "folder",
      value: <GameFoldersValue />,
      heading: t("home.gameFolders"),
      detail: <GameFoldersPane />
    },
    {
      id: "install",
      label: t("home.install"),
      glyph: "layers",
      heading: t("home.install"),
      detail: <p>{t("home.install.detail")}</p>,
      onSelect: () => send({ op: "installContent" })
    },
  );

  return {
    id: "home",
    label: t("home.tab"),
    title: "macXenos",
    tone: "green",
    feature: (
      <div className="grid justify-items-center gap-[0.4em]">
        <AppIcon />
        <p className="menu-spec m-0 text-[0.85em]">
          {t("home.version", { version: state.appVersion || t("home.developmentBuild") })}
        </p>
      </div>
    ),
    menu
  };
}
