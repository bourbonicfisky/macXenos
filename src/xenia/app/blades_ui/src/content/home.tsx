import type { BladeDef } from "../ui/blade";
import type { MenuItem } from "../ui/blade-menu";
import { AppIcon } from "../ui/app-icon";
import { send, type NativeState } from "../native";
import { Aside } from "./parts";

/* ==========================================================================
   Home: the dashboard you land on, as the console's own was. What is
   playing, the games you played last, and the way to start another. Rows
   here do something when chosen (A), rather than opening options.
   ========================================================================== */

function whenPlayed(seconds: number): string {
  if (!seconds) return "";
  const then = new Date(seconds * 1000);
  const days = Math.floor((Date.now() - then.getTime()) / 86_400_000);
  if (days <= 0) return "Played today";
  if (days === 1) return "Played yesterday";
  if (days < 7) return `Played ${days} days ago`;
  return `Played ${then.toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" })}`;
}

/** A game's own icon, as a lit tile where the app icon otherwise stands. */
function GameArt({ icon, name }: { icon: string; name: string }) {
  return (
    <div className="game-art relative grid aspect-square w-[min(17dvh,170px)] place-items-center overflow-hidden rounded-[14%]">
      <img className="h-full w-full object-cover" src={icon} alt={`${name} icon`} />
      <span className="game-art-gloss pointer-events-none absolute inset-0" aria-hidden="true" />
    </div>
  );
}

function fileName(path: string): string {
  return path.split("/").pop() ?? path;
}

/** "Guide on a controller, F6 or ⌘," — how to bring the dashboard back. */
export function dashboardShortcuts(state: NativeState): string {
  const ways: string[] = [];
  const button = String(state.cvars.preferences_button?.value ?? "guide");
  if (button === "guide") ways.push("a tap of Guide on a controller");
  else if (button === "back_start") ways.push("Back and Start together on a controller");
  const key = String(state.cvars.preferences_key?.value ?? "F6");
  if (key && key !== "none") ways.push(key);
  ways.push("⌘,");
  return ways.length > 1 ? `${ways.slice(0, -1).join(", ")} or ${ways[ways.length - 1]}` : ways[0];
}

export function home(state: NativeState): BladeDef {
  const menu: MenuItem[] = [];

  if (state.titleOpen) {
    menu.push(
      {
        id: "resume",
        section: "Now playing",
        label: `Resume ${state.titleName || "game"}`,
        glyph: "play",
        heading: state.titleName || "Now playing",
        detail: <p>Back to your game. It paused while the dashboard was open.</p>,
        onSelect: () => send({ op: "close" })
      },
      {
        id: "restart",
        label: "Restart game",
        glyph: "branch",
        heading: "Restart game",
        detail: (
          <>
            <p>Starts the game again from its first screen, keeping any settings you changed.</p>
            <Aside>Anything not saved in the game is lost.</Aside>
          </>
        ),
        onSelect: () => send({ op: "action", id: "relaunch" })
      },
      {
        id: "quit",
        label: "Quit game",
        glyph: "console",
        heading: "Quit game",
        detail: (
          <>
            <p>Stops the game and comes back here.</p>
            <Aside>Anything not saved in the game is lost.</Aside>
          </>
        ),
        onSelect: () => send({ op: "stopGame" })
      }
    );
  }

  state.recentTitles.slice(0, 8).forEach((title, index) => {
    menu.push({
      id: `recent-${index}`,
      section: index === 0 ? "Recent games" : undefined,
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
            {state.titleOpen
              ? "Choose to quit the game that is running and start this one."
              : "Choose to play."}
          </Aside>
        </>
      ),
      onSelect: () => send({ op: "launch", path: title.path })
    });
  });

  menu.push(
    {
      id: "open",
      section: state.recentTitles.length || state.titleOpen ? "More" : undefined,
      label: "Open a game",
      glyph: "folder",
      heading: "Open a game",
      detail: (
        <>
          <p>Pick a disc image (.iso), a game folder or a downloaded game to play.</p>
          <Aside>Games you play appear above for next time.</Aside>
        </>
      ),
      onSelect: () => send({ op: "openGame" })
    },
    {
      id: "install",
      label: "Install content",
      glyph: "layers",
      heading: "Install content",
      detail: (
        <p>
          Add title updates, downloadable content or Xbox Live Arcade games from their package
          files.
        </p>
      ),
      onSelect: () => send({ op: "installContent" })
    },
  );

  return {
    id: "home",
    label: "home",
    title: "macXenos",
    tone: "green",
    feature: (
      <div className="grid justify-items-center gap-[0.4em]">
        <AppIcon />
        <p className="menu-spec m-0 text-[0.85em]">macXenos {state.appVersion || "development build"}</p>
      </div>
    ),
    menu
  };
}
