import { useEffect, useState } from "react";
import { send, useNative } from "../native";
import { openSubBlade, SubBlade } from "./sub-blade";
import { showToast } from "./toast";

/**
 * Game Folders: where All Games looks, edited on a brushed-metal sub-blade
 * from the right, laid out like the console's Profile sub-blade: a plain
 * menu on the left and a card on the right about the highlighted row. The
 * first row adds a folder (the Mac's folder picker); every other row is a
 * folder, which A removes on a second press, so one stray press can't. Its
 * games leave All Games; nothing on disk is touched.
 */

export const openFolderEditor = () => openSubBlade("game-folders");

const lastPart = (path: string) => path.replace(/\/+$/, "").split("/").pop() || path;

export function FolderEditor() {
  const { gameFolders, allGames } = useNative();
  const folders = gameFolders.folders;
  // The folder waiting for its second A.
  const [armed, setArmed] = useState<string | null>(null);

  // A removal that isn't confirmed lapses.
  useEffect(() => {
    if (!armed) return;
    const timer = window.setTimeout(() => setArmed(null), 5000);
    return () => window.clearTimeout(timer);
  }, [armed]);

  const gamesIn = (folder: string) => {
    const prefix = folder.replace(/\/+$/, "") + "/";
    return allGames.filter((game) => game.path.startsWith(prefix)).length;
  };

  const activate = (index: number) => {
    if (index === 0) {
      setArmed(null);
      send({ op: "gameFolderAdd" });
      return;
    }
    const folder = folders[index - 1];
    if (!folder) return;
    if (armed !== folder) {
      setArmed(folder);
      return;
    }
    setArmed(null);
    send({ op: "gameFolderRemove", path: folder });
    showToast("Game folder removed", lastPart(folder));
  };

  return (
    <SubBlade
      id="game-folders"
      side="right"
      tone="metal"
      title="Game Folders"
      count={folders.length + 1}
      onActivate={activate}
      onFocusChange={(index) => {
        if (armed && folders[index - 1] !== armed) setArmed(null);
      }}
      legend={(focus) => ({
        left: [],
        right: [
          { button: "b", label: "Back" },
          {
            button: "a",
            label: focus === 0 ? "Add" : armed === folders[focus - 1] ? "Confirm" : "Remove"
          }
        ]
      })}
    >
      {(rowProps, focus) => {
        const folder = focus > 0 ? folders[focus - 1] : undefined;
        const count = folder ? gamesIn(folder) : 0;
        return (
          <div className="grid grid-cols-1 items-start gap-x-[clamp(16px,2.6%,40px)] gap-y-[1em] min-[860px]:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <ul className="subblade-list subblade-menu m-0 grid list-none p-0" aria-label="Game folders">
              <li className="subblade-item">
                <button type="button" {...rowProps(0)}>
                  <span className="subblade-label truncate">Add a Folder</span>
                </button>
              </li>
              {folders.length ? (
                <li className="menu-section subblade-section" aria-hidden="true">
                  Folders
                </li>
              ) : null}
              {folders.map((path, index) => (
                <li key={path} className="subblade-item">
                  <button type="button" {...rowProps(index + 1)} aria-label={`${path}, remove`}>
                    <span className="subblade-label truncate">{lastPart(path)}</span>
                    {armed === path ? (
                      <span className="subblade-action ml-auto text-[0.7em]" data-armed>
                        Remove?
                      </span>
                    ) : null}
                  </button>
                </li>
              ))}
            </ul>

            <div className="grid gap-[0.7em] text-[0.82em]">
              {folder ? (
                <>
                  <section className="subblade-card">
                    <h3 className="subblade-card-head m-0">{lastPart(folder)}</h3>
                    <dl className="subblade-card-body m-0 grid grid-cols-[auto_1fr] gap-x-[1em] gap-y-[0.25em]">
                      <dt>Games</dt>
                      <dd className="m-0 text-right">{gameFolders.scanning ? "Looking" : count}</dd>
                      <dt>Includes</dt>
                      <dd className="m-0 text-right">Folders inside it</dd>
                    </dl>
                  </section>
                  <section className="subblade-card">
                    <p className="subblade-card-body m-0 [overflow-wrap:anywhere]">{folder}</p>
                  </section>
                  <section className="subblade-card subblade-card--tall">
                    <p className="subblade-card-body m-0">
                      {armed === folder
                        ? "Press A again to take this folder out of All Games. Nothing on your Mac is deleted."
                        : "A takes this folder out of All Games. Its games stay on your Mac, and you can add it again."}
                    </p>
                  </section>
                </>
              ) : (
                <>
                  <section className="subblade-card">
                    <h3 className="subblade-card-head m-0">Add a Folder</h3>
                    <p className="subblade-card-body m-0">
                      Pick a folder that holds disc images, unpacked games or downloaded games. Folders inside it are
                      searched too.
                    </p>
                  </section>
                  <section className="subblade-card subblade-card--tall">
                    <p className="subblade-card-body m-0">
                      {folders.length
                        ? `${allGames.length} ${allGames.length === 1 ? "game" : "games"} in All Games from ${folders.length} ${folders.length === 1 ? "folder" : "folders"}.`
                        : "No folders yet. Games you have played still appear in All Games."}
                    </p>
                  </section>
                </>
              )}
            </div>
          </div>
        );
      }}
    </SubBlade>
  );
}
