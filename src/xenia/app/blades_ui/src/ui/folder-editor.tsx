import { useEffect, useState } from "react";
import { send, useNative } from "../native";
import type { BladeTone } from "./blade";
import { Glyph } from "./glyph";
import { openSubBlade, SubBlade } from "./sub-blade";
import { showToast } from "./toast";

/**
 * Game Folders: where All Games looks, edited on a sub-blade from the right.
 * The first row adds a folder (the Mac's folder picker); every other row is a
 * folder, which A removes on a second press, so one stray press can't. The
 * games in it leave All Games; nothing on disk is touched.
 */

export const openFolderEditor = () => openSubBlade("game-folders");

const lastPart = (path: string) => path.replace(/\/+$/, "").split("/").pop() || path;

export function FolderEditor({ tone }: { tone: BladeTone }) {
  const { gameFolders } = useNative();
  const folders = gameFolders.folders;
  // The folder waiting for its second A.
  const [armed, setArmed] = useState<string | null>(null);

  // A removal that isn't confirmed lapses.
  useEffect(() => {
    if (!armed) return;
    const timer = window.setTimeout(() => setArmed(null), 5000);
    return () => window.clearTimeout(timer);
  }, [armed]);

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
      tone={tone}
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
      {(rowProps) => (
        <ul className="subblade-list m-0 grid list-none p-0" aria-label="Game folders">
          <li className="subblade-item">
            <button type="button" {...rowProps(0)}>
              <Glyph name="folder" />
              <span className="grid min-w-0">
                <span className="subblade-name truncate">Add a folder</span>
                <span className="subblade-sub truncate">
                  Disc images, unpacked games and downloaded games, in it or in folders inside it
                </span>
              </span>
              <span />
            </button>
          </li>
          {folders.map((folder, index) => (
            <li key={folder} className="subblade-item">
              <button type="button" {...rowProps(index + 1)} aria-label={`${folder}, remove`}>
                <Glyph name="folder" />
                <span className="grid min-w-0">
                  <span className="subblade-name truncate">{lastPart(folder)}</span>
                  <span className="subblade-sub truncate">{folder}</span>
                </span>
                <span className="subblade-action text-[0.8em]" data-armed={armed === folder || undefined}>
                  {armed === folder ? "Press A again to remove" : "Remove"}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </SubBlade>
  );
}
