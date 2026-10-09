import { useEffect, useMemo, useState } from "react";
import { send, useNative, type Profile } from "../native";
import { DEFAULT_PICTURES, defaultPictureUrl } from "../default-pictures";
import type { BladeTone } from "./blade";
import { Gamercard } from "./gamercard";
import { startGamerPicturesDownload } from "./gamer-pictures";
import { closeSubBlade, openSubBlade, SubBlade, useSubBladeOpen } from "./sub-blade";
import { showToast } from "./toast";
import { t } from "../i18n";

/**
 * Choosing a profile's gamer picture, on a sub-blade from the right, in two
 * levels as the console's picker was: first the collections (macXenos's
 * own pictures, then each game's from the downloaded Xbox 360 collection,
 * or a row to download it), then a grid of the chosen collection's
 * pictures. A sets the picture (the app makes its small size), B goes back
 * a level. The profile's gamercard beside the grid previews the picture
 * under the cursor.
 */

type Collection = { id: string; name: string; pictures: string[] };

let target = "";
export function openPicturePicker(xuid: string) {
  target = xuid;
  openSubBlade("gamer-picture");
}

const DOWNLOADED_ROOT = "/content/0000000000000000/FFFE07D1/00020000";

/** The downloaded collection's games, once installed (gamer_pictures.json). */
function useDownloaded(ready: boolean): Collection[] {
  const [collections, setCollections] = useState<Collection[]>([]);
  useEffect(() => {
    if (!ready) {
      setCollections([]);
      return;
    }
    let live = true;
    fetch("/data/gamer_pictures.json")
      .then((response) => (response.ok ? response.json() : []))
      .then((titles: { id: string; name: string; pictures: string[] }[]) => {
        if (!live) return;
        setCollections(
          titles
            .map((title) => ({
              id: title.id,
              name: title.name,
              pictures: title.pictures.map((key) => `${DOWNLOADED_ROOT}/${title.id}/64_${key}.png`)
            }))
            .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }))
        );
      })
      .catch(() => live && setCollections([]));
    return () => {
      live = false;
    };
  }, [ready]);
  return collections;
}

export function PicturePicker({ tone }: { tone: BladeTone }) {
  const { profiles, gamerPictures } = useNative();
  // Drawn again when it opens, which is when `target` has just been set.
  const isOpen = useSubBladeOpen("gamer-picture");
  const profile: Profile | undefined = isOpen || target ? profiles.find((p) => p.xuid === target) : undefined;
  const downloaded = useDownloaded(gamerPictures.status === "ready");
  const collections = useMemo<Collection[]>(
    () => [
      { id: "macxenos", name: t("profiles.picker.macxenos"), pictures: DEFAULT_PICTURES.map(defaultPictureUrl) },
      ...downloaded
    ],
    [downloaded]
  );
  // The collection open in the grid, or null for the list of collections.
  const [open, setOpen] = useState<number | null>(null);
  const [lastCollection, setLastCollection] = useState(0);
  const [focus, setFocus] = useState(0);

  // A fresh picker starts at the collections.
  useEffect(() => {
    if (isOpen) {
      setOpen(null);
      setLastCollection(0);
    }
  }, [isOpen]);

  const offerMore = gamerPictures.status !== "ready" && gamerPictures.status !== "unavailable";
  const rows = collections.length + (offerMore ? 1 : 0);
  const grid = open !== null ? collections[open] : null;

  const activate = (index: number) => {
    if (!grid) {
      if (index < collections.length) {
        setLastCollection(index);
        setOpen(index);
      } else if (gamerPictures.status === "missing" || gamerPictures.status === "failed") {
        startGamerPicturesDownload();
      }
      return;
    }
    if (!profile) return;
    const picture = grid.pictures[index];
    send({ op: "profilePicture", xuid: profile.xuid, picture });
    showToast(t("profiles.picker.changed"), profile.gamertag);
    closeSubBlade();
    setOpen(null);
  };

  const moreValue =
    gamerPictures.status === "downloading"
      ? t("profiles.picker.more.downloading", {
          percent: Math.floor((100 * gamerPictures.received) / Math.max(gamerPictures.total, 1))
        })
      : gamerPictures.status === "installing"
        ? t("profiles.picker.more.installing")
        : t("profiles.picker.more.detail", { size: Math.round(gamerPictures.total / 1_000_000) });

  return (
    <SubBlade
      id="gamer-picture"
      side="right"
      tone={tone}
      title={grid ? grid.name : t("profiles.picker")}
      count={grid ? grid.pictures.length : rows}
      grid={grid !== null}
      resetKey={open}
      resetFocus={open === null ? lastCollection : Math.max(0, grid?.pictures.indexOf(profile?.picture ?? "") ?? 0)}
      onFocusChange={setFocus}
      onActivate={activate}
      onBack={() => {
        if (open === null) return false;
        setOpen(null);
        return true;
      }}
      legend={() => ({
        left: [],
        right: [
          { button: "b", label: t("legend.back") },
          { button: "a", label: grid ? t("profiles.picker.choose") : t("legend.select") }
        ]
      })}
    >
      {(rowProps, current) => (
        <div className="grid grid-cols-1 items-start gap-x-[clamp(16px,2.6%,40px)] gap-y-[1em] min-[860px]:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
          {grid ? (
            <ul className="picture-grid m-0 grid list-none p-0" aria-label={grid.name}>
              {grid.pictures.map((picture, index) => (
                <li key={picture}>
                  <button
                    type="button"
                    {...rowProps(index)}
                    aria-label={t("profiles.picker.item", { index: index + 1, count: grid.pictures.length })}
                    aria-current={picture === profile?.picture || undefined}
                  >
                    <img className="picture-tile-image" src={picture} alt="" loading="lazy" width={64} height={64} />
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <ul className="subblade-list m-0 grid list-none p-0" aria-label={t("profiles.picker")}>
              {collections.map((collection, index) => (
                <li key={collection.id} className="subblade-item">
                  <button type="button" {...rowProps(index)}>
                    <img className="game-icon" src={collection.pictures[0]} alt="" aria-hidden="true" />
                    <span className="grid min-w-0">
                      <span className="subblade-name truncate">{collection.name}</span>
                      <span className="subblade-sub truncate">
                        {t("profiles.picker.count", { count: collection.pictures.length })}
                      </span>
                    </span>
                    <span />
                  </button>
                </li>
              ))}
              {offerMore ? (
                <li className="subblade-item">
                  <button type="button" {...rowProps(collections.length)}>
                    <span className="game-icon picture-more" aria-hidden="true">
                      <span>+</span>
                    </span>
                    <span className="grid min-w-0">
                      <span className="subblade-name truncate">{t("profiles.picker.more")}</span>
                      <span className="subblade-sub truncate">{moreValue}</span>
                    </span>
                    <span />
                  </button>
                </li>
              ) : null}
            </ul>
          )}
          {/* Pinned while the grid scrolls, at a size its column fits. */}
          {profile ? (
            <div className="picker-card sticky top-0 text-[0.8em]">
              <Gamercard
                profile={profile}
                picture={grid ? grid.pictures[Math.min(current, grid.pictures.length - 1)] : undefined}
              />
            </div>
          ) : null}
        </div>
      )}
    </SubBlade>
  );
}
