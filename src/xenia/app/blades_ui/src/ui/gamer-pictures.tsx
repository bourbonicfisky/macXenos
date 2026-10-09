import { useEffect, useRef } from "react";
import { send, useNative, type GamerPictures } from "../native";
import { Aside } from "../content/parts";
import { showToast } from "./toast";
import { t } from "../i18n";

/**
 * The Xbox 360's official gamer pictures, offered in Change gamer picture as
 * a one-time download from the Internet Archive (gamer_pictures_mac.mm does
 * the work): starting it, its progress, and the explanation under the
 * picker's collections.
 */

const megabytes = (bytes: number) => Math.round(bytes / 1_000_000);

export function startGamerPicturesDownload() {
  send({ op: "gamerPicturesDownload" });
  showToast(t("profiles.pictures"), t("profiles.pictures.started"), { sound: false });
}

/** The More pictures row's second line: what A does, or how far along. */
export function morePicturesValue(pictures: GamerPictures): string {
  switch (pictures.status) {
    case "downloading":
      return t("profiles.picker.more.downloading", {
        percent: Math.floor((100 * pictures.received) / Math.max(pictures.total, 1))
      });
    case "installing":
      return t("profiles.picker.more.installing");
    case "failed":
      return t("profiles.picker.more.failed");
    default:
      return t("profiles.picker.more.detail", { size: megabytes(pictures.total) });
  }
}

/** Toasts once a download finishes, wherever the picker is. */
export function useDownloadNotice(pictures: GamerPictures) {
  const last = useRef(pictures.status);
  useEffect(() => {
    const was = last.current;
    last.current = pictures.status;
    if (was !== "downloading" && was !== "installing") return;
    if (pictures.status === "ready") showToast(t("profiles.pictures"), t("profiles.pictures.toast.ready"));
    else if (pictures.status === "failed") {
      showToast(t("profiles.pictures.toast.failed"), pictures.error, { sound: false });
    }
  }, [pictures.status, pictures.error]);
}

/** Under the picker's collections: where the official library comes from,
    and the download's progress while it runs. */
export function MorePicturesNote() {
  const { gamerPictures: pictures } = useNative();
  if (pictures.status === "unavailable") return null;
  const source = { collection: pictures.collection ?? "", site: pictures.site ?? "" };
  if (pictures.status === "ready") {
    return (
      <Aside>
        {t("profiles.pictures.installed", { ...source, pictures: pictures.pictures, titles: pictures.titles })}
      </Aside>
    );
  }
  const busy = pictures.status === "downloading" || pictures.status === "installing";
  const fraction = pictures.status === "installing" ? 1 : pictures.received / Math.max(pictures.total, 1);
  return (
    <div className="grid gap-[0.3em]">
      <Aside>{t("profiles.pictures.offer", { ...source, size: megabytes(pictures.total) })}</Aside>
      {busy ? (
        <>
          <div
            className="pref-progress rounded-row"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(fraction * 100)}
          >
            <span className="pref-progress-fill block h-full" style={{ width: `${Math.round(fraction * 100)}%` }} />
          </div>
          <Aside>
            {pictures.status === "installing"
              ? t("profiles.pictures.installing")
              : t("profiles.pictures.downloading", { done: megabytes(pictures.received), size: megabytes(pictures.total) })}
          </Aside>
        </>
      ) : null}
      {pictures.status === "failed" ? <Aside>{t("profiles.pictures.failed", { error: pictures.error })}</Aside> : null}
    </div>
  );
}
