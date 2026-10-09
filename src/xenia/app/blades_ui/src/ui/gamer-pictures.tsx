import { useEffect, useRef } from "react";
import { send, useNative, type GamerPictures } from "../native";
import { PaneAction, Aside } from "../content/parts";
import { showToast } from "./toast";
import { t } from "../i18n";

/**
 * The Xbox 360 gamer pictures, offered as a one-time download (Profiles >
 * Gamer pictures): what it is and costs, the download's progress, and what
 * was installed. The app does the work (gamer_pictures_mac.mm).
 */

const megabytes = (bytes: number) => Math.round(bytes / 1_000_000);

export function startGamerPicturesDownload() {
  send({ op: "gamerPicturesDownload" });
  showToast(t("profiles.pictures"), t("profiles.pictures.started"), { sound: false });
}

/** The row's value: how many are installed, how far along, or not yet. */
export function GamerPicturesValue() {
  const { gamerPictures: pictures } = useNative();
  switch (pictures.status) {
    case "ready":
      return <>{t("profiles.pictures.value.count", { count: pictures.pictures })}</>;
    case "downloading":
      return <>{`${Math.floor((100 * pictures.received) / Math.max(pictures.total, 1))}%`}</>;
    case "installing":
      return <>{t("profiles.pictures.value.installing")}</>;
    default:
      return <>{t("profiles.pictures.value.missing")}</>;
  }
}

/** Toasts once a download finishes, wherever it was started. */
function useDoneNotice(pictures: GamerPictures) {
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

export function GamerPicturesPane() {
  const { gamerPictures: pictures } = useNative();
  useDoneNotice(pictures);
  const size = megabytes(pictures.total);
  const busy = pictures.status === "downloading" || pictures.status === "installing";
  const fraction = pictures.status === "installing" ? 1 : pictures.received / Math.max(pictures.total, 1);
  return (
    <>
      <p className="m-0">
        {t("profiles.pictures.detail", {
          collection: pictures.collection ?? "",
          site: pictures.site ?? ""
        })}
      </p>
      {pictures.status === "ready" ? (
        <Aside>{t("profiles.pictures.ready", { pictures: pictures.pictures, titles: pictures.titles })}</Aside>
      ) : busy ? (
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
              : t("profiles.pictures.downloading", { done: megabytes(pictures.received), size })}
          </Aside>
        </>
      ) : (
        <>
          <Aside>{t("profiles.pictures.size", { size })}</Aside>
          {pictures.status === "failed" ? (
            <Aside>{t("profiles.pictures.failed", { error: pictures.error })}</Aside>
          ) : null}
          <PaneAction onSelect={startGamerPicturesDownload}>
            {pictures.status === "failed" ? t("profiles.pictures.retry") : t("profiles.pictures.download")}
          </PaneAction>
        </>
      )}
    </>
  );
}
