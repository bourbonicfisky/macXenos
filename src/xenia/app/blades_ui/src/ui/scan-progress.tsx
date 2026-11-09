import { useNative } from "../native";
import { Aside } from "../content/parts";
import { t } from "../i18n";

/** How far a look through the game folders has got (game_library.h): while
    it lists the folders, how many games it has found; then a bar while it
    reads each one. Shown only while a scan runs. */
export function ScanProgress() {
  const { gameFolders } = useNative();
  const { scanning, found, checked, total } = gameFolders;
  if (!scanning) return null;
  if (!total) return <Aside>{t("gameFolders.scan.listing", { found })}</Aside>;
  const percent = Math.round((Math.min(checked, total) / total) * 100);
  return (
    <div className="grid gap-[0.3em]">
      <div className="pref-progress rounded-row" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
        <span className="pref-progress-fill block h-full" style={{ width: `${percent}%` }} />
      </div>
      <Aside>{t("gameFolders.scan.checking", { checked: Math.min(checked, total), total })}</Aside>
    </div>
  );
}
