import type { BladeDef } from "../ui/blade";
import { send, useNative, type NativeState } from "../native";
import {
  ConsoleCheck,
  ConsoleLockNote,
  ConsoleOptions,
  ConsoleReadout,
  ConsoleSlider,
  ConsoleValue
} from "../ui/console-controls";
import { Check, NextLaunchNote, Options, TextField, Value, type Option } from "../ui/settings-controls";
import { Aside, PaneAction } from "./parts";
import { showToast } from "../ui/toast";
import { dashboard } from "./dashboard";
import { t } from "../i18n";

const LOG_LEVELS: Option[] = [
  { value: 0, get label() { return t("system.errors"); }, get short() { return t("system.errors"); } },
  { value: 1, get label() { return t("system.warnings"); }, get short() { return t("system.warnings"); } },
  { value: 2, get label() { return t("system.information"); }, get short() { return t("system.info"); } },
  { value: 3, get label() { return t("system.everything"); }, get short() { return t("system.debug"); } }
];

/** A folder setting: where it is now, the picker, and back to the default. */
function FolderSetting({ name, title, fallback, pickerTitle }: {
  name: "install_root" | "save_root";
  title: string;
  fallback: string;
  pickerTitle: string;
}) {
  const cvar = useNative().cvars[name];
  if (!cvar) return null;
  const chosen = String(cvar.value);
  return (
    <>
      <p className="pref-group">{t("folder.folder")}</p>
      <p className="m-0 break-all">{chosen || fallback}</p>
      <div className="flex flex-wrap gap-[0.5em]">
        <PaneAction onSelect={() => send({ op: "pickFolder", name, title: pickerTitle })}>
          {t("folder.choose")}
        </PaneAction>
        <PaneAction
          disabled={!chosen}
          onSelect={() => {
            send({ op: "reset", name });
            showToast(t("toast.preferenceSaved"), t("folder.toastDefault", { name: title }));
          }}
        >
          {t("folder.useDefault")}
        </PaneAction>
      </div>
    </>
  );
}

const folderValue = (v: unknown) => (String(v) ? t("folder.custom") : t("folder.default"));

export function system(state: NativeState): BladeDef {
  // The dashboard's own settings first, then the console's.
  const dashboardRows = dashboard(state).menu.map((item, index) =>
    index === 0 ? { ...item, section: t("dashboard.section") } : item
  );
  return {
    id: "system",
    label: t("system.tab"),
    title: t("system.title"),
    tone: "purple",
    menu: [
      ...dashboardRows,
      {
        id: "language",
        section: t("system.title"),
        label: t("system.languageAndRegion"),
        glyph: "globe",
        value: <ConsoleValue id="language" />,
        heading: t("system.languageAndRegion"),
        detail: (
          <>
            <p>{t("system.language.detail")}</p>
            <p className="pref-group">{t("system.language")}</p>
            <ConsoleOptions id="language" />
            <p className="pref-group">{t("system.country")}</p>
            <ConsoleOptions id="country" />
            <ConsoleLockNote />
          </>
        )
      },
      {
        id: "clock",
        label: t("system.dateAndTime"),
        glyph: "clock",
        value: <ConsoleValue id="time_zone" />,
        heading: t("system.dateAndTime"),
        detail: (
          <>
            <p>{t("system.clock.detail")}</p>
            <ConsoleOptions id="time_zone" />
            <ConsoleCheck id="dst_off" />
            <ConsoleCheck id="clock_24h" />
            <ConsoleLockNote />
          </>
        )
      },
      {
        id: "audio-output",
        label: t("audio.output"),
        glyph: "sound",
        heading: t("audio.output"),
        detail: (
          <>
            <p>{t("system.audio.detail")}</p>
            <ConsoleCheck id="audio_mono" />
            <ConsoleCheck id="dolby_pro_logic" />
            <ConsoleCheck id="dolby_digital" />
            <ConsoleCheck id="dolby_digital_wma_pro" />
            <p className="pref-group">{t("system.musicPlayerVolume")}</p>
            <ConsoleSlider id="music_volume" />
            <ConsoleLockNote />
          </>
        )
      },
      {
        id: "family",
        label: t("system.familySettings"),
        glyph: "shield",
        value: <ConsoleValue id="parental_control" />,
        heading: t("system.familySettings"),
        detail: (
          <>
            <p>{t("system.family.detail")}</p>
            <ConsoleCheck id="parental_control" />
            <ConsoleLockNote />
          </>
        )
      },
      {
        id: "console",
        label: t("system.console"),
        glyph: "console",
        heading: t("system.console"),
        detail: (
          <>
            <p>{t("system.console.detail")}</p>
            <ConsoleCheck id="dashboard_initialized" />
            <ConsoleCheck id="iptv_initialized" />
            <ConsoleCheck id="dvr_initialized" />
            <ConsoleCheck id="kinect_initialized" />
            <ConsoleReadout id="mac_address" />
            <ConsoleReadout id="network_id" />
            <PaneAction
              onSelect={() => {
                send({ op: "consoleReset" });
                showToast(t("console.settings"), t("settings.defaults"));
              }}
            >
              {t("system.resetConsoleSettings")}
            </PaneAction>
            <ConsoleLockNote />
          </>
        )
      },
      {
        id: "storage",
        label: t("advanced.storage"),
        glyph: "folder",
        heading: t("advanced.storage"),
        detail: (
          <>
            <p>{t("system.storage.detail")}</p>
            <p className="pref-group">{t("system.savesAndContent")}</p>
            <TextField name="content_root" title={t("system.contentFolder")} placeholder={t("folder.default")} />
            <p className="pref-group">{t("system.caches")}</p>
            <TextField name="cache_root" title={t("system.cacheFolder")} placeholder={t("folder.default")} />
            <p className="pref-group">{t("system.settingsAndProfiles")}</p>
            <TextField name="storage_root" title={t("system.storageFolder")} placeholder={t("folder.default")} />
            <Check name="mount_cache" label={t("system.mountCache")} />
            <Check name="mount_scratch" label={t("system.mountScratch")} />
            <Check name="mount_memory_unit" label={t("system.mountMemoryUnit")} />
            <NextLaunchNote names={["content_root", "cache_root", "storage_root", "mount_cache", "mount_scratch", "mount_memory_unit"]} />
          </>
        )
      },
      {
        id: "saves",
        label: t("saves.label"),
        glyph: "people",
        value: <Value name="save_root" format={folderValue} />,
        heading: t("saves.label"),
        detail: (
          <>
            <p>{t("saves.detail")}</p>
            <FolderSetting
              name="save_root"
              title={t("saves.label")}
              fallback={t("saves.default")}
              pickerTitle={t("saves.pickerTitle")}
            />
            <Aside>{t("saves.note")}</Aside>
          </>
        )
      },
      {
        id: "installs",
        label: t("installs.label"),
        glyph: "disc",
        value: <Value name="install_root" format={folderValue} />,
        heading: t("installs.label"),
        detail: (
          <>
            <p>{t("installs.detail")}</p>
            <FolderSetting
              name="install_root"
              title={t("installs.label")}
              fallback={t("installs.default")}
              pickerTitle={t("installs.pickerTitle")}
            />
            <Aside>{t("installs.note")}</Aside>
          </>
        )
      },
      {
        id: "updates",
        label: t("system.titleUpdates"),
        glyph: "branch",
        value: <Value name="apply_title_update" />,
        heading: t("system.titleUpdates"),
        detail: (
          <>
            <p>{t("system.updates.detail")}</p>
            <Check name="apply_title_update" label={t("system.applyTitleUpdates")} />
            <Check name="allow_incompatible_title_update" label={t("system.allowOtherVersionUpdates")} />
            <NextLaunchNote names={["apply_title_update", "allow_incompatible_title_update"]} />
          </>
        )
      },
      {
        id: "discord",
        label: "Discord",
        glyph: "people",
        value: <Value name="discord" />,
        heading: "Discord",
        detail: (
          <>
            <p>{t("system.discord.detail")}</p>
            <Check name="discord" label={t("system.discord.share")} />
            <NextLaunchNote names={["discord"]} />
          </>
        )
      },
      {
        id: "logging",
        label: t("advanced.logging"),
        glyph: "doc",
        value: <Value name="log_level" options={LOG_LEVELS} />,
        heading: t("advanced.logging"),
        detail: (
          <>
            <p>{t("system.logging.detail")}</p>
            <Options name="log_level" title={t("advanced.logging")} options={LOG_LEVELS} />
            <Aside>{t("system.logging.note")}</Aside>
          </>
        )
      }
    ]
  };
}
