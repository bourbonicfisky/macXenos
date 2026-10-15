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
  { value: 0, label: "Errors", short: "Errors" },
  { value: 1, label: "Warnings", short: "Warnings" },
  { value: 2, label: "Information", short: "Info" },
  { value: 3, label: "Everything", short: "Debug" }
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
    label: "system",
    title: "System",
    tone: "purple",
    menu: [
      ...dashboardRows,
      {
        id: "language",
        section: "System",
        label: "Language and region",
        glyph: "globe",
        value: <ConsoleValue id="language" />,
        heading: "Language and region",
        detail: (
          <>
            <p>The language and country the console reports. Games pick their text and ratings from these.</p>
            <p className="pref-group">Language</p>
            <ConsoleOptions id="language" />
            <p className="pref-group">Country</p>
            <ConsoleOptions id="country" />
            <ConsoleLockNote />
          </>
        )
      },
      {
        id: "clock",
        label: "Date and time",
        glyph: "clock",
        value: <ConsoleValue id="time_zone" />,
        heading: "Date and time",
        detail: (
          <>
            <p>The console&rsquo;s time zone. The date and time come from your Mac.</p>
            <ConsoleOptions id="time_zone" />
            <ConsoleCheck id="dst_off" />
            <ConsoleCheck id="clock_24h" />
            <ConsoleLockNote />
          </>
        )
      },
      {
        id: "audio-output",
        label: "Audio output",
        glyph: "sound",
        heading: "Audio output",
        detail: (
          <>
            <p>The sound formats the console reports, and the music player&rsquo;s volume.</p>
            <ConsoleCheck id="audio_mono" />
            <ConsoleCheck id="dolby_pro_logic" />
            <ConsoleCheck id="dolby_digital" />
            <ConsoleCheck id="dolby_digital_wma_pro" />
            <p className="pref-group">Music player volume</p>
            <ConsoleSlider id="music_volume" />
            <ConsoleLockNote />
          </>
        )
      },
      {
        id: "family",
        label: "Family settings",
        glyph: "shield",
        value: <ConsoleValue id="parental_control" />,
        heading: "Family settings",
        detail: (
          <>
            <p>Turns on the console&rsquo;s parental controls, which games check before showing rated content.</p>
            <ConsoleCheck id="parental_control" />
            <ConsoleLockNote />
          </>
        )
      },
      {
        id: "console",
        label: "Console",
        glyph: "console",
        heading: "Console",
        detail: (
          <>
            <p>
              Setup steps the console records as done, and its network identity.
              Some games read these.
            </p>
            <ConsoleCheck id="dashboard_initialized" />
            <ConsoleCheck id="iptv_initialized" />
            <ConsoleCheck id="dvr_initialized" />
            <ConsoleCheck id="kinect_initialized" />
            <ConsoleReadout id="mac_address" />
            <ConsoleReadout id="network_id" />
            <PaneAction
              onSelect={() => {
                send({ op: "consoleReset" });
                showToast("Console settings", "Back to defaults");
              }}
            >
              Reset console settings
            </PaneAction>
            <ConsoleLockNote />
          </>
        )
      },
      {
        id: "storage",
        label: "Storage",
        glyph: "folder",
        heading: "Storage",
        detail: (
          <>
            <p>
              Where saves, downloaded content and caches live. Empty means the
              default, in your user folder.
            </p>
            <p className="pref-group">Saves and content</p>
            <TextField name="content_root" title="Content folder" placeholder="Default" />
            <p className="pref-group">Caches</p>
            <TextField name="cache_root" title="Cache folder" placeholder="Default" />
            <p className="pref-group">Settings and profiles</p>
            <TextField name="storage_root" title="Storage folder" placeholder="Default" />
            <Check name="mount_cache" label="Mount the console's cache drive" />
            <Check name="mount_scratch" label="Mount the scratch drive" />
            <Check name="mount_memory_unit" label="Mount a memory unit" />
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
        label: "Title updates",
        glyph: "branch",
        value: <Value name="apply_title_update" />,
        heading: "Title updates",
        detail: (
          <>
            <p>Apply the updates you have installed for a game when it starts.</p>
            <Check name="apply_title_update" label="Apply title updates" />
            <Check name="allow_incompatible_title_update" label="Allow updates made for another version" />
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
            <p>
              Show the game you are playing on your Discord profile. Needs the
              Discord app open on this Mac; it appears there as Xenia Canary.
            </p>
            <Check name="discord" label="Share what I'm playing" />
            <NextLaunchNote names={["discord"]} />
          </>
        )
      },
      {
        id: "logging",
        label: "Logging",
        glyph: "doc",
        value: <Value name="log_level" options={LOG_LEVELS} />,
        heading: "Logging",
        detail: (
          <>
            <p>
              How much the log records. Information is enough for a bug report;
              Everything slows games down.
            </p>
            <Options name="log_level" title="Logging" options={LOG_LEVELS} />
            <Aside>The log is xenia.log in your user folder.</Aside>
          </>
        )
      }
    ]
  };
}
