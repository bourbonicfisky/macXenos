import type { BladeDef } from "../ui/blade";
import { Bullets } from "../ui/blade-menu";
import { useNative, type NativeState } from "../native";
import {
  PreferenceOptions,
  PreferenceValue,
  ReducedMotionOverride,
  SoundValue
} from "../ui/preference-controls";
import { Aside, PaneAction } from "./parts";
import { downloadDashSounds, packCredit } from "../ui/sound-offer";
import { Options, Value } from "../ui/settings-controls";
import { dashboardShortcuts } from "./home";
import { AVAILABLE, LOCALES, PSEUDO, t, useLanguage } from "../i18n";
import { getPreferences, usePreferences } from "../ui/preferences";

/** The official website, on About macXenos. */
const WEBSITE = "https://macxenos.com";

/** Where the sounds come from, and the way to get them if they aren't here. */
function SoundSource() {
  const { dashSounds } = useNative();
  const credit = packCredit(dashSounds);
  switch (dashSounds.status) {
    case "ready":
      return <Aside>{t("dashboard.sound.downloaded", { credit })}</Aside>;
    case "downloading":
      return <Aside>{t("dashboard.sound.downloadingFrom", { credit })}</Aside>;
    case "missing":
    case "failed":
      return (
        <>
          <p className="m-0">{t("dashboard.sound.missing", { credit })}</p>
          {dashSounds.status === "failed" ? (
            <Aside>
              {dashSounds.error
                ? t("dashboard.sound.failedBecause", { error: dashSounds.error })
                : t("dashboard.sound.failed")}
            </Aside>
          ) : null}
          <PaneAction onSelect={downloadDashSounds}>
            {dashSounds.status === "failed" ? t("dashboard.sound.retry") : t("dashboard.sound.download")}
          </PaneAction>
        </>
      );
    default:
      return null;
  }
}

/** The Sound pane. Until the sounds are downloaded, getting them comes first;
    after that, the switch and volume, with the pack credited underneath. */
function SoundDetail() {
  const { dashSounds } = useNative();
  const needed = dashSounds.status === "missing" || dashSounds.status === "failed";
  const settings = (
    <>
      {needed ? <p className="pref-group m-0">{t("dashboard.sound.onceDownloaded")}</p> : null}
      <PreferenceOptions pref="sound" />
      <p className="pref-group">{t("pref.volume")}</p>
      <PreferenceOptions pref="volume" />
    </>
  );
  if (needed) {
    return (
      <>
        <SoundSource />
        {settings}
      </>
    );
  }
  return (
    <>
      <p>{t("dashboard.sound.detail")}</p>
      {settings}
      <SoundSource />
    </>
  );
}

/** The Animation pane's Background group: the choice, and why it isn't
    moving when motion is reduced. */
function BackgroundDetail() {
  usePreferences();
  const held = document.documentElement.dataset.background === "still" && getPreferences().background === "shifting";
  return (
    <>
      <p className="pref-group m-0">{t("pref.background")}</p>
      <p className="m-0">{t("dashboard.background.detail")}</p>
      <PreferenceOptions pref="background" />
      {held ? <Aside>{t("dashboard.background.reduced")}</Aside> : null}
    </>
  );
}

/** The language the dashboard is in now, in its own words. */
function LanguageValue() {
  const language = useLanguage();
  return <>{language === PSEUDO ? "Pseudo" : LOCALES.find((locale) => locale.code === language)?.name}</>;
}

/** The Sound row's value: what's playing, or why nothing is. */
function SoundRowValue() {
  const { dashSounds } = useNative();
  if (dashSounds.status === "missing" || dashSounds.status === "failed") {
    return <>{t("dashboard.sound.notDownloaded")}</>;
  }
  if (dashSounds.status === "downloading") return <>{t("dashboard.sound.downloading")}</>;
  return <SoundValue />;
}

/** The blades' own look and sound, as on the site's Preferences blade. They
    lead the System blade (system.tsx) under a "Dashboard" heading. */
export function dashboard(state: NativeState): BladeDef {
  return {
    id: "dashboard",
    label: t("dashboard.tab"),
    title: t("dashboard.section"),
    tone: "teal",
    menu: [
      {
        id: "theme",
        label: t("pref.theme"),
        glyph: "theme",
        value: <PreferenceValue pref="theme" />,
        heading: t("dashboard.theme.heading"),
        detail: (
          <>
            <p>{t("dashboard.theme.detail")}</p>
            <PreferenceOptions pref="theme" />
          </>
        )
      },
      {
        // Speed, and whether the blades' light drifts: one row, two groups.
        id: "motion",
        label: t("dashboard.animation"),
        glyph: "motion",
        value: <PreferenceValue pref="motion" />,
        heading: t("dashboard.animation"),
        detail: (
          <>
            <p className="pref-group m-0">{t("dashboard.animation.speed")}</p>
            <p className="m-0">{t("dashboard.motion.detail")}</p>
            <PreferenceOptions pref="motion" />
            <ReducedMotionOverride />
            <BackgroundDetail />
          </>
        )
      },
      {
        id: "sound",
        label: t("dashboard.sound"),
        glyph: "sound",
        value: <SoundRowValue />,
        heading: t("dashboard.sound"),
        detail: <SoundDetail />
      },
      {
        id: "freeze",
        label: t("dashboard.pause"),
        glyph: "play",
        value: <Value name="dashboard_freezes_game" />,
        heading: t("dashboard.pause"),
        detail: (
          <>
            <p>{t("dashboard.pause.detail")}</p>
            <Options
              name="dashboard_freezes_game"
              title={t("dashboard.pause")}
              options={[
                { value: true, label: t("common.on"), recommended: true },
                { value: false, label: t("common.off") }
              ]}
            />
          </>
        )
      },
      {
        id: "notifications",
        label: t("dashboard.notifications"),
        glyph: "bell",
        value: <PreferenceValue pref="notifications" />,
        heading: t("pref.notifications"),
        detail: (
          <>
            <p>{t("dashboard.notifications.detail")}</p>
            <PreferenceOptions pref="notifications" />
          </>
        )
      },
      {
        id: "language",
        label: t("pref.language"),
        glyph: "globe",
        value: <LanguageValue />,
        heading: t("pref.language"),
        detail: (
          <>
            <p>{t("dashboard.language.detail")}</p>
            <PreferenceOptions pref="language" />
            {AVAILABLE.length < 2 ? <Aside>{t("dashboard.language.more")}</Aside> : null}
          </>
        )
      },
      {
        id: "shortcuts",
        label: t("dashboard.shortcuts"),
        glyph: "keys",
        heading: t("dashboard.shortcuts.heading"),
        detail: (
          <>
            <Bullets
              items={[
                t("dashboard.shortcuts.blades"),
                t("dashboard.shortcuts.rows"),
                t("dashboard.shortcuts.open"),
                t("dashboard.shortcuts.options"),
                t("dashboard.shortcuts.back"),
                t("dashboard.shortcuts.hide"),
                t("dashboard.shortcuts.theme"),
                t("dashboard.shortcuts.show", { ways: dashboardShortcuts(state) })
              ]}
            />
            <Aside>{t("dashboard.shortcuts.saved")}</Aside>
          </>
        )
      },
      {
        id: "about",
        label: t("dashboard.about"),
        glyph: "question",
        value: <>{state.appVersion ? state.appVersion.replace(/^1\.0 /, "") : ""}</>,
        heading: t("dashboard.about.heading", { version: state.appVersion }),
        detail: (
          <>
            <p>{t("dashboard.about.detail")}</p>
            {/* Opens in the Mac's browser (blades_overlay_mac.mm). */}
            <p className="m-0">
              <a className="pane-link" href={WEBSITE}>
                macxenos.com
              </a>
            </p>
            <p className="menu-spec m-0 text-[0.85em]">
              {t("dashboard.about.build", { build: state.appBuild || t("dashboard.about.unknownBuild") })}
            </p>
          </>
        )
      }
    ]
  };
}
