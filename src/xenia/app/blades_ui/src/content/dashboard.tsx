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

/** Where the sounds come from, and the way to get them if they aren't here. */
function SoundSource() {
  const { dashSounds } = useNative();
  switch (dashSounds.status) {
    case "ready":
      return <Aside>Downloaded: {packCredit(dashSounds)}.</Aside>;
    case "downloading":
      return <Aside>Downloading {packCredit(dashSounds)} in the background.</Aside>;
    case "missing":
    case "failed":
      return (
        <>
          <p className="m-0">
            macXenos doesn&rsquo;t include the dashboard&rsquo;s sounds. Download a fan-made pack,{" "}
            {packCredit(dashSounds)}, of about 1&nbsp;MB.
          </p>
          {dashSounds.status === "failed" ? (
            <Aside>The last download failed{dashSounds.error ? `: ${dashSounds.error}` : ""}.</Aside>
          ) : null}
          <PaneAction onSelect={downloadDashSounds}>
            {dashSounds.status === "failed" ? "Try again" : "Download sounds"}
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
      {needed ? <p className="pref-group m-0">Once downloaded</p> : null}
      <PreferenceOptions pref="sound" />
      <p className="pref-group">Volume</p>
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
      <p>The Xbox 360 dashboard&rsquo;s own sounds as you move through these blades.</p>
      {settings}
      <SoundSource />
    </>
  );
}

/** The Sound row's value: what's playing, or why nothing is. */
function SoundRowValue() {
  const { dashSounds } = useNative();
  if (dashSounds.status === "missing" || dashSounds.status === "failed") return <>Not downloaded</>;
  if (dashSounds.status === "downloading") return <>Downloading</>;
  return <SoundValue />;
}

/** The blades' own look and sound, as on the site's Preferences blade. They
    lead the System blade (system.tsx) under a "Dashboard" heading. */
export function dashboard(state: NativeState): BladeDef {
  return {
    id: "dashboard",
    label: "dashboard",
    title: "Dashboard",
    tone: "teal",
    menu: [
      {
        id: "theme",
        label: "Theme",
        glyph: "theme",
        value: <PreferenceValue pref="theme" />,
        heading: "Day or night",
        detail: (
          <>
            <p>Match your Mac&rsquo;s light or dark setting, or pick one. Y switches between them too.</p>
            <PreferenceOptions pref="theme" />
          </>
        )
      },
      {
        id: "motion",
        label: "Animation speed",
        glyph: "motion",
        value: <PreferenceValue pref="motion" />,
        heading: "Animation speed",
        detail: (
          <>
            <p>Fast is how the dashboard moved. Slow is gentler. Reduced motion switches blades instantly.</p>
            <PreferenceOptions pref="motion" />
            <ReducedMotionOverride />
          </>
        )
      },
      {
        id: "sound",
        label: "Dashboard sounds",
        glyph: "sound",
        value: <SoundRowValue />,
        heading: "Dashboard sounds",
        detail: <SoundDetail />
      },
      {
        id: "freeze",
        label: "Pause games",
        glyph: "play",
        value: <Value name="dashboard_freezes_game" />,
        heading: "Pause games",
        detail: (
          <>
            <p>
              Freezes the game while this dashboard is open, so nothing happens
              in it until you go back. Games also get the console&rsquo;s own
              notice, which most use to pause themselves.
            </p>
            <Options
              name="dashboard_freezes_game"
              title="Pause games"
              options={[
                { value: true, label: "On", recommended: true },
                { value: false, label: "Off" }
              ]}
            />
          </>
        )
      },
      {
        id: "notifications",
        label: "Notifications",
        glyph: "bell",
        value: <PreferenceValue pref="notifications" />,
        heading: "Notification position",
        detail: (
          <>
            <p>Where &ldquo;Setting saved&rdquo; appears. The Xbox 360 showed it at the bottom center.</p>
            <PreferenceOptions pref="notifications" />
          </>
        )
      },
      {
        id: "shortcuts",
        label: "Shortcuts",
        glyph: "keys",
        heading: "Getting around",
        detail: (
          <>
            <Bullets
              items={[
                "← and →, or the d-pad, change blade",
                "↑ and ↓ move through a menu",
                "A or Return opens a setting's options",
                "In the options, ↑ and ↓ move between groups and ← → choose",
                "B goes back, to Home from the top of a blade",
                "A tap of Guide (or Escape) hides the menu",
                "Y switches day and night",
                `${dashboardShortcuts(state)} opens the dashboard from anywhere`
              ]}
            />
            <Aside>Settings save as you choose them.</Aside>
          </>
        )
      },
      {
        id: "about",
        label: "About macXenos",
        glyph: "question",
        value: <>{state.appVersion ? state.appVersion.replace(/^1\.0 /, "") : ""}</>,
        heading: `macXenos ${state.appVersion}`,
        detail: (
          <>
            <p>Xbox 360 games on your Mac, with a native Metal renderer.</p>
            <p className="menu-spec m-0 text-[0.85em]">Build {state.appBuild || "unknown"}</p>
          </>
        )
      }
    ]
  };
}
