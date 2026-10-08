import type { BladeDef } from "../ui/blade";
import { Bullets } from "../ui/blade-menu";
import type { NativeState } from "../native";
import {
  PreferenceOptions,
  PreferenceValue,
  ReducedMotionOverride,
  SoundValue
} from "../ui/preference-controls";
import { Aside } from "./parts";
import { Options, Value } from "../ui/settings-controls";
import { dashboardShortcuts } from "./home";

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
        value: <SoundValue />,
        heading: "Dashboard sounds",
        detail: (
          <>
            <p>The Xbox 360 dashboard&rsquo;s own sounds as you move through these blades.</p>
            <PreferenceOptions pref="sound" />
            <p className="pref-group">Volume</p>
            <PreferenceOptions pref="volume" />
          </>
        )
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
