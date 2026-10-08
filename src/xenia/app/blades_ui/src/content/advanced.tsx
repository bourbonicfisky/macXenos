import { useEffect, useState } from "react";
import type { BladeDef } from "../ui/blade";
import type { MenuItem } from "../ui/blade-menu";
import type { GlyphName } from "../ui/glyph";
import { send, useNative, type Cvar, type NativeState } from "../native";
import {
  NEXT_LAUNCH,
  Options,
  ResetButton,
  TextField,
  announceSetting
} from "../ui/settings-controls";
import { Aside } from "./parts";

/* ==========================================================================
   Every setting in the config file, by category, with the app's own
   description of each. The other blades cover what most people change; this
   one is for everything else, and for following a guide that names a
   setting exactly.
   ========================================================================== */

const CATEGORY: Record<string, { label: string; glyph: GlyphName }> = {
  GPU: { label: "GPU", glyph: "monitor" },
  Display: { label: "Display", glyph: "eye" },
  Video: { label: "Video", glyph: "resolution" },
  Vulkan: { label: "Vulkan", glyph: "layers" },
  CPU: { label: "CPU", glyph: "chip" },
  a64: { label: "CPU code (Apple Silicon)", glyph: "chip" },
  x64: { label: "CPU code (Intel)", glyph: "chip" },
  Kernel: { label: "Kernel", glyph: "console" },
  Memory: { label: "Memory", glyph: "blocks" },
  Threading: { label: "Threading", glyph: "split" },
  APU: { label: "Audio", glyph: "sound" },
  HID: { label: "Input", glyph: "pad" },
  SDL: { label: "Controller database", glyph: "pad" },
  Storage: { label: "Storage", glyph: "folder" },
  Content: { label: "Content", glyph: "disc" },
  Profiles: { label: "Profiles", glyph: "people" },
  General: { label: "General", glyph: "gear" },
  UI: { label: "Interface", glyph: "sliders" },
  Logging: { label: "Logging", glyph: "doc" },
  HACKS: { label: "Hacks", glyph: "wrench" },
  Config: { label: "Config file", glyph: "doc" }
};

/** Kept by Reset to defaults, as in the app (blades_model.cc KeptByResetAll):
    who signs in at boot, where saves and caches live, the file's version. */
export function keptByResetAll(cvar: Cvar) {
  return (
    cvar.category === "Profiles" ||
    cvar.category === "Config" ||
    cvar.name === "storage_root" ||
    cvar.name === "content_root" ||
    cvar.name === "cache_root"
  );
}

/** At the top of every Advanced pane. */
function Warning() {
  return (
    <p className="pane-warning m-0 flex gap-[0.6em] rounded-row px-[0.8em] py-[0.55em] text-[0.85em]">
      <span className="pane-warning-mark shrink-0" aria-hidden="true">
        !
      </span>
      <span>
        These are the emulator&rsquo;s own settings, mostly for developers. A wrong value can stop
        games from starting or running correctly. Change one only when a guide names it, and use
        Reset to defaults, at the end of this blade, if something goes wrong.
      </span>
    </p>
  );
}

/** Every changed setting back to how macXenos ships, after a second press. */
function ResetAll() {
  const { cvars } = useNative();
  const [armed, setArmed] = useState(false);
  const changed = Object.values(cvars).filter(
    (cvar) => !keptByResetAll(cvar) && cvar.value !== cvar.defaultValue
  );
  // Disarm if the cursor wanders off and comes back later.
  useEffect(() => {
    if (!armed) return;
    const timer = window.setTimeout(() => setArmed(false), 6000);
    return () => window.clearTimeout(timer);
  }, [armed]);

  if (changed.length === 0) {
    return <p className="menu-note m-0 text-[0.78em]">Every setting is at its default.</p>;
  }
  return (
    <>
      <p className="m-0">
        {changed.length === 1 ? "1 setting differs" : `${changed.length} settings differ`} from the
        defaults.
      </p>
      <button
        type="button"
        className="pane-button pane-button--danger rounded-row mt-[0.3em] inline-flex min-h-[2.2em] cursor-pointer items-center justify-self-start px-[0.9em]"
        data-armed={armed || undefined}
        onClick={() => {
          if (!armed) {
            setArmed(true);
            return;
          }
          setArmed(false);
          send({ op: "resetAll" });
          announceSetting("All settings", "Defaults");
        }}
      >
        {armed ? "Press again to reset them" : "Reset to defaults"}
      </button>
    </>
  );
}

/** One setting: its exact name, what it does, and a control for its type. */
function SettingEntry({ cvar }: { cvar: Cvar }) {
  const later = NEXT_LAUNCH.has(cvar.name);
  return (
    <div className="setting-entry grid gap-[0.25em]">
      <p className="m-0 flex flex-wrap items-baseline gap-x-[0.6em]">
        <span className="setting-name">{cvar.name}</span>
        {cvar.value !== cvar.defaultValue ? <span className="setting-changed">changed</span> : null}
      </p>
      {cvar.description ? (
        <p className="menu-note m-0 text-[0.8em] whitespace-pre-line">{cvar.description}</p>
      ) : null}
      {cvar.type === "bool" ? (
        <Options
          name={cvar.name}
          title={cvar.name}
          options={[
            { value: false, label: "Off" },
            { value: true, label: "On" }
          ]}
        />
      ) : (
        <TextField
          name={cvar.name}
          title={cvar.name}
          numeric={cvar.type !== "string" && cvar.type !== "path"}
          placeholder={cvar.type === "string" || cvar.type === "path" ? "(empty)" : undefined}
        />
      )}
      <ResetButton name={cvar.name} title={cvar.name} />
      {later ? <Aside>Applies the next time a game starts.</Aside> : null}
    </div>
  );
}

/** Finds settings by name or description, across every category. */
function Search() {
  const { cvars } = useNative();
  const [query, setQuery] = useState("");
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const found =
    terms.length === 0
      ? []
      : Object.values(cvars)
          .filter((cvar) => {
            const text = `${cvar.name} ${cvar.description}`.toLowerCase();
            return terms.every((term) => text.includes(term));
          })
          .sort((a, b) => a.name.localeCompare(b.name))
          .slice(0, 40);
  return (
    <>
      <input
        type="search"
        spellCheck={false}
        aria-label="Find a setting"
        placeholder="Type part of a name, like vsync or deadzone"
        className="pref-text rounded-row mt-[0.3em] min-h-[2.2em] w-full px-[0.7em]"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Escape" || event.key === "Enter") {
            event.preventDefault();
            event.currentTarget.blur();
          }
        }}
      />
      {terms.length > 0 && found.length === 0 ? <Aside>No setting matches.</Aside> : null}
      <div className="grid">
        {found.map((cvar) => (
          <SettingEntry key={cvar.name} cvar={cvar} />
        ))}
      </div>
    </>
  );
}

export function advanced(state: NativeState): BladeDef {
  const byCategory = new Map<string, Cvar[]>();
  for (const cvar of Object.values(state.cvars)) {
    const list = byCategory.get(cvar.category) ?? [];
    list.push(cvar);
    byCategory.set(cvar.category, list);
  }
  const categories = [...byCategory.keys()].sort((a, b) =>
    (CATEGORY[a]?.label ?? a).localeCompare(CATEGORY[b]?.label ?? b)
  );

  const menu: MenuItem[] = [
    {
      id: "find",
      label: "Find a setting",
      glyph: "target",
      heading: "Find a setting",
      detail: (
        <>
          <Warning />
          <p>Every setting in the config file, by its exact name. Choose the field and start typing.</p>
          <Search />
        </>
      )
    },
    ...categories.map((category): MenuItem => {
      const list = (byCategory.get(category) ?? []).sort((a, b) => a.name.localeCompare(b.name));
      const changed = list.filter((cvar) => cvar.value !== cvar.defaultValue).length;
      return {
        id: `category-${category.toLowerCase()}`,
        label: CATEGORY[category]?.label ?? category,
        glyph: CATEGORY[category]?.glyph ?? "gear",
        value: <>{changed > 0 ? `${changed} changed` : list.length}</>,
        heading: `${CATEGORY[category]?.label ?? category} settings`,
        detail: (
          <>
            <Warning />
            <Aside>
              {list.length} settings. The other blades cover what most people change.
            </Aside>
            <div className="grid">
              {list.map((cvar) => (
                <SettingEntry key={cvar.name} cvar={cvar} />
              ))}
            </div>
          </>
        )
      };
    }),
    {
      id: "reset",
      label: "Reset to defaults",
      glyph: "wrench",
      heading: "Reset to defaults",
      detail: (
        <>
          <p>
            Puts every setting back to how macXenos ships, across all the blades. Your signed-in
            profiles, your save and cache folders, your console settings and button mappings are
            kept.
          </p>
          <ResetAll />
          <Aside>Some settings take effect the next time a game starts.</Aside>
        </>
      )
    }
  ];

  return { id: "advanced", label: "advanced", title: "Advanced", tone: "steel", menu };
}
