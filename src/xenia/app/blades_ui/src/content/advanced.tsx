import { t } from "../i18n";
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
  Display: { get label() { return t("advanced.display"); }, glyph: "eye" },
  Video: { get label() { return t("advanced.video"); }, glyph: "resolution" },
  Vulkan: { label: "Vulkan", glyph: "layers" },
  CPU: { label: "CPU", glyph: "chip" },
  a64: { get label() { return t("advanced.cpuCodeAppleSilicon"); }, glyph: "chip" },
  x64: { get label() { return t("advanced.cpuCodeIntel"); }, glyph: "chip" },
  Kernel: { get label() { return t("advanced.kernel"); }, glyph: "console" },
  Memory: { get label() { return t("advanced.memory"); }, glyph: "blocks" },
  Threading: { get label() { return t("advanced.threading"); }, glyph: "split" },
  APU: { get label() { return t("advanced.audio"); }, glyph: "sound" },
  HID: { get label() { return t("advanced.input"); }, glyph: "pad" },
  SDL: { get label() { return t("advanced.controllerDatabase"); }, glyph: "pad" },
  Storage: { get label() { return t("advanced.storage"); }, glyph: "folder" },
  Content: { get label() { return t("advanced.content"); }, glyph: "disc" },
  Profiles: { get label() { return t("advanced.profiles"); }, glyph: "people" },
  General: { get label() { return t("advanced.general"); }, glyph: "gear" },
  UI: { get label() { return t("advanced.interface"); }, glyph: "sliders" },
  Logging: { get label() { return t("advanced.logging"); }, glyph: "doc" },
  HACKS: { get label() { return t("advanced.hacks"); }, glyph: "wrench" },
  Config: { get label() { return t("advanced.configFile"); }, glyph: "doc" }
};

/** Kept by Reset to defaults, as in the app (blades_model.cc KeptByResetAll):
    who signs in at boot, where saves and caches live, the file's version. */
export function keptByResetAll(cvar: Cvar) {
  return (
    cvar.category === "Profiles" ||
    cvar.category === "Config" ||
    cvar.name === "storage_root" ||
    cvar.name === "content_root" ||
    cvar.name === "cache_root" ||
    cvar.name === "install_root" ||
    cvar.name === "save_root"
  );
}

/** At the top of every Advanced pane. */
function Warning() {
  return (
    <p className="pane-warning m-0 flex gap-[0.6em] rounded-row px-[0.8em] py-[0.55em] text-[0.85em]">
      <span className="pane-warning-mark shrink-0" aria-hidden="true">
        !
      </span>
      <span>{t("advanced.warning")}</span>
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
    return <p className="menu-note m-0 text-[0.78em]">{t("advanced.defaultsUnchanged")}</p>;
  }
  return (
    <>
      <p className="m-0">
        {t("advanced.changedSettings", { count: changed.length })}
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
          announceSetting(t("advanced.allSettings"), t("settings.defaults"));
        }}
      >
        {armed ? t("advanced.confirmReset") : t("settings.resetDefaults")}
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
        {cvar.value !== cvar.defaultValue ? <span className="setting-changed">{t("advanced.changed")}</span> : null}
      </p>
      {cvar.description ? (
        <p className="menu-note m-0 text-[0.8em] whitespace-pre-line">{cvar.description}</p>
      ) : null}
      {cvar.type === "bool" ? (
        <Options
          name={cvar.name}
          title={cvar.name}
          options={[
            { value: false, label: t("graphics.qos.off") },
            { value: true, label: t("graphics.unsetInputs.on") }
          ]}
        />
      ) : (
        <TextField
          name={cvar.name}
          title={cvar.name}
          numeric={cvar.type !== "string" && cvar.type !== "path"}
          placeholder={cvar.type === "string" || cvar.type === "path" ? t("settings.empty") : undefined}
        />
      )}
      <ResetButton name={cvar.name} title={cvar.name} />
      {later ? <Aside>{t("advanced.nextLaunch")}</Aside> : null}
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
        aria-label={t("advanced.findASetting")}
        placeholder={t("advanced.search.placeholder")}
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
      {terms.length > 0 && found.length === 0 ? <Aside>{t("advanced.search.empty")}</Aside> : null}
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
      label: t("advanced.findASetting"),
      glyph: "target",
      heading: t("advanced.findASetting"),
      detail: (
        <>
          <Warning />
          <p>{t("advanced.search.detail")}</p>
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
        value: <>{changed > 0 ? t("advanced.changedCount", { count: changed }) : list.length}</>,
        heading: t("advanced.categorySettings", { category: CATEGORY[category]?.label ?? category }),
        detail: (
          <>
            <Warning />
            <Aside>
              {t("advanced.categoryCount", { count: list.length })}
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
      label: t("settings.resetDefaults"),
      glyph: "wrench",
      heading: t("settings.resetDefaults"),
      detail: (
        <>
          <p>{t("advanced.reset.detail")}</p>
          <ResetAll />
          <Aside>{t("advanced.reset.nextLaunch")}</Aside>
        </>
      )
    }
  ];

  return { id: "advanced", label: t("advanced.tab"), title: t("advanced.title"), tone: "steel", menu };
}
