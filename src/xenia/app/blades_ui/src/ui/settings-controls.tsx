import { useEffect, useRef, useState, type ReactNode } from "react";
import { ADJUST_EVENT } from "./blade-menu";
import { Cvar, CvarValue, setCvar, send, useNative } from "../native";
import { Dropdown } from "./dropdown";
import { showToast } from "./toast";

/* ==========================================================================
   The controls a pane offers for one emulator setting. They read the app's
   state (native.ts) and send every change straight back, so a choice applies
   as it is made — the same contract as the site's Preferences blade.

   Each control is shaped for the blades' sub-menu (blade-menu.tsx): a set of
   options is one radiogroup, a slider is one [role=slider] that ← → step,
   anything else is its own button or input.
   ========================================================================== */

export type Option = {
  value: CvarValue;
  label: string;
  short?: string;
  /** Marked "Recommended" on its sheet. */
  recommended?: boolean;
};

/** Settings that take effect only when a game next starts. The app cannot
    tell these apart from live ones, so they are listed here, beside the copy
    that describes them. */
export const NEXT_LAUNCH = new Set([
  "gpu",
  "vsync",
  "framerate_limit",
  "draw_resolution_scale_x",
  "draw_resolution_scale_y",
  "anisotropic_override",
  "window_guest_resolution_mode",
  "metal_msaa_clear_fold",
  "metal_stencil_export",
  "metal_resolve_span_coalesce",
  "metal_raster_sync_skip",
  "metal_raster_mark_skip",
  "metal_fsr_half_precision",
  "mvk_config_defaults",
  "apu",
  "hid",
  "xma_decoder",
  "use_dedicated_xma_thread",
  "apu_max_queued_frames",
  "enable_xmp",
  "guest_thread_qos",
  "guest_thread_qos_class",
  "ignore_thread_priorities",
  "ignore_thread_affinities",
  "ignore_trap_instructions",
  "disable_context_promotion",
  "clock_no_scaling",
  "clock_source_raw",
  "apply_patches",
  "allow_plugins",
  "apply_title_update",
  "allow_incompatible_title_update",
  "mount_cache",
  "mount_scratch",
  "mount_memory_unit",
  "storage_root",
  "content_root",
  "cache_root",
  "keyboard_mode",
  "keyboard_user_index",
  "controller_hotkeys",
  "discord"
]);

/** "Setting saved — VSync: On", the site's achievement-style toast. */
export function announceSetting(name: string, valueLabel: string) {
  const later = NEXT_LAUNCH.has(name) ? " (next launch)" : "";
  showToast("Setting saved", `${name}: ${valueLabel}${later}`);
}

export function useCvar(name: string): Cvar | undefined {
  return useNative().cvars[name];
}

function same(a: CvarValue | undefined, b: CvarValue) {
  if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) < 1e-6;
  return a === b;
}

/** A setting's options, as a row of selectable sheets. */
export function Options({
  name,
  title,
  options,
  onChoose,
  current
}: {
  name: string;
  /** For the toast and the screen reader: "VSync", "Upscaler". */
  title: string;
  options: Option[];
  /** Instead of setting `name` (a choice that sets two settings at once). */
  onChoose?: (value: CvarValue) => void;
  /** What counts as chosen, when it isn't the stored value as-is (an empty
      upscaler setting means bilinear). */
  current?: CvarValue;
}) {
  const cvar = useCvar(name);
  if (!cvar) return null;
  const chosen = current ?? cvar.value;
  const choose = (option: Option) => {
    if (same(chosen, option.value)) return;
    if (onChoose) onChoose(option.value);
    else setCvar(name, option.value);
    announceSetting(title, option.label);
  };
  // Too many to lay out as sheets: a drop-down.
  if (options.length > 8 && options.every((o) => typeof o.value !== "boolean")) {
    const keyed = options.map((o) => ({ value: String(o.value), label: o.label, option: o }));
    return (
      <Dropdown
        title={title}
        options={keyed}
        value={String(chosen)}
        onChoose={(picked) => {
          const found = keyed.find((k) => k.value === picked.value);
          if (found) choose(found.option);
        }}
      />
    );
  }
  return (
    <div role="radiogroup" aria-label={title} className="mt-[0.3em] flex flex-wrap gap-[0.5em]">
      {options.map((option) => (
        <button
          key={String(option.value)}
          type="button"
          role="radio"
          aria-checked={same(chosen, option.value)}
          className="pref-option rounded-row inline-flex min-h-[2.2em] cursor-pointer items-center whitespace-nowrap px-[0.85em]"
          onClick={() => choose(option)}
        >
          {option.label}
          {option.recommended ? <span className="pref-tag">Recommended</span> : null}
        </button>
      ))}
    </div>
  );
}

/** Off / On, for a true-or-false setting. */
export function Toggle({
  name,
  title,
  off = "Off",
  on = "On",
  invert = false
}: {
  name: string;
  title: string;
  off?: string;
  on?: string;
  /** For "disable_*" settings, where On means the setting is false. */
  invert?: boolean;
}) {
  return (
    <Options
      name={name}
      title={title}
      options={[
        { value: invert, label: off },
        { value: !invert, label: on }
      ]}
    />
  );
}

/**
 * A value on a track. ← and → step it in a sub-menu (ADJUST_EVENT), a click
 * or drag on the track sets it, and the value reads out at the right.
 */
export function Slider({
  name,
  title,
  min,
  max,
  step,
  format = (v) => v.toFixed(2)
}: {
  name: string;
  title: string;
  min: number;
  max: number;
  step: number;
  format?: (value: number) => string;
}) {
  const cvar = useCvar(name);
  const ref = useRef<HTMLDivElement>(null);
  const value = typeof cvar?.value === "number" ? cvar.value : min;
  const valueRef = useRef(value);
  valueRef.current = value;

  const commit = (next: number) => {
    const snapped = Math.min(max, Math.max(min, Math.round(next / step) * step));
    const rounded = Number(snapped.toFixed(4));
    if (same(valueRef.current, rounded)) return;
    setCvar(name, rounded);
    announceSetting(title, format(rounded));
  };

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const onAdjust = (event: Event) => {
      commit(valueRef.current + (event as CustomEvent<number>).detail * step);
    };
    element.addEventListener(ADJUST_EVENT, onAdjust);
    return () => element.removeEventListener(ADJUST_EVENT, onAdjust);
  });

  if (!cvar) return null;
  const fraction = (value - min) / (max - min || 1);

  const fromPointer = (clientX: number) => {
    const track = ref.current?.querySelector(".pref-slider-track");
    if (!track) return;
    const box = track.getBoundingClientRect();
    commit(min + ((clientX - box.left) / box.width) * (max - min));
  };

  return (
    <div
      ref={ref}
      role="slider"
      tabIndex={0}
      aria-label={title}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={value}
      aria-valuetext={format(value)}
      className="pref-slider rounded-row mt-[0.3em] flex min-h-[2.2em] items-center gap-[0.8em] px-[0.85em]"
      onPointerDown={(event) => {
        (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
        fromPointer(event.clientX);
      }}
      onPointerMove={(event) => {
        if (event.buttons & 1) fromPointer(event.clientX);
      }}
    >
      <span className="pref-slider-track relative h-[0.45em] flex-1 rounded-full">
        <span
          className="pref-slider-fill absolute inset-y-0 left-0 rounded-full"
          style={{ width: `${fraction * 100}%` }}
        />
        <span
          className="pref-slider-knob absolute top-1/2 rounded-full"
          style={{ left: `${fraction * 100}%` }}
        />
      </span>
      <span className="min-w-[3.2em] text-right tabular-nums">{format(value)}</span>
    </div>
  );
}

/** A text or number entered by hand. Return or leaving the field saves it;
    Escape puts it back. ↑ ↓ leave the field for the next control. */
export function TextField({
  name,
  title,
  placeholder,
  numeric = false
}: {
  name: string;
  title: string;
  placeholder?: string;
  numeric?: boolean;
}) {
  const cvar = useCvar(name);
  const shown = cvar ? String(cvar.value) : "";
  const [draft, setDraft] = useState(shown);
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!editing) setDraft(shown);
  }, [shown, editing]);
  if (!cvar) return null;

  const commit = () => {
    setEditing(false);
    if (draft === shown) return;
    let value: CvarValue = draft;
    if (cvar.type === "int" || cvar.type === "uint") {
      const parsed = Number.parseInt(draft, 10);
      if (Number.isNaN(parsed) || (cvar.type === "uint" && parsed < 0)) {
        setDraft(shown);
        return;
      }
      value = parsed;
    } else if (cvar.type === "float") {
      const parsed = Number.parseFloat(draft);
      if (Number.isNaN(parsed)) {
        setDraft(shown);
        return;
      }
      value = parsed;
    }
    setCvar(name, value);
    announceSetting(title, String(value) || "(empty)");
  };

  return (
    <input
      type="text"
      inputMode={numeric ? "decimal" : undefined}
      spellCheck={false}
      aria-label={title}
      placeholder={placeholder}
      className="pref-text rounded-row mt-[0.3em] min-h-[2.2em] w-full px-[0.7em]"
      value={draft}
      onFocus={() => setEditing(true)}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          event.currentTarget.blur();
        } else if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          setDraft(shown);
          setEditing(false);
          (event.currentTarget.closest("[data-submenu]") as HTMLElement | null)?.focus();
          event.currentTarget.blur();
        }
      }}
    />
  );
}

/** One switch in a list of them (the troubleshooting switches, Advanced).
    `recommended` adds what the recommended setting is, taken from the value
    the app ships with. */
export function Check({
  name,
  label,
  invert = false,
  recommended = false
}: {
  name: string;
  label: ReactNode;
  invert?: boolean;
  recommended?: boolean;
}) {
  const cvar = useCvar(name);
  if (!cvar) return null;
  const checked = invert ? cvar.value === false : cvar.value === true;
  const recommendedOn = invert ? cvar.defaultValue === false : cvar.defaultValue === true;
  return (
    <label className="pref-check flex cursor-pointer items-center gap-[0.55em]">
      <input
        type="checkbox"
        className="pref-check-box shrink-0"
        checked={checked}
        onChange={(event) => {
          const next = invert ? !event.target.checked : event.target.checked;
          setCvar(name, next);
          announceSetting(typeof label === "string" ? label : name, event.target.checked ? "On" : "Off");
        }}
      />
      <span className="grid">
        <span>{label}</span>
        {recommended ? (
          <span className="pref-recommended" data-differs={checked !== recommendedOn || undefined}>
            Recommended {recommendedOn ? "on" : "off"}
          </span>
        ) : null}
      </span>
    </label>
  );
}

/** Puts a group of settings back to the recommended ones (how the app
    ships), or says they already are. */
export function UseRecommended({ names, title }: { names: string[]; title: string }) {
  const { cvars } = useNative();
  const differing = names.filter((n) => cvars[n] && !same(cvars[n].value, cvars[n].defaultValue));
  if (differing.length === 0) {
    return <p className="menu-note m-0 text-[0.78em]">These are the recommended settings.</p>;
  }
  return (
    <button
      type="button"
      className="pane-button rounded-row mt-[0.3em] inline-flex min-h-[2.2em] cursor-pointer items-center justify-self-start px-[0.9em]"
      onClick={() => {
        for (const name of differing) send({ op: "reset", name });
        announceSetting(title, "Recommended");
      }}
    >
      Use recommended settings
    </button>
  );
}

/** The current setting, at the right of its row. */
export function Value({
  name,
  options,
  format,
  current
}: {
  name: string;
  options?: Option[];
  format?: (value: CvarValue) => string;
  current?: CvarValue;
}) {
  const cvar = useCvar(name);
  if (!cvar) return null;
  if (options) {
    const found = options.find((o) => same(current ?? cvar.value, o.value));
    return <>{found?.short ?? found?.label ?? String(cvar.value)}</>;
  }
  if (format) return <>{format(cvar.value)}</>;
  if (typeof cvar.value === "boolean") return <>{cvar.value ? "On" : "Off"}</>;
  return <>{String(cvar.value) || "Default"}</>;
}

/** Shown under a pane once one of its settings waits for the next launch. */
export function NextLaunchNote({ names }: { names: string[] }) {
  const { cvars, titleOpen } = useNative();
  const waiting = names.some((n) => NEXT_LAUNCH.has(n) && cvars[n]?.changed);
  if (!waiting) return null;
  return (
    <p className="menu-note m-0 text-[0.78em]">
      {titleOpen
        ? "Takes effect when the game restarts. The tray below restarts it now."
        : "Takes effect when you next start a game."}
    </p>
  );
}

/** Puts one setting back to how the app ships. */
export function ResetButton({ name, title }: { name: string; title: string }) {
  const cvar = useCvar(name);
  if (!cvar || same(cvar.value, cvar.defaultValue)) return null;
  return (
    <button
      type="button"
      className="pane-button rounded-row mt-[0.3em] inline-flex min-h-[2.2em] cursor-pointer items-center justify-self-start px-[0.9em]"
      onClick={() => {
        send({ op: "reset", name });
        announceSetting(title, "Default");
      }}
    >
      Reset to default
    </button>
  );
}
