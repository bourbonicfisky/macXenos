import { useEffect, useRef } from "react";
import { ADJUST_EVENT } from "./blade-menu";
import { send, useNative, type ConsoleField } from "../native";
import { Dropdown } from "./dropdown";
import { showToast } from "./toast";

/** Above this many choices, a setting is a drop-down rather than sheets. */
const SHEETS_UP_TO = 8;

/* The console's own settings (the 360's XConfig: language, time zone, video
   and audio output). They live in the emulated console rather than the config
   file, and a running game holds them, so they lock while one is open. */

function useField(id: string): { field?: ConsoleField; editable: boolean } {
  const { console: consoleState } = useNative();
  return {
    field: consoleState.fields.find((f) => f.id === id),
    editable: consoleState.editable
  };
}

function announce(field: ConsoleField, label: string) {
  showToast("Console setting saved", `${field.label}: ${label}`);
}

function locked() {
  showToast("Console settings are locked", "Quit the game to change them");
}

/** A console setting's choices, as option sheets. */
export function ConsoleOptions({ id }: { id: string }) {
  const { field, editable } = useField(id);
  if (!field || !field.options) return null;
  if (field.options.length > SHEETS_UP_TO) {
    return (
      <Dropdown
        title={field.label}
        options={field.options}
        value={field.value as number}
        disabled={!editable}
        onChoose={(option) => {
          if (!editable) return locked();
          send({ op: "console", id, value: option.value });
          announce(field, option.label);
        }}
      />
    );
  }
  return (
    <div role="radiogroup" aria-label={field.label} className="mt-[0.3em] flex flex-wrap gap-[0.5em]">
      {field.options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={field.value === option.value}
          aria-disabled={!editable || undefined}
          className="pref-option rounded-row inline-flex min-h-[2.2em] cursor-pointer items-center px-[0.85em]"
          onClick={() => {
            if (field.value === option.value) return;
            if (!editable) return locked();
            send({ op: "console", id, value: option.value });
            announce(field, option.label);
          }}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

/** A console on/off setting, as a checkbox in a list. */
export function ConsoleCheck({ id }: { id: string }) {
  const { field, editable } = useField(id);
  if (!field) return null;
  return (
    <label className="pref-check flex cursor-pointer items-center gap-[0.55em]">
      <input
        type="checkbox"
        className="pref-check-box"
        checked={field.value === true}
        aria-disabled={!editable || undefined}
        onChange={(event) => {
          if (!editable) {
            event.preventDefault();
            return locked();
          }
          send({ op: "console", id, value: event.target.checked });
          announce(field, event.target.checked ? "On" : "Off");
        }}
      />
      <span>{field.label}</span>
    </label>
  );
}

/** A console value from 0 to 1 (the music player's volume). */
export function ConsoleSlider({ id }: { id: string }) {
  const { field, editable } = useField(id);
  const ref = useRef<HTMLDivElement>(null);
  const value = typeof field?.value === "number" ? field.value : 0;
  const valueRef = useRef(value);
  valueRef.current = value;

  const commit = (next: number) => {
    if (!field) return;
    if (!editable) return locked();
    const clamped = Math.round(Math.min(1, Math.max(0, next)) * 20) / 20;
    if (Math.abs(clamped - valueRef.current) < 1e-6) return;
    send({ op: "console", id, value: clamped });
    announce(field, `${Math.round(clamped * 100)}%`);
  };

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const onAdjust = (event: Event) =>
      commit(valueRef.current + (event as CustomEvent<number>).detail * 0.05);
    element.addEventListener(ADJUST_EVENT, onAdjust);
    return () => element.removeEventListener(ADJUST_EVENT, onAdjust);
  });

  if (!field) return null;
  return (
    <div
      ref={ref}
      role="slider"
      tabIndex={0}
      aria-label={field.label}
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuenow={value}
      aria-valuetext={`${Math.round(value * 100)}%`}
      aria-disabled={!editable || undefined}
      className="pref-slider rounded-row mt-[0.3em] flex min-h-[2.2em] items-center gap-[0.8em] px-[0.85em]"
      onPointerDown={(event) => {
        const track = event.currentTarget.querySelector(".pref-slider-track");
        if (!track) return;
        const box = track.getBoundingClientRect();
        commit((event.clientX - box.left) / box.width);
      }}
    >
      <span className="pref-slider-track relative h-[0.45em] flex-1 rounded-full">
        <span className="pref-slider-fill absolute inset-y-0 left-0 rounded-full" style={{ width: `${value * 100}%` }} />
        <span className="pref-slider-knob absolute top-1/2 rounded-full" style={{ left: `${value * 100}%` }} />
      </span>
      <span className="min-w-[3.2em] text-right tabular-nums">{Math.round(value * 100)}%</span>
    </div>
  );
}

/** A console value shown, not edited (the network identity). */
export function ConsoleReadout({ id }: { id: string }) {
  const { field } = useField(id);
  if (!field) return null;
  return (
    <p className="m-0 flex flex-wrap gap-x-[0.6em]">
      <span>{field.label}</span>
      <span className="setting-name font-normal">{String(field.value)}</span>
    </p>
  );
}

/** The chosen option's label, at the right of a row. */
export function ConsoleValue({ id }: { id: string }) {
  const { field } = useField(id);
  if (!field) return null;
  if (field.options) return <>{field.options.find((o) => o.value === field.value)?.label ?? "—"}</>;
  if (typeof field.value === "boolean") return <>{field.value ? "On" : "Off"}</>;
  return <>{String(field.value)}</>;
}

/** Said once per pane while a game holds the console settings. */
export function ConsoleLockNote() {
  const { console: consoleState } = useNative();
  if (consoleState.editable) return null;
  return (
    <p className="menu-note m-0 text-[0.78em]">
      A game is running, which holds these. Quit it to change them.
    </p>
  );
}
