import { useState, type ReactNode } from "react";
import { BladeMenu, MenuItem, Tray, TrayPill } from "./blade-menu";
import { TabLabel } from "./tab-label";
import { toggleTheme, useTheme } from "./theme";
import { send } from "../native";
import { LegendStack } from "./legend";

export type BladeTone = "green" | "orange" | "amber" | "blue" | "purple" | "teal" | "rose" | "steel";

export type BladeDef = {
  id: string;
  /** Vertical text on the tab, and the label in the mobile strip. Lowercase,
      as the dashboard set them. */
  label: string;
  /** The heading that sits at the top of the open blade — its name, nothing
      more, as "Games" or "Media" was. */
  title: string;
  tone: BladeTone;
  menu: MenuItem[];
  feature?: ReactNode;
  tray?: Tray;
};

type BladeProps = {
  blade: BladeDef;
  state: "open" | "tab";
  side: "left" | "center" | "right";
  depth: number;
  onOpen: () => void;
};

/**
 * One blade: its metal tab when folded, its full chrome when open.
 *
 * The curved surface is a separate skin layer rather than the section's own
 * background, because the clip that bends it would otherwise take the drop
 * shadow and the text with it.
 */
export function Blade({ blade, state, side, depth, onOpen }: BladeProps) {
  const theme = useTheme();
  // While A has a pane's sub-menu open, the legend reads Back (B).
  const [submenu, setSubmenu] = useState(false);

  return (
    <section
      id={blade.id}
      className="blade relative flex min-w-0"
      data-state={state}
      data-side={side}
      data-bend={side === "right" ? "right" : "left"}
      data-tone={blade.tone}
      style={{ zIndex: 20 - depth }}
      aria-label={blade.label}
    >
      {/* The tab never leaves the rail — the open blade keeps its own, and its
          colour lives in the overlay, not the base sheet.

          Paint order matters: the panel goes down first so the tab sitting on
          top of it throws a real shadow across the content — that is the left
          inner edge. The right one cannot work the same way, because the open
          blade has to stay above the rails or its own tab gets buried, so that
          side is an inset shadow on the panel instead. */}
      {state === "open" ? (
        <span
          className="blade-shadow blade-shadow--panel pointer-events-none absolute inset-y-0 right-0 left-[var(--tab-w)] -z-20"
          aria-hidden="true"
        >
          <span className="blade-skin blade-skin--panel absolute inset-0" />
        </span>
      ) : null}

      <span
        className="blade-shadow blade-shadow--tab pointer-events-none absolute inset-y-0 left-0 -z-10 w-[var(--tab-w)]"
        aria-hidden="true"
      >
        <span className="blade-skin blade-skin--tab absolute inset-0" />
        {/* The highlight sits inside a copy of the ribbon's clip, so it can
            never hang past the ribbon's edge however the curve is tuned. */}
        <span className="blade-ribbon absolute inset-0">
          <span className="blade-skin blade-skin--gloss absolute inset-0" />
        </span>
      </span>

      <button
        type="button"
        className="blade-tab relative flex shrink-0 grow-0 basis-[var(--tab-w)] cursor-pointer items-start border-0 bg-transparent pt-[21.5dvh] pb-8 text-[clamp(1.05rem,3.2dvh,2.1rem)] font-normal tracking-[0.01em]"
        onClick={onOpen}
        aria-current={state === "open" ? "true" : undefined}
      >
        <TabLabel id={blade.id} label={blade.label} side={side} />
      </button>

      <div
        className="blade-inner grid min-w-0 flex-auto grid-rows-[auto_minmax(0,1fr)_auto_auto]"
        inert={state === "tab"}
      >
        <header className="blade-head flex min-h-[clamp(64px,12.5dvh,128px)] items-end px-[var(--pad)] pb-[clamp(8px,1.6dvh,16px)]">
          <h2 className="m-0 text-[clamp(1.5rem,4.1dvh,2.6rem)] leading-none font-normal tracking-[0.01em]">
            {blade.title}
          </h2>
        </header>

        {/* A container, so content columns respond to the blade's own width
            rather than the viewport — the rails eat 300px or more. */}
        <div className="blade-scroll @container flex min-h-0 flex-col overflow-y-auto px-[var(--pad)] pt-[clamp(8px,1.4dvh,14px)] pb-[clamp(8px,1.6dvh,16px)] text-[clamp(1.02rem,3dvh,1.7rem)] leading-[1.4] [overscroll-behavior:contain]">
          <BladeMenu
            bladeId={blade.id}
            items={blade.menu}
            feature={blade.feature}
            active={state === "open"}
            onSubmenuChange={setSubmenu}
          />
        </div>

        <div className="px-[var(--pad)] pb-[clamp(10px,3.4dvh,28px)] text-[clamp(1.02rem,3dvh,1.7rem)] leading-[1.4]">
          {blade.tray ? <TrayPill tray={blade.tray} /> : null}
        </div>

        {/* The controller legend, on the raised sheet the dashboard ended every
            blade with. Y is live here: it flips day and night. */}
        <footer className="blade-legend flex min-h-[clamp(64px,16.5dvh,170px)] items-center justify-between gap-4 px-[var(--pad)] pt-[clamp(8px,1.6dvh,16px)] pb-[clamp(12px,3.4dvh,32px)] text-[clamp(0.9rem,2.6dvh,1.4rem)]">
          <span className="flex items-center gap-[clamp(14px,2vw,28px)]">
            <LegendStack
              side="left"
              entries={[{ button: "y", label: theme === "dark" ? "Day" : "Night", onClick: toggleTheme }]}
            />
            <span className="legend-hint inline-flex items-center gap-1.5 text-[0.78em] max-[560px]:hidden">
              <kbd className="legend-key font-ui grid h-[1.4em] min-w-[1.4em] place-items-center rounded-[5px] px-1 leading-none">
                ←
              </kbd>
              <kbd className="legend-key font-ui grid h-[1.4em] min-w-[1.4em] place-items-center rounded-[5px] px-1 leading-none">
                →
              </kbd>
              {submenu ? "Choose" : "Change blade"}
            </span>
          </span>

          <span className="inline-flex items-center gap-[clamp(14px,2vw,28px)]">
            <button
              type="button"
              className="legend-toggle inline-flex cursor-pointer items-center gap-[0.4em] border-0 bg-transparent p-0 text-inherit"
              onClick={() => send({ op: "close" })}
            >
              Hide menu
              <span className="pad-chip pad-chip--guide legend-guide" aria-label="Guide" />
            </button>
            <LegendStack
              side="right"
              entries={[
                { button: "b", label: "Back" },
                { button: "a", label: "Select" }
              ]}
            />
          </span>
        </footer>
      </div>
    </section>
  );
}
