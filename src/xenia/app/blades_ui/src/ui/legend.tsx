import type { ReactNode } from "react";

/**
 * The controller legend's face buttons, stacked as the 360 dashboard stacked
 * them: Y above X at the left edge, B above A at the right, each orb lined up
 * in a column on the outside with its label towards the middle. Only the
 * buttons that do something on the current screen are listed.
 */

export type FaceButton = "a" | "b" | "x" | "y";

export type LegendEntry = {
  button: FaceButton;
  label: ReactNode;
  /** Clicking the entry does what the button does. */
  onClick?: () => void;
};

export function LegendStack({ side, entries }: { side: "left" | "right"; entries: LegendEntry[] }) {
  if (entries.length === 0) return null;
  return (
    <span className={`legend-stack legend-stack--${side} grid gap-y-[0.2em]`}>
      {entries.map((entry) => {
        const orb = (
          <span className={`pad-key pad-key--${entry.button}`} aria-hidden="true">
            {entry.button.toUpperCase()}
          </span>
        );
        const content =
          side === "left" ? (
            <>
              {orb}
              {entry.label}
            </>
          ) : (
            <>
              {entry.label}
              {orb}
            </>
          );
        const className = `legend-entry inline-flex items-center gap-[0.4em]${
          side === "right" ? " justify-self-end" : ""
        }`;
        return entry.onClick ? (
          <button
            key={entry.button}
            type="button"
            className={`${className} legend-toggle cursor-pointer border-0 bg-transparent p-0 text-inherit`}
            onClick={entry.onClick}
            aria-keyshortcuts={entry.button.toUpperCase()}
          >
            {content}
          </button>
        ) : (
          <span key={entry.button} className={className}>
            {content}
          </span>
        );
      })}
    </span>
  );
}
