import type { ReactNode } from "react";

/** The faint line under a pane: a caveat, or where a number came from. */
export function Aside({ children }: { children: ReactNode }) {
  return <p className="menu-note m-0 text-[0.78em]">{children}</p>;
}

/** The one action a pane offers, sitting under its text. */
export function PaneAction({
  children,
  onSelect,
  disabled = false
}: {
  children: ReactNode;
  onSelect: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      aria-disabled={disabled || undefined}
      className="pane-button rounded-row mt-[0.3em] inline-flex min-h-[2.2em] cursor-pointer items-center justify-self-start px-[0.9em]"
      onClick={() => {
        if (!disabled) onSelect();
      }}
    >
      {children}
    </button>
  );
}
