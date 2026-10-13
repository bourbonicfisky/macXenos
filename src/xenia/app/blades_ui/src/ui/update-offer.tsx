import { useEffect, useRef, useState } from "react";
import { send, useNative } from "../native";
import type { Option } from "./settings-controls";
import { t } from "../i18n";

/**
 * The first-launch question about update checks (updater_mac.mm). A
 * console-style message box over the blades, after the dashboard-sounds offer
 * has had its turn: ↑ ↓ pick a schedule, A chooses it, B means never. Either
 * answer is final (check_for_updates_asked); System > Software update changes
 * it later. Until it is answered the app does not look for updates by itself.
 */

type Schedule = "launch" | "weekly" | "monthly" | "never";

/** When the app looks for a newer version (check_for_updates). */
export function updateSchedule(): Option[] {
  return [
    {
      value: "launch",
      label: t("dashboard.update.launch"),
      short: t("dashboard.update.launchShort"),
      recommended: true
    },
    { value: "weekly", label: t("dashboard.update.weekly"), short: t("dashboard.update.weeklyShort") },
    { value: "monthly", label: t("dashboard.update.monthly"), short: t("dashboard.update.monthlyShort") },
    { value: "never", label: t("dashboard.update.never") }
  ];
}

export function UpdateOffer() {
  const { connected, updater, dashSounds } = useNative();
  const [answered, setAnswered] = useState(false);
  const [focus, setFocus] = useState(0);
  const options = updateSchedule();
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);

  // The sounds offer goes first; this one waits until it is answered.
  const soundsPending = !dashSounds.asked && dashSounds.status === "missing";
  const open = connected && !answered && updater.available && !updater.asked && !soundsPending;

  const answer = (schedule: Schedule) => {
    setAnswered(true);
    send({ op: "updateOffer", schedule });
  };

  useEffect(() => {
    if (!open) return;
    buttons.current[focus]?.focus();
  });

  useEffect(() => {
    if (!open) return;
    // Captured before the blades see it: while the box is up, these keys are
    // its own. Others (the dashboard key) pass through.
    const onKey = (event: KeyboardEvent) => {
      const key = event.key;
      let handled = true;
      if (key === "ArrowUp" || key === "ArrowLeft") setFocus(Math.max(0, focus - 1));
      else if (key === "ArrowDown" || key === "ArrowRight") {
        setFocus(Math.min(options.length - 1, focus + 1));
      } else if (key === "Enter" || key === " " || key === "a" || key === "A") {
        answer(options[focus].value as Schedule);
      } else if (key === "b" || key === "B" || key === "Escape") answer("never");
      else if (key === "Tab") setFocus((focus + 1) % options.length);
      else if (key === "x" || key === "y" || key === "X" || key === "Y") {
        // Nothing here, and not for the blades underneath either.
      } else handled = false;
      if (handled) {
        event.preventDefault();
        event.stopImmediatePropagation();
      }
    };
    window.addEventListener("keydown", onKey, { capture: true });
    return () => window.removeEventListener("keydown", onKey, { capture: true });
  });

  if (!open) return null;
  return (
    <div className="offer-backdrop fixed inset-0 z-50 grid place-items-center p-4">
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="update-offer-title"
        aria-describedby="update-offer-text"
        className="offer grid w-full max-w-[34em] gap-[0.8em] rounded-[14px] p-[1.4em_1.6em]"
      >
        <h2 id="update-offer-title" className="offer-title m-0">
          {t("updateOffer.title")}
        </h2>
        <div id="update-offer-text" className="grid gap-[0.6em]">
          <p className="m-0">{t("updateOffer.detail")}</p>
          <p className="offer-note m-0">{t("updateOffer.later")}</p>
        </div>
        <div className="mt-[0.4em] grid gap-[0.5em]">
          {options.map((option, index) => (
            <button
              key={String(option.value)}
              ref={(element) => {
                buttons.current[index] = element;
              }}
              type="button"
              className="offer-button inline-flex min-h-[2.4em] cursor-pointer items-center gap-[0.5em] rounded-[8px] px-[1em]"
              data-focused={focus === index || undefined}
              onMouseEnter={() => setFocus(index)}
              onClick={() => answer(option.value as Schedule)}
            >
              {focus === index ? (
                <span className="pad-key pad-key--a" aria-hidden="true">
                  A
                </span>
              ) : option.value === "never" ? (
                <span className="pad-key pad-key--b" aria-hidden="true">
                  B
                </span>
              ) : null}
              {option.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
