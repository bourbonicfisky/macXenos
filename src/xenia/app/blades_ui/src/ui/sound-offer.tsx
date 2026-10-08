import { useEffect, useRef, useState } from "react";
import { send, useNative, type DashSounds } from "../native";
import { showToast } from "./toast";

/**
 * The first-run offer to download the dashboard's sounds, which the app does
 * not include. A console-style message box over the blades: A downloads in
 * the background, B declines. Either answer is final (dashboard_sounds_asked);
 * System > Dashboard sounds can download them later.
 */

export function packCredit(sounds: DashSounds): string {
  const pack = sounds.pack ?? "a community sound pack";
  const by = sounds.author ? ` by ${sounds.author}` : "";
  const from = sounds.site ? `, from ${sounds.site}` : "";
  return `${pack}${by}${from}`;
}

/** Starts the download and says so. */
export function downloadDashSounds() {
  send({ op: "dashSounds", action: "download" });
  showToast("Dashboard sounds", "Downloading in the background", { sound: false });
}

/** Toasts once a download finishes, wherever it was started. */
function useDownloadNotice(status: DashSounds["status"], error: string) {
  const last = useRef(status);
  useEffect(() => {
    const was = last.current;
    last.current = status;
    if (was !== "downloading") return;
    if (status === "ready") showToast("Dashboard sounds", "Ready");
    else if (status === "failed") showToast("Sound download failed", error || "Try again in System", { sound: false });
  }, [status, error]);
}

export function SoundOffer() {
  const { connected, dashSounds } = useNative();
  const [answered, setAnswered] = useState(false);
  const [focus, setFocus] = useState<0 | 1>(0);
  const buttons = [useRef<HTMLButtonElement>(null), useRef<HTMLButtonElement>(null)];
  useDownloadNotice(dashSounds.status, dashSounds.error);

  const open = connected && !answered && !dashSounds.asked && dashSounds.status === "missing";

  const answer = (download: boolean) => {
    setAnswered(true);
    if (download) downloadDashSounds();
    else send({ op: "dashSounds", action: "decline" });
  };

  useEffect(() => {
    if (!open) return;
    buttons[focus].current?.focus();
  });

  useEffect(() => {
    if (!open) return;
    // Captured before the blades see it: while the box is up, these keys are
    // its own. Others (the dashboard key) pass through.
    const onKey = (event: KeyboardEvent) => {
      const key = event.key;
      let handled = true;
      if (key === "ArrowLeft" || key === "ArrowUp") setFocus(0);
      else if (key === "ArrowRight" || key === "ArrowDown") setFocus(1);
      else if (key === "Enter" || key === " ") answer(focus === 0);
      else if (key === "a" || key === "A") answer(true);
      else if (key === "b" || key === "B" || key === "Escape") answer(false);
      else if (key === "Tab") setFocus(focus === 0 ? 1 : 0);
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
        aria-labelledby="offer-title"
        aria-describedby="offer-text"
        className="offer grid w-full max-w-[34em] gap-[0.8em] rounded-[14px] p-[1.4em_1.6em]"
      >
        <h2 id="offer-title" className="offer-title m-0">
          Dashboard sounds
        </h2>
        <div id="offer-text" className="grid gap-[0.6em]">
          <p className="m-0">
            macXenos doesn&rsquo;t include the Xbox 360 dashboard&rsquo;s sounds. It can download a
            fan-made pack, {packCredit(dashSounds)}, in the background. It&rsquo;s about 1&nbsp;MB.
          </p>
          <p className="offer-note m-0">You can change your mind later in System, under Dashboard sounds.</p>
        </div>
        <div className="mt-[0.4em] flex flex-wrap justify-end gap-[0.8em]">
          <button
            ref={buttons[0]}
            type="button"
            className="offer-button inline-flex min-h-[2.4em] cursor-pointer items-center gap-[0.5em] rounded-[8px] px-[1em]"
            data-focused={focus === 0 || undefined}
            onMouseEnter={() => setFocus(0)}
            onClick={() => answer(true)}
          >
            <span className="pad-key pad-key--a" aria-hidden="true">
              A
            </span>
            Download
          </button>
          <button
            ref={buttons[1]}
            type="button"
            className="offer-button inline-flex min-h-[2.4em] cursor-pointer items-center gap-[0.5em] rounded-[8px] px-[1em]"
            data-focused={focus === 1 || undefined}
            onMouseEnter={() => setFocus(1)}
            onClick={() => answer(false)}
          >
            <span className="pad-key pad-key--b" aria-hidden="true">
              B
            </span>
            No thanks
          </button>
        </div>
      </div>
    </div>
  );
}
