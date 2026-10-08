import { useCallback, useEffect, useRef, useState } from "react";
import { Blade, BladeDef } from "./blade";
import { GlyphDefs } from "./glyph";
import { isDashboardKey } from "./keys";
import { MobileMenu } from "./mobile-menu";
import { getPreferences } from "./preferences";
import { loadSounds, playSound, switchSound } from "./sound";
import { toggleTheme } from "./theme";
import { getNative, send } from "../native";
import { Toast } from "./toast";
import { AllGames } from "./all-games";
import { FolderEditor } from "./folder-editor";

/**
 * The dashboard's mechanism, and only the mechanism: which blade is open, how
 * you get to another one, and where the rest stack. Blades you have passed fold
 * down the left edge, blades ahead down the right.
 */
export function BladeDeck({ blades }: { blades: BladeDef[] }) {
  const [active, setActive] = useState(0);
  const activeRef = useRef(0);
  activeRef.current = active;

  const open = useCallback(
    (index: number, updateHash = true) => {
      const next = Math.max(0, Math.min(index, blades.length - 1));
      // A blade sliding open swooshes; pushing past either end bumps.
      playSound(next === activeRef.current ? "bump" : switchSound(next));
      setActive(next);
      if (updateHash && window.location.hash.replace("#", "").split("/")[0] !== blades[next].id) {
        history.replaceState(null, "", `#${blades[next].id}`);
      }
    },
    [blades]
  );

  // Preferences apply from the first render on (theme, motion), and the
  // sounds load up front only for people who have turned them on.
  useEffect(() => {
    if (getPreferences().sound === "on") void loadSounds();
  }, []);

  // Choosing a row, a pane button, the tray or a link inside a pane clicks.
  // Tabs and the day/night button have their own sounds.
  useEffect(() => {
    const onClick = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest(".menu-row, .pane-button, .tray, .pane-link")) {
        playSound("select");
      }
    };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, []);

  // Deep links land on the right blade.
  useEffect(() => {
    let first = true;
    const fromHash = () => {
      const id = window.location.hash.replace("#", "").split("/")[0];
      const index = blades.findIndex((blade) => blade.id === id);
      if (index >= 0) {
        // A link inside the page (Support the work → #support) switches
        // blades; the initial deep link on page load stays quiet.
        if (!first && index !== activeRef.current) playSound(switchSound(index));
        setActive(index);
      }
      first = false;
    };
    fromHash();
    window.addEventListener("hashchange", fromHash);
    return () => window.removeEventListener("hashchange", fromHash);
  }, [blades]);

  // The open blade's hue washes the background field behind the deck.
  useEffect(() => {
    document.documentElement.dataset.tone = blades[active].tone;
  }, [active, blades]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      // The key that opened the dashboard closes it too.
      const dashboardKey = String(getNative().cvars.preferences_key?.value ?? "F6");
      if (dashboardKey !== "none" && event.key === dashboardKey) {
        event.preventDefault();
        send({ op: "close" });
        return;
      }
      if (!isDashboardKey(event)) return;
      // An open sub-menu has ← and → for choosing (blade-menu.tsx).
      const inSubmenu = document.documentElement.dataset.submenu === "on";
      if (inSubmenu && (event.key === "ArrowLeft" || event.key === "ArrowRight")) return;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        open(active - 1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        open(active + 1);
      } else if (event.key === "Escape" && !inSubmenu && !event.repeat) {
        // The keyboard has no Guide button: Escape at the top hides the
        // dashboard (in a sub-menu it goes back, like B).
        event.preventDefault();
        playSound("back");
        send({ op: "close" });
      } else if ((event.key === "b" || event.key === "B") && !inSubmenu && !event.repeat) {
        // B at the top of a blade goes back to Home; on Home there is nowhere
        // further back.
        event.preventDefault();
        if (active === 0) playSound("bump");
        else open(0);
      } else if ((event.key === "y" || event.key === "Y") && !event.repeat) {
        // Y on the legend is day/night, so the Y key is too. Holding it
        // down flips once, not on every key repeat.
        toggleTheme();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, open]);

  return (
    <>
      <GlyphDefs />
      <Toast />

      {/* Below the rails' breakpoint the blades are a hamburger menu. */}
      <MobileMenu blades={blades} active={active} onOpen={open} />

      <div className="deck relative z-1 flex h-dvh items-stretch px-[var(--field-inset)] py-0">
        {blades.map((blade, index) => (
          <Blade
            key={blade.id}
            blade={blade}
            state={index === active ? "open" : "tab"}
            side={
              index === active ? "center" : index < active ? "left" : "right"
            }
            depth={Math.abs(index - active)}
            onOpen={() => open(index)}
          />
        ))}
      </div>

      <AllGames tone={blades[active].tone} />
      <FolderEditor tone={blades[active].tone} />
    </>
  );
}
