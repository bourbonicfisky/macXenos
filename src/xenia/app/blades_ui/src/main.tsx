import { StrictMode, useMemo } from "react";
import { createRoot } from "react-dom/client";
import "./globals.css";
import { buildBlades } from "./content/blades";
import { useNative } from "./native";
import { useLanguage } from "./i18n";
import { BladeDeck } from "./ui/blade-deck";
import { DashboardField } from "./ui/dashboard-field";
import { startGamepad } from "./ui/gamepad";
import { SoundOffer } from "./ui/sound-offer";
import { UpdateOffer } from "./ui/update-offer";

function App() {
  const state = useNative();
  // The blades' text is read when they are built: build them again when the
  // language changes.
  const language = useLanguage();
  const blades = useMemo(() => buildBlades(state), [state, language]);
  if (!state.connected) return <DashboardField />;
  return (
    <>
      <DashboardField />
      <BladeDeck blades={blades} />
      <SoundOffer />
      <UpdateOffer />
    </>
  );
}

startGamepad();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
