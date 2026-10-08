import { StrictMode, useMemo } from "react";
import { createRoot } from "react-dom/client";
import "./globals.css";
import { buildBlades } from "./content/blades";
import { useNative } from "./native";
import { BladeDeck } from "./ui/blade-deck";
import { DashboardField } from "./ui/dashboard-field";
import { startGamepad } from "./ui/gamepad";

function App() {
  const state = useNative();
  const blades = useMemo(() => buildBlades(state), [state]);
  if (!state.connected) return <DashboardField />;
  return (
    <>
      <DashboardField />
      <BladeDeck blades={blades} />
    </>
  );
}

startGamepad();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
