import { useTheme } from "./theme";

/**
 * The app icon, standing where the dashboard put the Xbox 360 logo: the red
 * ring, still lit green in the middle. A slow ember bloom sits behind it and a
 * specular creeps around the edge, so the mark reads as hardware rather than
 * art.
 */
export function AppIcon() {
  const theme = useTheme();

  return (
    <div className="relative grid aspect-square w-[min(17dvh,170px)] max-w-full place-items-center">
      <span className="icon-bloom absolute inset-[4%] rounded-full" aria-hidden="true" />
      <img
        className="icon-art relative h-auto w-[92%]"
        src={
          theme === "dark"
            ? "/macxenos-icon-dark.webp"
            : "/macxenos-icon-light.webp"
        }
        alt="The macXenos app icon: a red ring of light with a green power symbol still glowing at its centre."
        width={768}
        height={768}
        fetchPriority="high"
      />
      {/* The frame clips to the icon; the light turns inside it. */}
      <span
        className="icon-specular-frame pointer-events-none absolute aspect-square w-[92%] overflow-hidden"
        aria-hidden="true"
      >
        <span className="icon-specular absolute inset-[-25%]" />
      </span>
    </div>
  );
}
