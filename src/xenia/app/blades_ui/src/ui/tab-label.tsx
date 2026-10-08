import { useEffect, useRef, useState } from "react";

type Geometry = {
  width: number;
  height: number;
  centre: number;
  bow: number;
  overrun: number;
};

/**
 * A tab label that bends with its ribbon. The ribbon's edge is the cubic in
 * blade.css — top to bottom, both control points --bow-ctl toward screen
 * centre at 25% and 75% of the height — so the label runs along that same
 * curve, shifted sideways to the middle of the visible ribbon.
 *
 * SVG paths take pixels, so the tab is measured once mounted. Until then (and
 * in the server HTML) the straight label stands in; it stays afterwards as the
 * button's accessible name.
 */
export function TabLabel({
  id,
  label,
  side
}: {
  id: string;
  label: string;
  side: "left" | "center" | "right";
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const [geometry, setGeometry] = useState<Geometry | null>(null);

  useEffect(() => {
    const tab = ref.current?.parentElement;
    if (!tab) return;

    const measure = () => {
      const style = getComputedStyle(tab);
      const width = tab.clientWidth;
      // The label already centres between the tab's paddings, which keep it
      // clear of the neighbouring ribbon; the curve starts from that line.
      const centre =
        (parseFloat(style.paddingLeft) + width - parseFloat(style.paddingRight)) / 2;
      setGeometry({
        width,
        height: tab.clientHeight,
        centre,
        bow: parseFloat(style.getPropertyValue("--bow")) || 0,
        overrun: parseFloat(style.getPropertyValue("--overrun")) || 0
      });
    };

    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(tab);
    return () => observer.disconnect();
    // Which rail a tab sits in changes its paddings, not its size.
  }, [side]);

  const pathId = `tab-path-${id}`;
  let path = "";
  if (geometry) {
    // The ribbon's curve runs --overrun past both ends of the tab (see
    // tokens.css), so the path does too.
    const { height: h, centre: c, bow, overrun: o } = geometry;
    const lean = side === "right" ? -1 : 1;
    const ctl = c + lean * bow * (4 / 3);
    path = `M ${c} ${-o} C ${ctl} ${h * 0.25 - o / 2}, ${ctl} ${h * 0.75 + o / 2}, ${c} ${h + o}`;
  }

  return (
    <>
      <span
        ref={ref}
        className={`tab-label [writing-mode:vertical-rl] whitespace-nowrap${
          geometry ? " sr-only" : ""
        }`}
      >
        {label}
      </span>
      {geometry ? (
        <svg
          className="tab-curve pointer-events-none absolute inset-0 overflow-visible"
          width={geometry.width}
          height={geometry.height}
          aria-hidden="true"
        >
          <path id={pathId} d={path} fill="none" />
          {/* startOffset matches the straight label's 21.5dvh drop. The path
              starts --overrun above the tab and is all but vertical, so
              distance along it is height plus that overrun. */}
          <text dominantBaseline="central">
            <textPath href={`#${pathId}`} startOffset={geometry.height * 0.215 + geometry.overrun}>
              {label}
            </textPath>
          </text>
        </svg>
      ) : null}
    </>
  );
}
