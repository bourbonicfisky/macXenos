import type { ReactNode } from "react";

/**
 * The small silver pictograms the dashboard set beside every menu row. Bodies
 * fill with the shared #glyph-sheen gradient (declared once in BladeDeck);
 * `.glyph-cut` marks the recessed details punched into them.
 */

export type GlyphName =
  | "mac"
  | "console"
  | "disc"
  | "heart"
  | "star"
  | "blocks"
  | "layers"
  | "plug"
  | "split"
  | "branch"
  | "target"
  | "chip"
  | "shield"
  | "bolt"
  | "play"
  | "doc"
  | "people"
  | "globe"
  | "theme"
  | "motion"
  | "sound"
  | "keys"
  | "bell"
  | "question"
  | "pad"
  | "monitor"
  | "gear"
  | "clock"
  | "folder"
  | "sliders"
  | "wrench"
  | "speed"
  | "resolution"
  | "eye";

const shapes: Record<GlyphName, ReactNode> = {
  // A controller: two grips under one body, a d-pad cut on the left and the
  // face buttons on the right.
  pad: (
    <>
      <path d="M7.2 6.2h9.6c2.4 0 3.9 1.6 4.5 4l1.2 5.3c.5 2.2-.7 3.9-2.4 3.9-1.2 0-2-.7-2.8-1.8l-1.2-1.6H7.9l-1.2 1.6c-.8 1.1-1.6 1.8-2.8 1.8-1.7 0-2.9-1.7-2.4-3.9l1.2-5.3c.6-2.4 2.1-4 4.5-4Z" />
      <path className="glyph-cut" d="M6.4 9.3h1.5v1.6h1.6v1.5H7.9V14H6.4v-1.6H4.8v-1.5h1.6Z" />
      <circle className="glyph-cut" cx="16.6" cy="9.9" r="0.95" />
      <circle className="glyph-cut" cx="18.4" cy="11.7" r="0.95" />
      <circle className="glyph-cut" cx="14.8" cy="11.7" r="0.95" />
      <circle className="glyph-cut" cx="16.6" cy="13.5" r="0.95" />
    </>
  ),
  monitor: (
    <>
      <rect x="2" y="3.8" width="20" height="13" rx="1.8" />
      <rect className="glyph-cut" x="4" y="5.8" width="16" height="9" rx="0.6" />
      <path d="M9.5 17.6h5l.8 2.6H8.7Z" />
      <rect x="6.8" y="19.8" width="10.4" height="1.6" rx="0.8" />
    </>
  ),
  gear: (
    <>
      <path d="M10.4 2.2h3.2l.5 2.6c.7.2 1.4.5 2 .9l2.2-1.5 2.3 2.3-1.5 2.2c.4.6.7 1.3.9 2l2.6.5v3.2l-2.6.5c-.2.7-.5 1.4-.9 2l1.5 2.2-2.3 2.3-2.2-1.5c-.6.4-1.3.7-2 .9l-.5 2.6h-3.2l-.5-2.6c-.7-.2-1.4-.5-2-.9l-2.2 1.5-2.3-2.3 1.5-2.2c-.4-.6-.7-1.3-.9-2l-2.6-.5v-3.2l2.6-.5c.2-.7.5-1.4.9-2L4.4 6.5l2.3-2.3 2.2 1.5c.6-.4 1.3-.7 2-.9Z" />
      <circle className="glyph-cut" cx="12" cy="12" r="3.3" />
    </>
  ),
  clock: (
    <>
      <circle cx="12" cy="12" r="9.5" />
      <path
        className="glyph-cut-line"
        d="M12 6.6V12l3.6 2.4"
        fill="none"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </>
  ),
  folder: (
    <>
      <path d="M2.4 6.4c0-1 .8-1.8 1.8-1.8h4.9l2 2.2h8.7c1 0 1.8.8 1.8 1.8v9.6c0 1-.8 1.8-1.8 1.8H4.2c-1 0-1.8-.8-1.8-1.8Z" />
      <path className="glyph-cut" d="M2.4 9.6h19.2v1.2H2.4Z" />
    </>
  ),
  sliders: (
    <>
      <path
        className="glyph-line"
        d="M3 6.5h18M3 12h18M3 17.5h18"
        fill="none"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <circle cx="8" cy="6.5" r="2.6" />
      <circle cx="15.5" cy="12" r="2.6" />
      <circle cx="10" cy="17.5" r="2.6" />
    </>
  ),
  wrench: (
    <>
      <path d="M15.2 2.6a5.6 5.6 0 0 1 2.6.6l-3.4 3.4.6 2.9 2.9.6 3.4-3.4a5.6 5.6 0 0 1-7 7.3l-7.4 7.4a2.3 2.3 0 0 1-3.3-3.3l7.4-7.4a5.6 5.6 0 0 1 4.2-8.1Z" />
      <circle className="glyph-cut" cx="5.2" cy="18.8" r="1.1" />
    </>
  ),
  speed: (
    <>
      <path d="M12 4.2a9.8 9.8 0 0 1 9.8 9.8c0 1.6-.4 3.1-1.1 4.4H3.3A9.8 9.8 0 0 1 12 4.2Z" />
      <path
        className="glyph-cut-line"
        d="M12 14.6l4.4-5"
        fill="none"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
      <circle className="glyph-cut" cx="12" cy="14.6" r="1.6" />
    </>
  ),
  resolution: (
    <>
      <rect x="2" y="4" width="20" height="16" rx="1.8" />
      <path className="glyph-cut" d="M4.6 6.6h5v1.6H6.2v3.4H4.6Zm14.8 10.8h-5v-1.6h3.4v-3.4h1.6Z" />
    </>
  ),
  eye: (
    <>
      <path d="M12 5.2c4.6 0 8.2 3 10 6.8-1.8 3.8-5.4 6.8-10 6.8S3.8 15.8 2 12c1.8-3.8 5.4-6.8 10-6.8Z" />
      <circle className="glyph-cut" cx="12" cy="12" r="3.4" />
    </>
  ),
  question: (
    <>
      <circle cx="12" cy="12" r="9.5" />
      <path
        className="glyph-cut-line"
        d="M9.4 9.4a2.7 2.7 0 1 1 3.9 2.4c-.8.4-1.3 1-1.3 1.9v.4"
        fill="none"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
      <circle className="glyph-cut" cx="12" cy="17.3" r="1.35" />
    </>
  ),
  bell: (
    <>
      <path d="M12 2.4a1.6 1.6 0 0 1 1.6 1.6v.7a6.2 6.2 0 0 1 4.6 6v4.1l2 2.6v1.2H3.8v-1.2l2-2.6v-4.1a6.2 6.2 0 0 1 4.6-6V4A1.6 1.6 0 0 1 12 2.4Z" />
      <path d="M9.5 20.1h5a2.5 2.5 0 0 1-5 0Z" />
    </>
  ),
  theme: (
    <>
      <circle cx="12" cy="12" r="9.5" />
      <path className="glyph-cut" d="M12 4.6a7.4 7.4 0 0 1 0 14.8Z" />
    </>
  ),
  motion: (
    <>
      <path d="M3.5 5.2 11 12l-7.5 6.8v-3.9L7.2 12 3.5 9.1Z" />
      <path d="M12 5.2 19.5 12 12 18.8v-3.9l3.7-2.9L12 9.1Z" />
    </>
  ),
  sound: (
    <>
      <path d="M2.8 9h3.9l5.3-4.6v15.2L6.7 15H2.8Z" />
      <path
        className="glyph-line"
        d="M15.2 8.6a4.8 4.8 0 0 1 0 6.8M17.8 6a8.5 8.5 0 0 1 0 12"
        fill="none"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </>
  ),
  keys: (
    <>
      <rect x="1.8" y="6" width="20.4" height="12" rx="2" />
      <path
        className="glyph-cut"
        d="M4.6 8.6h2.2v2.2H4.6Zm3.6 0h2.2v2.2H8.2Zm3.6 0H14v2.2h-2.2Zm3.6 0h2.2v2.2h-2.2Zm-10.8 3.6h2.2v2.2H4.6Zm3.6 0h2.2v2.2H8.2Zm3.6 0H14v2.2h-2.2Zm3.6 0h2.2v2.2h-2.2ZM7 15h10v1.4H7Z"
      />
    </>
  ),
  mac: (
    <>
      <rect x="4" y="4.5" width="16" height="11.5" rx="1.6" />
      <rect className="glyph-cut" x="5.8" y="6.3" width="12.4" height="7.9" rx="0.6" />
      <path d="M1.8 17.6h20.4l-1.6 2.4H3.4Z" />
    </>
  ),
  console: (
    <>
      <rect x="7.5" y="2" width="9" height="20" rx="2.2" />
      <circle className="glyph-cut" cx="12" cy="6" r="1.6" />
      <rect className="glyph-cut" x="10.6" y="10" width="2.8" height="8.5" rx="1" />
    </>
  ),
  disc: (
    <path
      fillRule="evenodd"
      d="M12 2.5a9.5 9.5 0 1 0 0 19 9.5 9.5 0 1 0 0-19Zm0 7a2.5 2.5 0 1 1 0 5 2.5 2.5 0 1 1 0-5Z"
    />
  ),
  heart: (
    <path d="M12 21s-7.5-4.6-9.3-9.2C1.4 8.4 3.6 4.5 7.3 4.5c2 0 3.6 1.1 4.7 2.7 1.1-1.6 2.7-2.7 4.7-2.7 3.7 0 5.9 3.9 4.6 7.3C19.5 16.4 12 21 12 21Z" />
  ),
  star: (
    <path d="m12 2.6 2.9 5.9 6.4.9-4.7 4.5 1.1 6.4L12 17.3l-5.7 3 1.1-6.4-4.7-4.5 6.4-.9Z" />
  ),
  blocks: (
    <>
      <rect x="2.5" y="13" width="8.5" height="8.5" rx="1.3" />
      <rect x="13" y="13" width="8.5" height="8.5" rx="1.3" />
      <rect x="7.75" y="2.5" width="8.5" height="8.5" rx="1.3" />
    </>
  ),
  layers: (
    <>
      <path d="m12 2.5 10 5.2-10 5.2L2 7.7Z" />
      <path d="m4.3 10.9 7.7 4 7.7-4 2.3 1.2-10 5.2-10-5.2Z" />
      <path d="m4.3 15.3 7.7 4 7.7-4 2.3 1.2-10 5.2-10-5.2Z" />
    </>
  ),
  plug: (
    <>
      <rect x="7.4" y="1.8" width="2.4" height="6" rx="1" />
      <rect x="14.2" y="1.8" width="2.4" height="6" rx="1" />
      <path d="M4.5 7.2h15v3.3a7.5 7.5 0 0 1-5.8 7.3v4.4h-3.4v-4.4a7.5 7.5 0 0 1-5.8-7.3Z" />
    </>
  ),
  split: (
    <>
      <circle cx="8.5" cy="12" r="6.5" />
      <circle cx="15.5" cy="12" r="6.5" fillOpacity="0.72" />
    </>
  ),
  branch: (
    <>
      <path
        className="glyph-line"
        d="M6.5 6v12M6.5 14c0-3.5 11-2.5 11-6.5"
        fill="none"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      <circle cx="6.5" cy="5" r="2.8" />
      <circle cx="6.5" cy="19" r="2.8" />
      <circle cx="17.5" cy="6" r="2.8" />
    </>
  ),
  target: (
    <>
      <path
        fillRule="evenodd"
        d="M12 2.5a9.5 9.5 0 1 0 0 19 9.5 9.5 0 1 0 0-19Zm0 3.6a5.9 5.9 0 1 1 0 11.8 5.9 5.9 0 1 1 0-11.8Z"
      />
      <circle cx="12" cy="12" r="3" />
    </>
  ),
  chip: (
    <>
      <path
        className="glyph-line"
        d="M9 2.5v3M12 2.5v3M15 2.5v3M9 18.5v3M12 18.5v3M15 18.5v3M2.5 9h3M2.5 12h3M2.5 15h3M18.5 9h3M18.5 12h3M18.5 15h3"
        fill="none"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <rect x="5.5" y="5.5" width="13" height="13" rx="2.2" />
      <rect className="glyph-cut" x="9" y="9" width="6" height="6" rx="1" />
    </>
  ),
  shield: (
    <>
      <path d="m12 2.3 8.3 3.1v6.1c0 5.1-3.6 8.8-8.3 10.2-4.7-1.4-8.3-5.1-8.3-10.2V5.4Z" />
      <path className="glyph-cut" d="M12 5.2v14c-3.2-1.2-5.6-4-5.6-7.6V7.3Z" />
    </>
  ),
  bolt: <path d="M13.8 2 4.6 13.6h6l-1.7 8.4 9.6-12h-6.1Z" />,
  play: (
    <>
      <rect x="2" y="5" width="20" height="14" rx="4" />
      <path className="glyph-cut" d="M10 8.7v6.6l5.4-3.3Z" />
    </>
  ),
  doc: (
    <>
      <path d="M5.5 2.2h8.8l4.7 4.7v14.9H5.5Z" />
      <path className="glyph-cut" d="M8.2 10.6h7.6v1.5H8.2Zm0 3.4h7.6v1.5H8.2Zm0 3.4h5v1.5h-5Z" />
    </>
  ),
  people: (
    <>
      <circle cx="16.4" cy="7" r="3.1" />
      <path d="M16.8 11.9c3 .2 4.9 2.7 4.9 6.6h-4.3c0-2.5-.9-4.6-2.5-6 .5-.4 1.1-.6 1.9-.6Z" />
      <circle cx="9" cy="7.8" r="3.7" />
      <path d="M2.3 20.5c0-4.1 3-6.9 6.7-6.9s6.7 2.8 6.7 6.9Z" />
    </>
  ),
  globe: (
    <>
      <circle cx="12" cy="12" r="9.5" />
      <path
        className="glyph-cut-line"
        d="M2.8 12h18.4M12 2.8c-3.3 3.4-3.3 15 0 18.4M12 2.8c3.3 3.4 3.3 15 0 18.4M4.4 7h15.2M4.4 17h15.2"
        fill="none"
        strokeWidth="1.3"
      />
    </>
  )
};

export function Glyph({ name }: { name: GlyphName }) {
  return (
    <svg viewBox="0 0 24 24" className="glyph shrink-0" aria-hidden="true">
      {shapes[name]}
    </svg>
  );
}

/** The gradient every glyph body fills with — render once per document. */
export function GlyphDefs() {
  return (
    <svg width="0" height="0" className="absolute" aria-hidden="true">
      <defs>
        <linearGradient id="glyph-sheen" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="0.55" stopColor="#e4e4e9" />
          <stop offset="1" stopColor="#b9b9c2" />
        </linearGradient>
      </defs>
    </svg>
  );
}
