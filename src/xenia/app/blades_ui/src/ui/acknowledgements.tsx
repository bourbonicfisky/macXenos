import { useEffect, useState } from "react";
import { openSubBlade, SubBlade, useSubBladeOpen } from "./sub-blade";
import { t } from "../i18n";

/**
 * Acknowledgements: the third-party software macXenos includes and the
 * notices its licenses ask for, on a brushed-metal sub-blade from the right.
 * The list is acknowledgements.json beside the page, which the build collects
 * from docs/acknowledgements/manifest.json (tools/build/collect_licenses.py).
 * The first level lists the components, with the highlighted one's license on
 * a card; A opens its full text, one paragraph a row so ↑ ↓ read through it,
 * and B goes back to the list.
 */

export type Acknowledgement = {
  name: string;
  license: string;
  url: string;
  note: string;
  text: string;
};

export const openAcknowledgements = () => openSubBlade("acknowledgements");

/** What the page reads; a small stand-in on the dev server, which has none. */
async function load(): Promise<Acknowledgement[]> {
  try {
    const response = await fetch("/acknowledgements.json", { cache: "no-store" });
    if (!response.ok) throw new Error(String(response.status));
    const data = (await response.json()) as { components?: Acknowledgement[] };
    return data.components ?? [];
  } catch (error) {
    if (import.meta.env.DEV) {
      const { mockAcknowledgements } = await import("../mock");
      return mockAcknowledgements;
    }
    throw error;
  }
}

const paragraphs = (text: string) =>
  text
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.replace(/\s+$/, ""))
    .filter((paragraph) => paragraph.trim().length > 0);

export function Acknowledgements() {
  const open = useSubBladeOpen("acknowledgements");
  const [components, setComponents] = useState<Acknowledgement[] | null>(null);
  const [failed, setFailed] = useState(false);
  // The component being read, or null for the list.
  const [reading, setReading] = useState<number | null>(null);
  const [highlighted, setHighlighted] = useState(0);

  useEffect(() => {
    if (!open || components) return;
    load()
      .then(setComponents)
      .catch(() => setFailed(true));
  }, [open, components]);

  useEffect(() => {
    if (!open) setReading(null);
  }, [open]);

  const list = components ?? [];
  const current = reading !== null ? list[reading] : undefined;
  const rows = current ? paragraphs(current.text) : [];
  const card = list[highlighted];

  return (
    <SubBlade
      id="acknowledgements"
      side="right"
      tone="metal"
      title={current ? current.name : t("acknowledgements.title")}
      count={current ? rows.length : list.length}
      resetKey={reading === null ? "list" : `read-${reading}`}
      resetFocus={reading === null ? highlighted : 0}
      onActivate={(index) => {
        if (reading === null && list[index]) setReading(index);
      }}
      onFocusChange={(index) => {
        if (reading === null) setHighlighted(index);
      }}
      onBack={() => {
        if (reading === null) return false;
        setReading(null);
        return true;
      }}
      legend={() => ({
        left: [],
        right: current
          ? [{ button: "b", label: t("legend.back") }]
          : [
              { button: "b", label: t("legend.back") },
              { button: "a", label: t("acknowledgements.read") }
            ]
      })}
      empty={failed ? t("acknowledgements.missing") : t("acknowledgements.loading")}
    >
      {(rowProps) =>
        current ? (
          <ul className="subblade-list m-0 grid list-none p-0" aria-label={current.name}>
            {rows.map((paragraph, index) => (
              <li key={index} className="subblade-item">
                <button type="button" {...rowProps(index)}>
                  <span className="subblade-label ack-paragraph whitespace-pre-wrap text-left text-[0.62em] leading-[1.45]">
                    {paragraph}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <div className="grid grid-cols-1 items-start gap-x-[clamp(16px,2.6%,40px)] gap-y-[1em] min-[860px]:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <ul className="subblade-list subblade-menu m-0 grid list-none p-0" aria-label={t("acknowledgements.title")}>
              {list.map((component, index) => (
                <li key={component.name} className="subblade-item">
                  <button type="button" {...rowProps(index)}>
                    <span className="subblade-label truncate">{component.name}</span>
                  </button>
                </li>
              ))}
            </ul>
            {card ? (
              <div className="grid gap-[0.7em] text-[0.82em]">
                <section className="subblade-card">
                  <h3 className="subblade-card-head m-0">{card.name}</h3>
                  <dl className="subblade-card-body m-0 grid grid-cols-[auto_1fr] gap-x-[1em] gap-y-[0.25em]">
                    <dt>{t("acknowledgements.license")}</dt>
                    <dd className="m-0 text-right">{card.license}</dd>
                  </dl>
                </section>
                {card.note ? (
                  <section className="subblade-card">
                    <p className="subblade-card-body m-0">{card.note}</p>
                  </section>
                ) : null}
                <section className="subblade-card">
                  <p className="subblade-card-body m-0 [overflow-wrap:anywhere]">{card.url}</p>
                </section>
                <section className="subblade-card subblade-card--tall">
                  <p className="subblade-card-body m-0">{t("acknowledgements.readHint")}</p>
                </section>
              </div>
            ) : null}
          </div>
        )
      }
    </SubBlade>
  );
}
