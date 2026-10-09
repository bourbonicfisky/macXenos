import type { Profile } from "../native";
import { t } from "../i18n";

/**
 * A profile's gamercard, as the Xbox 360 dashboard showed it at the head of
 * the Xbox LIVE blade: the gamertag on a lighter band, the gamer picture,
 * and Games, Gamerscore and Achievements down the side. The picture is a
 * placeholder silhouette until profiles carry pictures.
 */
export function Gamercard({ profile }: { profile: Profile }) {
  const { stats } = profile;
  const value = (n: number | undefined) => (n === undefined ? t("profiles.card.unknown") : String(n));
  return (
    <section className="gamercard w-full" aria-label={t("profiles.card.label", { gamertag: profile.gamertag })}>
      <h3 className="gamercard-head m-0 truncate">{profile.gamertag}</h3>
      <div className="gamercard-body grid grid-cols-[auto_minmax(0,1fr)] items-center gap-x-[0.8em]">
        <GamerPicture />
        <dl className="gamercard-stats m-0 grid grid-cols-[minmax(0,1fr)_auto] gap-x-[1em]">
          <dt>{t("profiles.card.games")}</dt>
          <dd className="m-0 text-right">{value(stats?.games)}</dd>
          <dt className="inline-flex items-center gap-[0.25em]">
            <span className="gamerscore-g" aria-hidden="true">
              G
            </span>
            {t("profiles.card.gamerscore")}
          </dt>
          <dd className="m-0 text-right">{value(stats?.gamerscore)}</dd>
          <dt>{t("profiles.card.achievements")}</dt>
          <dd className="m-0 text-right">{value(stats?.achievements)}</dd>
        </dl>
      </div>
      {stats ? null : <p className="gamercard-note m-0">{t("profiles.card.signIn")}</p>}
    </section>
  );
}

/** The placeholder gamer picture: a head and shoulders on a dark tile. */
function GamerPicture() {
  return (
    <svg className="gamercard-picture" viewBox="0 0 64 64" role="img" aria-label={t("profiles.card.picture")}>
      <defs>
        <linearGradient id="gamerpic-tile" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#5b5f66" />
          <stop offset="1" stopColor="#24262b" />
        </linearGradient>
        <linearGradient id="gamerpic-figure" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#e9ebef" />
          <stop offset="1" stopColor="#a9aeb8" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="5" fill="url(#gamerpic-tile)" />
      <circle cx="32" cy="25" r="11" fill="url(#gamerpic-figure)" />
      <path d="M12 60c1.5-12 9.5-19 20-19s18.5 7 20 19z" fill="url(#gamerpic-figure)" />
    </svg>
  );
}
